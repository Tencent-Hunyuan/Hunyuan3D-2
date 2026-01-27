# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from .camera_utils import (
    transform_pos,
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
)
from .mesh_processor import meshVerticeInpaint
from .mesh_utils import load_mesh, save_mesh


def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))


def scatter_add_nd_with_count(input, count, indices, values, weights=None):
    # input: [..., C], D dimension + C channel
    # count: [..., 1], D dimension
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    count = count.view(-1, 1)

    flatten_indices = (indices * torch.tensor(stride,
                                              dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    if weights is None:
        weights = torch.ones_like(values[..., :1])

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)


def linear_grid_put_2d(H, W, coords, values, return_count=False):
    # coords: [N, 2], float in [0, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = coords * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device,
                         dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device,
                        dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(
        result, count, indices_00, values * w_00.unsqueeze(1), weights * w_00.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_01, values * w_01.unsqueeze(1), weights * w_01.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_10, values * w_10.unsqueeze(1), weights * w_10.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_11, values * w_11.unsqueeze(1), weights * w_11.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


class MeshRender():
    def __init__(
        self,
        camera_distance=1.45, camera_type='orth',
        default_resolution=1024, texture_size=1024,
        use_antialias=True, max_mip_level=None, filter_mode='linear',
        bake_mode='linear', raster_mode='cr', device='cuda'):

        self.device = device

        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)

        self.camera_distance = camera_distance
        self.use_antialias = use_antialias
        self.max_mip_level = max_mip_level
        self.filter_mode = filter_mode

        self.bake_angle_thres = 75
        self.bake_unreliable_kernel_size = int(
            (2 / 512) * max(self.default_resolution[0], self.default_resolution[1]))
        self.bake_mode = bake_mode

        self.raster_mode = raster_mode
        if self.raster_mode == 'cr':
            import custom_rasterizer as cr
            self.raster = cr
        else:
            raise f'No raster named {self.raster_mode}'

        if camera_type == 'orth':
            self.ortho_scale = 1.2
            self.camera_proj_mat = get_orthographic_projection_matrix(
                left=-self.ortho_scale * 0.5, right=self.ortho_scale * 0.5,
                bottom=-self.ortho_scale * 0.5, top=self.ortho_scale * 0.5,
                near=0.1, far=100
            )
        elif camera_type == 'perspective':
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13, self.default_resolution[1] / self.default_resolution[0],
                0.01, 100.0
            )
        else:
            raise f'No camera type {camera_type}'

    def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):

        if self.raster_mode == 'cr':
            rast_out_db = None
            if pos.dim() == 2:
                pos = pos.unsqueeze(0)
            findices, barycentric = self.raster.rasterize(pos, tri, resolution)
            rast_out = torch.cat((barycentric, findices.unsqueeze(-1)), dim=-1)
            rast_out = rast_out.unsqueeze(0)
        else:
            raise f'No raster named {self.raster_mode}'

        return rast_out, rast_out_db

    def raster_interpolate(self, uv, rast_out, uv_idx, rast_db=None, diff_attrs=None):

        if self.raster_mode == 'cr':
            textd = None
            barycentric = rast_out[0, ..., :-1]
            findices = rast_out[0, ..., -1]
            if uv.dim() == 2:
                uv = uv.unsqueeze(0)
            textc = self.raster.interpolate(uv, findices, barycentric, uv_idx)
        else:
            raise f'No raster named {self.raster_mode}'

        return textc, textd

    def raster_texture(self, tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto',
                       boundary_mode='wrap', max_mip_level=None):

        if self.raster_mode == 'cr':
            raise f'Texture is not implemented in cr'
        else:
            raise f'No raster named {self.raster_mode}'

        return color

    def raster_antialias(self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):

        if self.raster_mode == 'cr':
            # Antialias has not been supported yet
            color = color
        else:
            raise f'No raster named {self.raster_mode}'

        return color

    def load_mesh(
        self,
        mesh,
        scale_factor=1.15,
        auto_center=True,
    ):
        vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data = load_mesh(mesh)
        self.mesh_copy = mesh
        self.set_mesh(vtx_pos, pos_idx,
                      vtx_uv=vtx_uv, uv_idx=uv_idx,
                      scale_factor=scale_factor, auto_center=auto_center
                      )
        if texture_data is not None:
            self.set_texture(texture_data)

    def save_mesh(self):
        texture_data = self.get_texture()
        texture_data = Image.fromarray((texture_data * 255).astype(np.uint8))
        return save_mesh(self.mesh_copy, texture_data)

    def set_mesh(
        self,
        vtx_pos, pos_idx,
        vtx_uv=None, uv_idx=None,
        scale_factor=1.15, auto_center=True
    ):

        self.vtx_pos = torch.from_numpy(vtx_pos).to(self.device).float()
        self.pos_idx = torch.from_numpy(pos_idx).to(self.device).to(torch.int)
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv = torch.from_numpy(vtx_uv).to(self.device).float()
            self.uv_idx = torch.from_numpy(uv_idx).to(self.device).to(torch.int)
        else:
            self.vtx_uv = None
            self.uv_idx = None

        self.vtx_pos[:, [0, 1]] = -self.vtx_pos[:, [0, 1]]
        self.vtx_pos[:, [1, 2]] = self.vtx_pos[:, [2, 1]]
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv[:, 1] = 1.0 - self.vtx_uv[:, 1]

        if auto_center:
            max_bb = (self.vtx_pos - 0).max(0)[0]
            min_bb = (self.vtx_pos - 0).min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(self.vtx_pos - center, dim=1).max() * 2.0
            self.vtx_pos = (self.vtx_pos - center) * \
                           (scale_factor / float(scale))
            self.scale_factor = scale_factor

    def set_texture(self, tex):
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex = tex.cpu().numpy()
            tex = Image.fromarray((tex * 255).astype(np.uint8))

        tex = tex.resize(self.texture_size).convert('RGB')
        tex = np.array(tex) / 255.0
        self.tex = torch.from_numpy(tex).to(self.device)
        self.tex = self.tex.float()

    def set_default_render_resolution(self, default_resolution):
        if isinstance(default_resolution, int):
            default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_default_texture_resolution(self, texture_size):
        if isinstance(texture_size, int):
            texture_size = (texture_size, texture_size)
        self.texture_size = texture_size

    def get_mesh(self):
        vtx_pos = self.vtx_pos.cpu().numpy()
        pos_idx = self.pos_idx.cpu().numpy()
        vtx_uv = self.vtx_uv.cpu().numpy()
        uv_idx = self.uv_idx.cpu().numpy()

        # 坐标变换的逆变换
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]

        vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def get_texture(self):
        return self.tex.cpu().numpy()

    def to(self, device):
        self.device = device

        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(self.device))

    def color_rgb_to_srgb(self, image):
        if isinstance(image, Image.Image):
            image_rgb = torch.tesnor(
                np.array(image) /
                255.0).float().to(
                self.device)
        elif isinstance(image, np.ndarray):
            image_rgb = torch.tensor(image).float()
        else:
            image_rgb = image.to(self.device)

        image_srgb = torch.where(
            image_rgb <= 0.0031308,
            12.92 * image_rgb,
            1.055 * torch.pow(image_rgb, 1 / 2.4) - 0.055
        )

        if isinstance(image, Image.Image):
            image_srgb = Image.fromarray(
                (image_srgb.cpu().numpy() *
                 255).astype(
                    np.uint8))
        elif isinstance(image, np.ndarray):
            image_srgb = image_srgb.cpu().numpy()
        else:
            image_srgb = image_srgb.to(image.device)

        return image_srgb

    def _render(
        self,
        glctx,
        mvp,
        pos,
        pos_idx,
        uv,
        uv_idx,
        tex,
        resolution,
        max_mip_level,
        keep_alpha,
        filter_mode
    ):
        pos_clip = transform_pos(mvp, pos)
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            glctx, pos_clip, pos_idx, resolution=resolution)

        tex = tex.contiguous()
        if filter_mode == 'linear-mipmap-linear':
            texc, texd = self.raster_interpolate(
                uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
            color = self.raster_texture(
                tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            texc, _ = self.raster_interpolate(uv[None, ...], rast_out, uv_idx)
            color = self.raster_texture(tex[None, ...], texc, filter_mode=filter_mode)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        color = color * visible_mask  # Mask out background.
        if self.use_antialias:
            color = self.raster_antialias(color, rast_out, pos_clip, pos_idx)

        if keep_alpha:
            color = torch.cat([color, visible_mask], dim=-1)
        return color[0, ...]

    def render(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        tex=None,
        keep_alpha=True,
        bgcolor=None,
        filter_mode=None,
        return_type='th'
    ):

        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        if tex is not None:
            if isinstance(tex, Image.Image):
                tex = torch.tensor(np.array(tex) / 255.0)
            elif isinstance(tex, np.ndarray):
                tex = torch.tensor(tex)
            if tex.dim() == 2:
                tex = tex.unsqueeze(-1)
            tex = tex.float().to(self.device)
        image = self._render(r_mvp, self.vtx_pos, self.pos_idx, self.vtx_uv, self.uv_idx,
                             self.tex if tex is None else tex,
                             self.default_resolution if resolution is None else resolution,
                             self.max_mip_level, True, filter_mode if filter_mode else self.filter_mode)
        mask = (image[..., [-1]] == 1).float()
        if bgcolor is None:
            bgcolor = [0 for _ in range(image.shape[-1] - 1)]
        image = image * mask + (1 - mask) * \
                torch.tensor(bgcolor + [0]).to(self.device)
        if keep_alpha == False:
            image = image[..., :-1]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_normal(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        bg_color=[1, 1, 1],
        use_abs_coor=False,
        normalize_rgb=True,
        return_type='th'
    ):

        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)
        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        if use_abs_coor:
            mesh_triangles = self.vtx_pos[self.pos_idx[:, :3], :]
        else:
            pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
            mesh_triangles = pos_camera[self.pos_idx[:, :3], :]
        face_normals = F.normalize(
            torch.cross(mesh_triangles[:,
                        1,
                        :] - mesh_triangles[:,
                             0,
                             :],
                        mesh_triangles[:,
                        2,
                        :] - mesh_triangles[:,
                             0,
                             :],
                        dim=-1),
            dim=-1)

        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=self.vtx_pos.shape[0],
                                                              faces=self.pos_idx.cpu(),
                                                              face_normals=face_normals.cpu(), )
        vertex_normals = torch.from_numpy(
            vertex_normals).float().to(self.device).contiguous()

        # Interpolate normal values across the rasterized pixels
        normal, _ = self.raster_interpolate(
            vertex_normals[None, ...], rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        normal = normal * visible_mask + \
                 torch.tensor(bg_color, dtype=torch.float32, device=self.device) * (1 -
                                                                                    visible_mask)  

        if normalize_rgb:
            normal = (normal + 1) * 0.5
        if self.use_antialias:
            normal = self.raster_antialias(normal, rast_out, pos_clip, self.pos_idx)

        image = normal[0, ...]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))

        return image

    def convert_normal_map(self, image):
        # blue is front, red is left, green is top
        if isinstance(image, Image.Image):
            image = np.array(image)
        mask = (image == [255, 255, 255]).all(axis=-1)

        image = (image / 255.0) * 2.0 - 1.0

        image[..., [1]] = -image[..., [1]]
        image[..., [1, 2]] = image[..., [2, 1]]
        image[..., [0]] = -image[..., [0]]

        image = (image + 1.0) * 0.5

        image = (image * 255).astype(np.uint8)
        image[mask] = [127, 127, 255]

        return Image.fromarray(image)

    def get_pos_from_mvp(self, elev, azim, camera_distance, center):
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)

        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)

        return pos_camera, pos_clip

    def render_depth(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        return_type='th'
    ):
        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)

        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        tex_depth = pos_camera[:, 2].reshape(1, -1, 1).contiguous()

        # Interpolate depth values across the rasterized pixels
        depth, _ = self.raster_interpolate(tex_depth, rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        depth_max, depth_min = depth[visible_mask >
                                     0].max(), depth[visible_mask > 0].min()
        depth = (depth - depth_min) / (depth_max - depth_min)

        depth = depth * visible_mask  # Mask out background.
        if self.use_antialias:
            depth = self.raster_antialias(depth, rast_out, pos_clip, self.pos_idx)

        image = depth[0, ...]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_position(self, elev, azim, camera_distance=None, center=None,
                        resolution=None, bg_color=[1, 1, 1], return_type='th'):
        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)
        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        tex_position = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
        tex_position = tex_position.contiguous()

        # Interpolate depth values across the rasterized pixels
        position, _ = self.raster_interpolate(
            tex_position[None, ...], rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)

        position = position * visible_mask + \
                   torch.tensor(bg_color, dtype=torch.float32, device=self.device) * (1 -
                                                                                      visible_mask) 
        if self.use_antialias:
            position = self.raster_antialias(position, rast_out, pos_clip, self.pos_idx)

        image = position[0, ...]

        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_uvpos(self, return_type='th'):
        image = self.uv_feature_map(self.vtx_pos * 0.5 + 0.5)
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def uv_feature_map(self, vert_feat, bg=None):
        vtx_uv = self.vtx_uv * 2 - 1.0
        vtx_uv = torch.cat(
            [vtx_uv, torch.zeros_like(self.vtx_uv)], dim=1).unsqueeze(0)
        vtx_uv[..., -1] = 1
        uv_idx = self.uv_idx
        rast_out, rast_out_db = self.raster_rasterize(
            vtx_uv, uv_idx, resolution=self.texture_size)
        feat_map, _ = self.raster_interpolate(vert_feat[None, ...], rast_out, uv_idx)
        feat_map = feat_map[0, ...]
        if bg is not None:
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]
            feat_map[visible_mask == 0] = bg
        return feat_map

    def render_sketch_from_geometry(self, normal_image, depth_image):
        normal_image_np = normal_image.cpu().numpy()
        depth_image_np = depth_image.cpu().numpy()

        normal_image_np = (normal_image_np * 255).astype(np.uint8)
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        normal_image_np = cv2.cvtColor(normal_image_np, cv2.COLOR_RGB2GRAY)

        normal_edges = cv2.Canny(normal_image_np, 80, 150)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)

        combined_edges = np.maximum(normal_edges, depth_edges)

        sketch_image = torch.from_numpy(combined_edges).to(
            normal_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)

        return sketch_image

    def render_sketch_from_depth(self, depth_image):
        depth_image_np = depth_image.cpu().numpy()
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        combined_edges = depth_edges
        sketch_image = torch.from_numpy(combined_edges).to(
            depth_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)
        return sketch_image

    def back_project(self, image, elev, azim,
                     camera_distance=None, center=None, method=None):
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image) / 255.0)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(-1)
        image = image.float().to(self.device)
        resolution = image.shape[:2]
        channel = image.shape[-1]
        texture = torch.zeros(self.texture_size + (channel,)).to(self.device)
        cos_map = torch.zeros(self.texture_size + (1,)).to(self.device)

        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        v0 = pos_camera[self.pos_idx[:, 0], :]
        v1 = pos_camera[self.pos_idx[:, 1], :]
        v2 = pos_camera[self.pos_idx[:, 2], :]
        face_normals = F.normalize(
            torch.cross(
                v1 - v0,
                v2 - v0,
                dim=-1),
            dim=-1)
        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=self.vtx_pos.shape[0],
                                                              faces=self.pos_idx.cpu(),
                                                              face_normals=face_normals.cpu(), )
        vertex_normals = torch.from_numpy(
            vertex_normals).float().to(self.device).contiguous()
        tex_depth = pos_camera[:, 2].reshape(1, -1, 1).contiguous()
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]

        normal, _ = self.raster_interpolate(
            vertex_normals[None, ...], rast_out, self.pos_idx)
        normal = normal[0, ...]
        uv, _ = self.raster_interpolate(self.vtx_uv[None, ...], rast_out, self.uv_idx)
        depth, _ = self.raster_interpolate(tex_depth, rast_out, self.pos_idx)
        depth = depth[0, ...]

        depth_max, depth_min = depth[visible_mask >
                                     0].max(), depth[visible_mask > 0].min()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_image = depth_normalized * visible_mask  # Mask out background.

        sketch_image = self.render_sketch_from_depth(depth_image)

        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = torch.nn.functional.cosine_similarity(
            lookat, normal.view(-1, 3))
        cos_image = cos_image.view(normal.shape[0], normal.shape[1], 1)

        cos_thres = np.cos(self.bake_angle_thres / 180 * np.pi)
        cos_image[cos_image < cos_thres] = 0

        # shrink
        kernel_size = self.bake_unreliable_kernel_size * 2 + 1
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), dtype=torch.float32).to(
            sketch_image.device)

        visible_mask = visible_mask.permute(2, 0, 1).unsqueeze(0).float()
        visible_mask = F.conv2d(
            1.0 - visible_mask,
            kernel,
            padding=kernel_size // 2)
        visible_mask = 1.0 - (visible_mask > 0).float()  # 二值化
        visible_mask = visible_mask.squeeze(0).permute(1, 2, 0)

        sketch_image = sketch_image.permute(2, 0, 1).unsqueeze(0)
        sketch_image = F.conv2d(sketch_image, kernel, padding=kernel_size // 2)
        sketch_image = (sketch_image > 0).float()  # 二值化
        sketch_image = sketch_image.squeeze(0).permute(1, 2, 0)
        visible_mask = visible_mask * (sketch_image < 0.5)

        cos_image[visible_mask == 0] = 0

        method = self.bake_mode if method is None else method

        if method == 'linear':
            proj_mask = (visible_mask != 0).view(-1)
            uv = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
            image = image.squeeze(0).contiguous().view(-1, channel)[proj_mask]
            cos_image = cos_image.contiguous().view(-1, 1)[proj_mask]
            sketch_image = sketch_image.contiguous().view(-1, 1)[proj_mask]

            texture = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], image)
            cos_map = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], cos_image)
            boundary_map = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], sketch_image)
        else:
            raise f'No bake mode {method}'

        return texture, cos_map, boundary_map

    def bake_texture(self, colors, elevs, azims,
                     camera_distance=None, center=None, exp=6, weights=None):
        for i in range(len(colors)):
            if isinstance(colors[i], Image.Image):
                colors[i] = torch.tensor(
                    np.array(
                        colors[i]) / 255.0,
                    device=self.device).float()
        if weights is None:
            weights = [1.0 for _ in range(colors)]
        textures = []
        cos_maps = []
        for color, elev, azim, weight in zip(colors, elevs, azims, weights):
            texture, cos_map, _ = self.back_project(
                color, elev, azim, camera_distance, center)
            cos_map = weight * (cos_map ** exp)
            textures.append(texture)
            cos_maps.append(cos_map)

        texture_merge, trust_map_merge = self.fast_bake_texture(
            textures, cos_maps)
        return texture_merge, trust_map_merge

    @torch.no_grad()
    def fast_bake_texture(self, textures, cos_maps):

        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(
            self.texture_size + (channel,)).to(self.device)
        trust_map_merge = torch.zeros(self.texture_size + (1,)).to(self.device)
        for texture, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            painted_sum = ((cos_map > 0) * (trust_map_merge > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map
        texture_merge = texture_merge / torch.clamp(trust_map_merge, min=1E-8)

        return texture_merge, trust_map_merge > 1E-8

    def uv_inpaint(self, texture, mask):

        if isinstance(texture, torch.Tensor):
            texture_np = texture.cpu().numpy()
        elif isinstance(texture, np.ndarray):
            texture_np = texture
        elif isinstance(texture, Image.Image):
            texture_np = np.array(texture) / 255.0

        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()

        texture_np, mask = meshVerticeInpaint(
            texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)

        texture_np = cv2.inpaint(
            (texture_np *
             255).astype(
                np.uint8),
            255 -
            mask,
            3,
            cv2.INPAINT_NS)

        return texture_np
