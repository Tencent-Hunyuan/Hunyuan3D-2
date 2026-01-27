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

import numpy as np

def meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = [np.zeros(texture_channel, dtype=np.float32) for _ in range(vtx_num)]
    uncolored_vtxs = []
    G = [[] for _ in range(vtx_num)]

    for i in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
            uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
            if mask[uv_u, uv_v] > 0:
                vtx_mask[vtx_idx] = 1.0
                vtx_color[vtx_idx] = texture[uv_u, uv_v]
            else:
                uncolored_vtxs.append(vtx_idx)
            G[pos_idx[i, k]].append(pos_idx[i, (k + 1) % 3])

    smooth_count = 2
    last_uncolored_vtx_count = 0
    while smooth_count > 0:
        uncolored_vtx_count = 0
        for vtx_idx in uncolored_vtxs:
            sum_color = np.zeros(texture_channel, dtype=np.float32)
            total_weight = 0.0
            vtx_0 = vtx_pos[vtx_idx]
            for connected_idx in G[vtx_idx]:
                if vtx_mask[connected_idx] > 0:
                    vtx1 = vtx_pos[connected_idx]
                    dist = np.sqrt(np.sum((vtx_0 - vtx1) ** 2))
                    dist_weight = 1.0 / max(dist, 1e-4)
                    dist_weight *= dist_weight
                    sum_color += vtx_color[connected_idx] * dist_weight
                    total_weight += dist_weight
            if total_weight > 0:
                vtx_color[vtx_idx] = sum_color / total_weight
                vtx_mask[vtx_idx] = 1.0
            else:
                uncolored_vtx_count += 1

        if last_uncolored_vtx_count == uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_vtx_count

    new_texture = texture.copy()
    new_mask = mask.copy()
    for face_idx in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[face_idx, k]
            vtx_idx = pos_idx[face_idx, k]
            if vtx_mask[vtx_idx] == 1.0:
                uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
                uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
                new_texture[uv_u, uv_v] = vtx_color[vtx_idx]
                new_mask[uv_u, uv_v] = 255
    return new_texture, new_mask

def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    if method == "smooth":
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
    else:
        raise ValueError("Invalid method. Use 'smooth' or 'forward'.")