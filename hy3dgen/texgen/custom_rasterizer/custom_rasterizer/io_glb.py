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

import base64
import io
import os

import numpy as np
from PIL import Image as PILImage
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R


# Function to extract buffer data
def get_buffer_data(gltf, buffer_view):
    buffer = gltf.buffers[buffer_view.buffer]
    buffer_data = gltf.get_data_from_buffer_uri(buffer.uri)
    byte_offset = buffer_view.byteOffset if buffer_view.byteOffset else 0
    byte_length = buffer_view.byteLength
    return buffer_data[byte_offset:byte_offset + byte_length]


# Function to extract attribute data
def get_attribute_data(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer_data = get_buffer_data(gltf, buffer_view)

    comptype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    dtype = comptype[accessor.componentType]

    t2n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
    num_components = t2n[accessor.type]

    # Calculate the correct slice of data
    byte_offset = accessor.byteOffset if accessor.byteOffset else 0
    byte_stride = buffer_view.byteStride if buffer_view.byteStride else num_components * np.dtype(dtype).itemsize
    count = accessor.count

    # Extract the attribute data
    attribute_data = np.zeros((count, num_components), dtype=dtype)
    for i in range(count):
        start = byte_offset + i * byte_stride
        end = start + num_components * np.dtype(dtype).itemsize
        attribute_data[i] = np.frombuffer(buffer_data[start:end], dtype=dtype)

    return attribute_data


# Function to extract image data
def get_image_data(gltf, image, folder):
    if image.uri:
        if image.uri.startswith('data:'):
            # Data URI
            header, encoded = image.uri.split(',', 1)
            data = base64.b64decode(encoded)
        else:
            # External file
            fn = image.uri
            if not os.path.isabs(fn):
                fn = folder + '/' + fn
            with open(fn, 'rb') as f:
                data = f.read()
    else:
        buffer_view = gltf.bufferViews[image.bufferView]
        data = get_buffer_data(gltf, buffer_view)
    return data


# Function to convert triangle strip to triangles
def convert_triangle_strip_to_triangles(indices):
    triangles = []
    for i in range(len(indices) - 2):
        if i % 2 == 0:
            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
        else:
            triangles.append([indices[i], indices[i + 2], indices[i + 1]])
    return np.array(triangles).reshape(-1, 3)


# Function to convert triangle fan to triangles
def convert_triangle_fan_to_triangles(indices):
    triangles = []
    for i in range(1, len(indices) - 1):
        triangles.append([indices[0], indices[i], indices[i + 1]])
    return np.array(triangles).reshape(-1, 3)


# Function to get the transformation matrix from a node
def get_node_transform(node):
    if node.matrix:
        return np.array(node.matrix).reshape(4, 4).T
    else:
        T = np.eye(4)
        if node.translation:
            T[:3, 3] = node.translation
        if node.rotation:
            R_mat = R.from_quat(node.rotation).as_matrix()
            T[:3, :3] = R_mat
        if node.scale:
            S = np.diag(node.scale + [1])
            T = T @ S
        return T


def get_world_transform(gltf, node_index, parents, world_transforms):
    if parents[node_index] == -2:
        return world_transforms[node_index]

    node = gltf.nodes[node_index]
    if parents[node_index] == -1:
        world_transforms[node_index] = get_node_transform(node)
        parents[node_index] = -2
        return world_transforms[node_index]

    parent_index = parents[node_index]
    parent_transform = get_world_transform(gltf, parent_index, parents, world_transforms)
    world_transforms[node_index] = parent_transform @ get_node_transform(node)
    parents[node_index] = -2
    return world_transforms[node_index]


def LoadGlb(path):
    # Load the GLB file using pygltflib
    gltf = GLTF2().load(path)

    primitives = []
    images = {}
    # Iterate through the meshes in the GLB file

    world_transforms = [np.identity(4) for i in range(len(gltf.nodes))]
    parents = [-1 for i in range(len(gltf.nodes))]
    for node_index, node in enumerate(gltf.nodes):
        for idx in node.children:
            parents[idx] = node_index
    # for i in range(len(gltf.nodes)):
    #    get_world_transform(gltf, i, parents, world_transform)

    for node_index, node in enumerate(gltf.nodes):
        if node.mesh is not None:
            world_transform = get_world_transform(gltf, node_index, parents, world_transforms)
            # Iterate through the primitives in the mesh
            mesh = gltf.meshes[node.mesh]
            for primitive in mesh.primitives:
                # Access the attributes of the primitive
                attributes = primitive.attributes.__dict__
                mode = primitive.mode if primitive.mode is not None else 4  # Default to TRIANGLES
                result = {}
                if primitive.indices is not None:
                    indices = get_attribute_data(gltf, primitive.indices)
                    if mode == 4:  # TRIANGLES
                        face_indices = indices.reshape(-1, 3)
                    elif mode == 5:  # TRIANGLE_STRIP
                        face_indices = convert_triangle_strip_to_triangles(indices)
                    elif mode == 6:  # TRIANGLE_FAN
                        face_indices = convert_triangle_fan_to_triangles(indices)
                    else:
                        continue
                    result['F'] = face_indices

                # Extract vertex positions
                if 'POSITION' in attributes and attributes['POSITION'] is not None:
                    positions = get_attribute_data(gltf, attributes['POSITION'])
                    # Apply the world transformation to the positions
                    positions_homogeneous = np.hstack([positions, np.ones((positions.shape[0], 1))])
                    transformed_positions = (world_transform @ positions_homogeneous.T).T[:, :3]
                    result['V'] = transformed_positions

                # Extract vertex colors
                if 'COLOR_0' in attributes and attributes['COLOR_0'] is not None:
                    colors = get_attribute_data(gltf, attributes['COLOR_0'])
                    if colors.shape[-1] > 3:
                        colors = colors[..., :3]
                    result['VC'] = colors

                # Extract UVs
                if 'TEXCOORD_0' in attributes and not attributes['TEXCOORD_0'] is None:
                    uvs = get_attribute_data(gltf, attributes['TEXCOORD_0'])
                    result['UV'] = uvs

                if primitive.material is not None:
                    material = gltf.materials[primitive.material]
                    if (
                        material.pbrMetallicRoughness is not None 
                        and material.pbrMetallicRoughness.baseColorTexture is not None
                    ):
                        texture_index = material.pbrMetallicRoughness.baseColorTexture.index
                        texture = gltf.textures[texture_index]
                        image_index = texture.source
                        if not image_index in images:
                            image = gltf.images[image_index]
                            image_data = get_image_data(gltf, image, os.path.dirname(path))
                            pil_image = PILImage.open(io.BytesIO(image_data))
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            images[image_index] = pil_image
                        result['TEX'] = image_index
                    elif material.emissiveTexture is not None:
                        texture_index = material.emissiveTexture.index
                        texture = gltf.textures[texture_index]
                        image_index = texture.source
                        if not image_index in images:
                            image = gltf.images[image_index]
                            image_data = get_image_data(gltf, image, os.path.dirname(path))
                            pil_image = PILImage.open(io.BytesIO(image_data))
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            images[image_index] = pil_image
                        result['TEX'] = image_index
                    else:
                        if material.pbrMetallicRoughness is not None:
                            base_color = material.pbrMetallicRoughness.baseColorFactor
                        else:
                            base_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
                        result['MC'] = base_color

                primitives.append(result)

    return primitives, images


def RotatePrimitives(primitives, transform):
    for i in range(len(primitives)):
        if 'V' in primitives[i]:
            primitives[i]['V'] = primitives[i]['V'] @ transform.T


if __name__ == '__main__':
    path = 'data/test.glb'
    LoadGlb(path)
