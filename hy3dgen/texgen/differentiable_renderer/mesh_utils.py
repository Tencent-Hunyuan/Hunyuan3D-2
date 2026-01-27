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

import trimesh


def load_mesh(mesh):
    vtx_pos = mesh.vertices if hasattr(mesh, 'vertices') else None
    pos_idx = mesh.faces if hasattr(mesh, 'faces') else None

    vtx_uv = mesh.visual.uv if hasattr(mesh.visual, 'uv') else None
    uv_idx = mesh.faces if hasattr(mesh, 'faces') else None

    texture_data = None

    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def save_mesh(mesh, texture_data):
    material = trimesh.visual.texture.SimpleMaterial(image=texture_data, diffuse=(255, 255, 255))
    texture_visuals = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=texture_data, material=material)
    mesh.visual = texture_visuals
    return mesh
