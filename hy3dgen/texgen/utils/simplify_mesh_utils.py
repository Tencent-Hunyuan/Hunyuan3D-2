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


def remesh_mesh(mesh_path, remesh_path, method='trimesh'):
    if method == 'trimesh':
        mesh_simplify_trimesh(mesh_path, remesh_path)
    else:
        raise f'Method {method} has not been implemented.'


def mesh_simplify_trimesh(inputpath, outputpath):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    ms.save_current_mesh(outputpath.replace('.glb', '.obj'), save_textures=False)

    courent = trimesh.load(outputpath.replace('.glb', '.obj'), force='mesh')
    face_num = courent.faces.shape[0]

    if face_num > 100000:
        courent = courent.simplify_quadric_decimation(40000)
    courent.export(outputpath)
