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


def LoadObj(fn):
    lines = [l.strip() for l in open(fn)]
    vertices = []
    faces = []
    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words) == 0:
            continue
        if words[0] == 'v':
            v = [float(words[i]) for i in range(1, 4)]
            vertices.append(v)
        elif words[0] == 'f':
            f = [int(words[i]) - 1 for i in range(1, 4)]
            faces.append(f)

    return np.array(vertices).astype('float32'), np.array(faces).astype('int32')


def LoadObjWithTexture(fn, tex_fn):
    lines = [l.strip() for l in open(fn)]
    vertices = []
    vertex_textures = []
    faces = []
    face_textures = []
    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words) == 0:
            continue
        if words[0] == 'v':
            v = [float(words[i]) for i in range(1, len(words))]
            vertices.append(v)
        elif words[0] == 'vt':
            v = [float(words[i]) for i in range(1, len(words))]
            vertex_textures.append(v)
        elif words[0] == 'f':
            f = []
            ft = []
            for i in range(1, len(words)):
                t = words[i].split('/')
                f.append(int(t[0]) - 1)
                ft.append(int(t[1]) - 1)
            for i in range(2, len(f)):
                faces.append([f[0], f[i - 1], f[i]])
                face_textures.append([ft[0], ft[i - 1], ft[i]])

    tex_image = cv2.cvtColor(cv2.imread(tex_fn), cv2.COLOR_BGR2RGB)
    return np.array(vertices).astype('float32'), np.array(vertex_textures).astype('float32'), np.array(faces).astype(
        'int32'), np.array(face_textures).astype('int32'), tex_image
