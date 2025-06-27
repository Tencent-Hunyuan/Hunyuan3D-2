import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

images = {
    "front": "assets/example_mv_images/1/front.png",
    "left": "assets/example_mv_images/1/left.png",
    "back": "assets/example_mv_images/1/back.png"
}

for key in images:
    image = Image.open(images[key])
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images[key] = image

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv',
    variant='fp16'
)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

start_time = time.time()
mesh = pipeline(
    image=images,
    num_inference_steps=50,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(f'demo_white_mesh_mv.glb')

mesh = pipeline_texgen(mesh, image=images["front"])
mesh.export('demo_textured_mv.glb')
