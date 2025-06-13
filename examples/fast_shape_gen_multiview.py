import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

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
    subfolder='hunyuan3d-dit-v2-mv-turbo',
    variant='fp16'
)
pipeline.enable_flashvdm()
start_time = time.time()
mesh = pipeline(
    image=images,
    num_inference_steps=5,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(f'demo_mv3.glb')
