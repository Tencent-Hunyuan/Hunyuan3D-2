import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

image_path = 'assets/demo.png'
image = Image.open(image_path)
if image.mode not in ['RGB', 'RGBA']:
    image = image.convert('RGB')
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini',
    variant='fp16'
)

start_time = time.time()
mesh = pipeline(
    image=image,
    num_inference_steps=50,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(f'demo_mini.glb')
