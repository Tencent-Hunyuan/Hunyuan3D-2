#!/usr/bin/env python
"""Bootstrap launcher — cleans sys.path to avoid Hermes PYTHONPATH pollution.
Usage: python run_hy3d.py --image_path YOUR_IMAGE.jpg [--output OUTPUT.glb]"""

import os, sys, argparse

# ── Bootstrap: remove Hermes paths from sys.path before any imports ──
hermes_markers = ['hermes-agent', 'hermes_workspace']
cleaned = []
for p in list(sys.path):
    if any(marker in p.lower() for marker in hermes_markers):
        cleaned.append(p)
sys.path[:] = [p for p in sys.path if p not in cleaned]

os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('HF_HOME', 'G:/Hunyuan3D-2-Standalone/models')

# Now safe to import everything
from PIL import Image
import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Input image')
    parser.add_argument('--output', default=None, help='Output .glb file path')
    args = parser.parse_args()

    # Load pipeline
    print("Loading Hunyuan3D model...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0"
    )
    
    # Background removal
    print("Removing background...")
    bg_remover = BackgroundRemover()
    input_image_path = args.image_path
    if not os.path.exists(input_image_path):
        print(f"ERROR: Image not found at {input_image_path}")
        sys.exit(1)
    
    image_pil = Image.open(input_image_path).convert("RGB")
    image_masked = bg_remover(image_pil)
    
    # Generate 3D shape
    print("Generating 3D mesh (this takes ~2-5 minutes)...")
    meshes_list = pipeline_shapegen(
        image=image_masked,
        num_inference_steps=30,
        mc_algo="mc",
    )
    
    output_path = args.output or "output.glb"
    from trimesh.exchange import glb
    mesh_obj = meshes_list[0]
    with open(output_path, "wb") as f:
        f.write(glb.export_glb(mesh_obj))
    
    print(f"✓ Done! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
