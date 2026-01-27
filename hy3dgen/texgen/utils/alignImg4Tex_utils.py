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

import torch
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, \
    AutoencoderKL


class Img2img_Control_Ip_adapter:
    def __init__(self, device):
        controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.load_ip_adapter('h94/IP-Adapter', subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
        pipe.set_ip_adapter_scale(0.7)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        self.pipe = pipe.to(device)

    def __call__(
        self,
        prompt,
        control_image,
        ip_adapter_image,
        negative_prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=8.0,
        controlnet_conditioning_scale=1.0,
        output_type="pil",
        **kwargs,
    ):
        results = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            ip_adapter_image=ip_adapter_image,
            generator=torch.manual_seed(42),
            seed=42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=1,
            # clip_skip=2,
            height=height,
            width=width,
            output_type=output_type,
            **kwargs,
        ).images[0]
        return results


################################################################

class HesModel:
    def __init__(self, ):
        controlnet_depth = ControlNetModel.from_pretrained(
            'diffusers/controlnet-depth-sdxl-1.0',
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet_depth,
            use_safetensors=True,
        )
        self.pipe.vae = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix',
            torch_dtype=torch.float16
        )

        self.pipe.load_ip_adapter('h94/IP-Adapter', subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
        self.pipe.set_ip_adapter_scale(0.7)
        self.pipe.to("cuda")

    def __call__(self,
                 init_image,
                 control_image,
                 ip_adapter_image=None,
                 prompt='3D image',
                 negative_prompt='2D image',
                 seed=42,
                 strength=0.8,
                 num_inference_steps=40,
                 guidance_scale=7.5,
                 controlnet_conditioning_scale=0.5,
                 **kwargs
                 ):
        image = self.pipe(
            prompt=prompt,
            image=init_image,
            control_image=control_image,
            ip_adapter_image=ip_adapter_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            **kwargs
        ).images[0]
        return image
