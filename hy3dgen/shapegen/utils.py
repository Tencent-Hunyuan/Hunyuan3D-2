# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    resolved_model_path = None

    # 1. Try direct local path: <model_path>/<subfolder>
    potential_local_path = os.path.join(original_model_path, subfolder)
    if os.path.isdir(potential_local_path):
        logger.info(f'Found model at local path: {potential_local_path}')
        resolved_model_path = potential_local_path

    if resolved_model_path is None:
        # 2. Try cache path: ~/.cache/hy3dgen/<model_path>/<subfolder>
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        cache_path = os.path.expanduser(os.path.join(base_dir, original_model_path, subfolder))
        logger.info(f'Trying to load model from cache path: {cache_path}')
        if os.path.isdir(cache_path):
            logger.info(f'Found model at cache path: {cache_path}')
            resolved_model_path = cache_path

    if resolved_model_path is None:
        # 3. Try to download from Hugging Face Hub
        logger.info(f'Model not found locally or in cache, trying to download from Hugging Face Hub: {original_model_path}')
        try:
            from huggingface_hub import snapshot_download
            # Download only the specified subfolder
            hub_path = snapshot_download(
                repo_id=original_model_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            # The path returned by snapshot_download is the root of the downloaded repo,
            # so we need to append the subfolder.
            resolved_model_path = os.path.join(hub_path, subfolder)
            logger.info(f'Successfully downloaded model to: {resolved_model_path}')
        except ImportError:
            logger.warning(
                "You need to install HuggingFace Hub to load models from the hub. "
                "Skipping download attempt."
            )
        except Exception as e:
            logger.error(f"Error downloading model from Hugging Face Hub: {e}")
            # We don't raise here, as we want to check one last time if the path exists after all attempts.

    if resolved_model_path is None or not os.path.isdir(resolved_model_path):
        raise FileNotFoundError(
            f"Model not found. Attempted paths:\n"
            f"  - Direct: {potential_local_path}\n"
            f"  - Cache: {cache_path if 'cache_path' in locals() else 'Not attempted'}\n"
            f"  - Hugging Face Hub download for: {original_model_path}/{subfolder}"
        )

    model_path = resolved_model_path

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant}.{extension}'
    config_path = os.path.join(model_path, 'config.yaml')
    ckpt_path = os.path.join(model_path, ckpt_name)
    return config_path, ckpt_path
