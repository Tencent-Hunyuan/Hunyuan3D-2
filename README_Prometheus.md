# Hunyuan3D 2.0: AI-Powered High-Resolution 3D Asset Generation Platform

## Project Overview

Hunyuan3D 2.0 is an advanced AI-powered system for generating high-resolution, textured 3D assets through state-of-the-art machine learning techniques. The project aims to democratize 3D content creation by providing a powerful, user-friendly platform for generating and manipulating 3D models from images or text inputs.

### Key Features

#### Advanced 3D Generation
- **Two-Stage Generation Pipeline**: Seamlessly creates 3D assets by first generating a base mesh and then applying high-quality textures
- Supports multiple generation modes:
  - Image-to-3D conversion
  - Multi-view 3D generation
  - Texture synthesis for existing meshes

#### Cutting-Edge AI Models
- **Hunyuan3D-DiT**: A large-scale shape generation model built on a scalable flow-based diffusion transformer
- **Hunyuan3D-Paint**: A texture synthesis model that produces vibrant, high-resolution texture maps
- Multiple model variants including mini, multiview, and turbo versions

#### Versatile Capabilities
- Generate 3D assets from a single reference image
- Create textured 3D models with high geometric detail and condition alignment
- Supports various input types and generation scenarios

### Core Benefits
- Lowers the barrier to 3D content creation
- Provides professional-grade 3D asset generation capabilities
- Flexible and adaptable across different use cases
- Supports multiple platforms (MacOS, Windows, Linux)
- Open-source with comprehensive documentation and community support

### Performance Highlights
The system has demonstrated superior performance compared to existing 3D generation methods, consistently outperforming both open-source and closed-source models in key metrics such as condition following, texture quality, and geometric detail.

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU recommended
- Operating Systems: Linux, macOS, Windows

### Installation

You can install the project using pip:

```bash
pip install git+https://github.com/Tencent/Hunyuan3D-2.git
```

Alternatively, you can clone the repository and install locally:

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -e .
```

### Dependencies

The project requires the following key dependencies:
- PyTorch
- Diffusers
- Transformers
- OpenCV
- Gradio (for demos)
- FastAPI (for server)
- ONNX Runtime
- Trimesh
- Numpy

A full list of dependencies is available in `requirements.txt`.

### Quick Start

#### Using Gradio Web Interface

To launch the web interface:

```bash
python gradio_app.py
```

#### Using Python Script

Here's a minimal example for 3D shape generation:

```python
from hy3dgen import ShapeGenerator

# Initialize the generator
generator = ShapeGenerator()

# Generate a 3D shape from a text prompt
shape = generator.generate("A detailed dragon sculpture")

# Save the generated shape
shape.save("dragon.glb")
```

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### GPU Requirements

- CUDA 11.x or newer recommended
- Minimum 8GB GPU memory for most generation tasks
- More memory allows for higher resolution and more complex generations

### Troubleshooting

- Ensure you have the latest version of pip
- If you encounter CUDA-related issues, verify your CUDA and PyTorch versions are compatible
- For any installation problems, check the project's GitHub issues page

## API Reference

### Text-to-Image Generation

#### `HunyuanDiTPipeline`
A class for generating high-quality images from text prompts using the Hunyuan DiT model.

##### Constructor
```python
HunyuanDiTPipeline(
    model_path="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", 
    device='cuda'
)
```
- `model_path`: Path to the pre-trained model (default is the official Hunyuan DiT model)
- `device`: Computing device to use (default is 'cuda')

##### Methods
- `compile()`: Optimizes the model's transformer for faster inference
- `__call__(prompt: str, seed: int = 0) -> Image`: Generates an image from a text prompt
  - `prompt`: Text description of the desired image
  - `seed`: Random seed for reproducibility (default is 0)
  - Returns a generated image

###### Example Usage
```python
pipeline = HunyuanDiTPipeline()
pipeline.compile()  # Optional optimization step
image = pipeline("A beautiful landscape with mountains")
```

#### Utility Functions

##### `seed_everything(seed: int)`
Sets a global random seed for reproducibility across NumPy, PyTorch, and Python's random module.

### Rembg Module
The library includes a rembg module for background removal, which can be imported from `hy3dgen.rembg`.

### Shape Generation
The `hy3dgen.shapegen` module provides functionality for 3D shape generation with several submodules:
- `models`: Neural network architectures for shape generation
- `pipelines`: Processing pipelines
- `preprocessors`: Input preprocessing utilities
- `postprocessors`: Output post-processing utilities
- `schedulers`: Noise scheduling for generation
- `utils`: Utility functions for shape generation

### Texture Generation
The `hy3dgen.texgen` module offers texture generation capabilities with components:
- `custom_rasterizer`: Custom rendering utilities
- `differentiable_renderer`: Differentiable mesh rendering
- `hunyuanpaint`: Texture painting pipeline
- `utils`: Various texture-related utility functions

### Additional Utilities
- `text2image`: Text-to-image generation module
- Configuration and model utilities are available throughout the package

## Project Structure

The project is a Python-based 3D generation library with a structured, modular architecture. The key directories and their purposes are as follows:

### Main Package
The `hy3dgen` directory serves as the core package, containing the primary implementation of the 3D generation library:

#### Submodules
- `shapegen`: Handles shape generation capabilities
  - `models`: Contains machine learning models for shape generation
    - `autoencoders`: Implementations of autoencoder architectures
    - `denoisers`: Denoising models and techniques
  - `pipelines.py`: Defines processing pipelines for shape generation
  - `preprocessors.py` and `postprocessors.py`: Handle input preprocessing and output postprocessing
  - `schedulers.py`: Manages generation scheduling

- `texgen`: Responsible for texture generation and rendering
  - `custom_rasterizer`: Custom rendering implementation with C++ and CUDA kernel extensions
  - `differentiable_renderer`: Provides differentiable rendering utilities
  - `hunyuanpaint`: Texture painting pipeline
  - `pipelines.py`: Texture generation processing pipelines
  - `utils`: Utility functions for texture-related operations

### Examples and Demonstrations
The `examples` directory contains script demonstrations of various generation capabilities:
- Shape generation scripts
- Multiview generation examples
- Texture generation scripts

### Documentation
The `docs` directory includes documentation resources:
- Sphinx documentation configuration
- Markdown documentation files
- Configuration for documentation generation

### Additional Resources
- `assets`: Contains various supporting resources like images, environment maps, and example data
- `blender_addon.py`: A Blender plugin integration
- `gradio_app.py`: A Gradio web interface for the library
- `api_server.py`: An API server implementation

### Project Configuration
- `setup.py`: Package setup and installation configuration
- `requirements.txt`: Python package dependencies
- `LICENSE` and `NOTICE`: Legal and licensing information

The project follows a clean, modular design that separates concerns between shape generation, texture generation, and utility functions, making it extensible and easy to understand.

## Technologies Used

### Programming Languages
- Python

### Machine Learning and Deep Learning
- PyTorch: Core deep learning framework
- Diffusers: Generative AI model pipeline
- Transformers: Advanced natural language and machine learning models
- Accelerate: Distributed training and inference library

### Computer Vision and Image Processing
- OpenCV: Image and video processing
- Torchvision: Computer vision utilities
- Rembg: Background removal library
- ONNX Runtime: Machine learning model inference

### 3D Graphics and Mesh Processing
- Trimesh: 3D mesh manipulation
- PyMeshLab: Mesh processing and editing
- PyGLTFlib: GLTF file handling
- XAtlas: UV mapping and texture coordinate generation

### Web and API Frameworks
- Gradio: Machine learning demo and interface creation
- FastAPI: Web API development
- Uvicorn: ASGI server for web applications

### Utility Libraries
- NumPy: Numerical computing
- tqdm: Progress bar and iteration tracking
- Einops: Tensor manipulation and reshaping
- OmegaConf: Configuration management

### Development and Build Tools
- Ninja: Build system
- PyBind11: C++ and Python interoperability

## Additional Notes

### Compatibility and System Requirements

Hunyuan3D 2.0 is designed to be cross-platform, supporting macOS, Windows, and Linux. The project requires:
- Minimum 6 GB VRAM for shape generation
- 16 GB VRAM for combined shape and texture generation
- PyTorch (installed from official site)
- Python environment with dependencies specified in `requirements.txt`

### Performance Considerations

The framework offers multiple model variants optimized for different use cases:
- Mini models (0.6B parameters) for lower computational requirements
- Multiview models for enhanced 3D generation
- Turbo and Fast variants with reduced inference time

### Troubleshooting and Support

For technical support and community discussions, users can:
- Join the Discord community: [Hunyuan3D Discord](https://discord.gg/dNBrdrGGMa)
- Explore community-contributed extensions and tools
- Refer to example scripts in the `examples/` directory for advanced usage

### Upcoming Features

The project roadmap includes:
- TensorRT optimization
- Finetuning capabilities
- Ongoing model and performance improvements

### Research and Citations

If you use Hunyuan3D in academic or commercial projects, consider citing the project's technical reports available on arXiv.

## Contributing

We welcome contributions from the community! By participating in this project, you can help improve Hunyuan3D and make it even more powerful.

### Ways to Contribute

1. **Reporting Issues**
   - Open GitHub issues for bug reports, feature requests, or documentation improvements
   - Provide clear, detailed descriptions and, when possible, reproducible examples

2. **Proposing Extensions**
   - Submit pull requests for bug fixes
   - Develop new features or improvements to existing functionality
   - Enhance documentation or add examples

### Contribution Guidelines

#### Code Contributions
- Ensure your code follows the project's existing code style and conventions
- Write clear, concise, and descriptive commit messages
- Include appropriate tests for new features or bug fixes
- Update documentation to reflect your changes

#### Pull Request Process
- Fork the repository and create your branch from `main`
- Ensure all tests pass and there are no merge conflicts
- Include a clear description of your changes in the pull request
- Your pull request will be reviewed by the maintainers

#### Development Setup
```bash
# Clone the repository
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install additional rendering dependencies 
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

### Community
- Join our [Discord community](https://discord.gg/dNBrdrGGMa) for discussions and support
- Check out [community resources and extensions](https://github.com/Tencent/Hunyuan3D-2#community-resources)

### Code of Conduct
- Be respectful and considerate of others
- Collaborate constructively
- Focus on improving the project for everyone

### Licensing
By contributing, you agree that your contributions will be licensed under the [Tencent Hunyuan Non-Commercial License Agreement](LICENSE).

## License

The project is licensed under the **Tencent Hunyuan 3D 2.0 Community License Agreement**.

#### Key License Terms
- The license is valid only in territories outside the European Union, United Kingdom, and South Korea
- Non-exclusive, non-transferable, and royalty-free limited license
- Must comply with the Acceptable Use Policy
- Restrictions apply for commercial use, especially for products with over 1 million monthly active users

#### License Compatibility
- You may distribute the work within the specified territory
- Must include the original license text with distributions
- Modifications must be clearly marked

#### Full License Details
For complete license terms, please refer to the [LICENSE](LICENSE) file in the repository.

#### Important Notes
- No warranties are provided with the software
- Users are responsible for compliance with the license terms
- Trademark usage is restricted