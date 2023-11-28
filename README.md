# Gaussian-Splatterer
<p align="center">
  <img src="https://github.com/osreboot/Gaussian-Splatterer/blob/master/res/example1.png" alt="">
  Based on the research paper available <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">here</a>.
</p>

# What is this?
*Gaussian-Splatterer* is an educational tool designed to showcase the strengths and quirks of Gaussian splatting. It features a suite of controls for converting triangle meshes into splats using ray-traced synthetic photogrammetry and CUDA-accelerated gradient descent.

*Gaussian-Splatterer* currently supports the following:
- OBJ model & PNG/TGA/JPG diffuse texture loading
- Two user-defined camera spheres for truth data collection
- User-defined periodic densify and camera randomization steps
- Runtime-customizable learning rates and splat culling/division parameters
- Automatic screenshot generation
- Saving/loading splats using a pseudo-OBJ file format
- Saving/loading projects and settings

# Instructions (For Users)

*Gaussian-Splatterer* must be run on a system with an NVIDIA Volta GPU (20xx series) or newer. The tool functions as a standalone executable and the latest version is available in the [releases section](https://github.com/osreboot/Gaussian-Splatterer/releases).

To convert a triangle mesh model into Gaussian splats:
1. Load the model and texture image using the buttons in `1. Input Model Data`
2. Configure camera sphere parameters using the fields in `2. Build Truth Data`. Note that more cameras will result in slower training times. You can preview truth cameras using the `4. Visualize Splats`->`View Truth` options. For best results, try to have a large number of cameras capture the entire silhouette of the model.
3. Click `2. Build Truth Data`->`Capture Truth` to collect the first set of truth images.
4. Click `3. Train Splats`->`Auto Train`->`Start`.

Parameters that I've found to work well:

### Initial Training

### High-Quality Training

# Instructions (For Maintainers/Experimentalists)

Building *Gaussian-Splatterer* requires [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix).

# Sample Images

# Special Thanks
