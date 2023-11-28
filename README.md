# Gaussian-Splatterer
<p align="center">
  <img src="https://github.com/osreboot/Gaussian-Splatterer/blob/master/res/example1.png" alt="">
  Based on the research paper available <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">here</a>. <i>Spotted Red Mushroom source model available on <a href="https://quixel.com/megascans/home/">Quixel</a>.</i>
</p>

# What is this?
*Gaussian-Splatterer* is an educational tool designed to showcase the strengths and quirks of Gaussian splatting. It features a suite of controls for converting triangle meshes into splats using ray-traced synthetic photogrammetry and CUDA-accelerated gradient descent.

*Gaussian-Splatterer* currently supports the following:
- OBJ model & PNG/TGA/JPG diffuse texture loading (with transparency support)
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

Parameters that I've found to work well (omitted values are default):

| Early-Stage Training | 0-10k Splats |
| --- | --- |
| Sphere 1 Count | 8 |
| Sphere 1 Distance/FOV | Fits entire model, e.g. 10.0/60.0 |
| Sphere 2 Count | 0 |
| Truth RT Samples | 50 |

| Mid-Stage Training | 10k-50k Splats |
| --- | --- |
| Sphere 1 Count | 8-16 |
| Sphere 1 Distance/FOV | Fits entire model, e.g. 10.0/60.0 |
| Sphere 2 Count | 8-16 |
| Sphere 2 Distance/FOV | Close-up shots, e.g. 10.0/20.0 |
| Truth RT Samples | 50 |
| Learning Rate: Color | 0.01 |
| Learning Rate: Opacity | 0.01 |

| Late-Stage Training | 50k+ Splats |
| --- | --- |
| Sphere 1 Count | 16+ |
| Sphere 1 Distance/FOV | Fits entire model, e.g. 10.0/60.0 |
| Sphere 2 Count | 16+ |
| Sphere 2 Distance/FOV | Close-up shots, e.g. 10.0/20.0 |
| Truth RT Samples | 100 |
| Learning Rate: Color | 0.2 |
| Learning Rate: Opacity | 0.2 |

Rule of thumb: if splats are unstable (location-wise, scale-wise, or color-wise) wait to change parameters again until the model converges somewhat and becomes stable again. I like to slowly increase color/opacity learning rates throughout training (as this results in the best low-level detail), however increasing these too quickly will cause severe instability (and even possibly model destruction). Adding more cameras or changing the distance/FOV will likely dramatically increase the model error, so expect to wait after chaning these values as well.

# Instructions (For Maintainers/Experimentalists)

Building *Gaussian-Splatterer* requires [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix). Additionally, the `OptiX_ROOT_DIR` CMake variable needs to be set to the location of your OptiX install.

# Sample Images
<p align="center">
  <img src="https://github.com/osreboot/Gaussian-Splatterer/blob/master/res/example2.png" alt="">
  <i>Madagascar Giant Day Gecko source model available on <a href="https://www.turbosquid.com/3d-models/gecko-rig-model-1181653">TurboSquid</a>.</i>
</p>

<p align="center">
  <img src="https://github.com/osreboot/Gaussian-Splatterer/blob/master/res/example3.png" alt="">
  <i>Bigleaf Hydrangea source model available on <a href="https://quixel.com/megascans/home/">Quixel</a>.</i>
</p>

# Special Thanks
This tool is based on Gaussian Splatting, a novel computer graphics modeling/rendering approach developed by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. You can find their research paper and more [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This tool also depends on (and wouldn't be possible without) the following projects:
- [OptiX Wrapper Library](https://github.com/owl-project/owl) powers the truth ray-tracer.
- [wxWidgets](https://github.com/wxWidgets/wxWidgets) used to build the interface.
- [JSON for Modern C++](https://github.com/nlohmann/json) serializes project settings.
- [stb](https://github.com/nothings/stb) used to save screenshots.
