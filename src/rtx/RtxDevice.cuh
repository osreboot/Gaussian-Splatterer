#pragma once

#include <owl/owl.h>
#include <owl/common/math/random.h>
#include <cuda_runtime.h>

#define PROGRAM_SAMPLES 50
#define PROGRAM_EXPOSURE_FACTOR 1.0f
#define SPLAT_CAMERA_DOT_SIZE 0.025f

// Material properties
struct Material {
    bool fullbright; // Is it a light source?
    float reflectivity; // Probability to reflect instead of absorb
    float diffuse; // Scattering magnitude
    owl::vec3f color; // Surface color
    float gloss = 0.0f;
    int textureDiffuse = -1;
};

// Raw geometry data
struct WorldGeometry {
    owl::vec3f* vertices;
    owl::vec3i* triangles;
    owl::vec2f* textureCoords;
    cudaTextureObject_t* textures;
};

// Data used by the ray generator
struct RayGenerator {
    uint32_t* frameBuffer;
    owl::vec2i size;
    owl::vec3f background;

    OptixTraversableHandle worldHandle;

    owl::vec3f cameraLocation;
    float* cameraMatrix;

    int splatCamerasCount;
    owl::vec3f* splatCameras;
};

// Data associated with each ray
struct PerRayData {
    owl::LCG<4> random; // A random object
    bool hitDetected; // Did the ray hit something?
    owl::vec3f hitOrigin; // Collision location
    owl::vec3f bounceDirection; // New ray direction (post-collision)
    owl::vec3f color; // New ray color (post-collision)
    bool splatTouchesCamera = false;
};
