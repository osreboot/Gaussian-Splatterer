#pragma once

#include <owl/owl.h>
#include <owl/common/math/random.h>
#include <cuda_runtime.h>

#define PROGRAM_EXPOSURE_FACTOR 1.0f
#define SPLAT_CAMERA_DOT_SIZE 0.025f

#define TEXTURE_COUNT 1
#define TEXTURE_DIFFUSE 0

// TODO add support for roughness textures and replace this
#define MATERIAL_REFLECTIVITY 0.0f

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
    int samples;

    OptixTraversableHandle worldHandle;

    owl::vec3f cameraLocation;
    float* cameraMatrix;

    int splatCamerasCount;
    owl::vec3f* splatCameras;
};

// Data associated with each ray
struct PerRayData {
    owl::LCG<4> random; // A random object
    bool shouldTerminate; // Did the ray hit the sky or a light source?
    owl::vec3f hitOrigin; // Collision location
    owl::vec3f bounceDirection; // New ray direction (post-collision)
    owl::vec3f color; // New ray color (post-collision)
    bool splatTouchesCamera = false; // Does the ray intersect with a splat camera indicator orb?
    bool reflected = false; // Has the ray been reflected off a surface?
};
