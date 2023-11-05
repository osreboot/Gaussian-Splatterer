#pragma once

#include <owl/owl.h>
#include <owl/common/math/random.h>
#include <cuda_runtime.h>

#define PROGRAM_SAMPLES 50
#define PROGRAM_EXPOSURE_FACTOR 1.0f
#define SPLAT_CAMERA_DOT_SIZE 0.025f

#define NESTED_MATERIALS_MAX 50

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

    OptixTraversableHandle worldHandle;

    // The ray tracer projects rays from the camera's 'location' point through the plane specified by:
    //   originPixel + dirRight * x + dirUp * y
    // This effectively creates a perspective projection, with zNear and zFar planes determined by the limits of the
    // ray tracer itself.
    struct {
        owl::vec3f location;
        owl::vec3f originPixel;
        owl::vec3f dirRight;
        owl::vec3f dirUp;
    } camera;

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
