#include "RtxDevice.cuh"

#include <optix_device.h>

using namespace owl;

// Selects a random point inside a unit sphere
inline __device__ vec3f randomUnitSphere(LCG<4>& random) {
    vec3f v;
    do {
        v = 2.0f * vec3f(random(), random(), random()) - vec3f(1.0f, 1.0f, 1.0f);
    } while(dot(v, v) >= 1.0f);
    return v;
}

// Tracks a ray through multiple bounces in the world
inline __device__ vec3f tracePath(const RayGenerator& rayGen, Ray& ray, PerRayData& prd) {
    vec3f colorAtten = vec3f(1.0f);

    bool hasReflected = false;

    // Loop as long as we haven't reached the maximum bounce depth
    for (int i = 0; i < 50; i++) {
        prd.shouldTerminate = true;
        prd.reflected = false;

        // Launch the ray
        traceRay(rayGen.worldHandle, ray, prd);

        // Apply ray color attenuation
        colorAtten *= vec3f(prd.color.x, prd.color.y, prd.color.z);
        if (prd.reflected) hasReflected = true;

        // Detect if the ray hit a splat camera indicator orb. This is inefficient and causes some lag, a better
        // solution would be to use proper hardware acceleration via a custom geometry type.
        if (i == 0) { // Only check on the first iteration because we don't want the camera orbs to cast light/shadows.
            for (int c = 0; c < rayGen.splatCamerasCount; c++) {
                if(!prd.shouldTerminate && dot(ray.direction, rayGen.splatCameras[c] - ray.origin) > dot(ray.direction, prd.hitOrigin - ray.origin)) continue;

                vec3f rayClosest = ray.origin + ray.direction * dot(ray.direction, rayGen.splatCameras[c] - ray.origin);
                vec3f rayDelta = rayGen.splatCameras[c] - rayClosest;
                if (dot(rayDelta, rayDelta) < SPLAT_CAMERA_DOT_SIZE * SPLAT_CAMERA_DOT_SIZE) {
                    prd.splatTouchesCamera = true;
                    break;
                }
            }
        }

        // The ray hit the sky or a light source
        if (prd.shouldTerminate) return hasReflected ? colorAtten : rayGen.background;

        // Re-initialize the ray based on collision parameters
        ray = Ray(prd.hitOrigin, prd.bounceDirection, 1e-3f, 1e10f);
    }

    // Max bounces exceeded
    return {0.0f, 0.0f, 0.0f};
}

// Ray generation program
OPTIX_RAYGEN_PROGRAM(rayGenProgram)() {
    const RayGenerator& rayGen = getProgramData<RayGenerator>();
    const vec2i pixel = getLaunchIndex();

    vec3f color = vec3f(0.0f);
    PerRayData prd;

    // Cast rays to fulfill the number of required samples
    for(int i = 0; i < rayGen.samples; i++){
        // Create ray from camera
        Ray ray;
        ray.origin = rayGen.cameraLocation;

        // Calculate the ray's direction based on the camera matrix
        const vec2f pixelFine = vec2f(pixel) + vec2f(prd.random(), prd.random()) + vec2f(0.5);
        const vec3f viewFarZ = vec3f((pixelFine.x * 2.0 / rayGen.size.x) - 1.0f, (pixelFine.y * 2.0 / rayGen.size.y) - 1.0f, 1.0f);
        const vec4f rayFarZ = vec4f(viewFarZ.x * rayGen.cameraMatrix[0] + viewFarZ.y * rayGen.cameraMatrix[4] + viewFarZ.z * rayGen.cameraMatrix[8] + rayGen.cameraMatrix[12],
                                    viewFarZ.x * rayGen.cameraMatrix[1] + viewFarZ.y * rayGen.cameraMatrix[5] + viewFarZ.z * rayGen.cameraMatrix[9] + rayGen.cameraMatrix[13],
                                    viewFarZ.x * rayGen.cameraMatrix[2] + viewFarZ.y * rayGen.cameraMatrix[6] + viewFarZ.z * rayGen.cameraMatrix[10] + rayGen.cameraMatrix[14],
                                    viewFarZ.x * rayGen.cameraMatrix[3] + viewFarZ.y * rayGen.cameraMatrix[7] + viewFarZ.z * rayGen.cameraMatrix[11] + rayGen.cameraMatrix[15]);

        ray.direction = normalize(vec3f(rayFarZ.x / rayFarZ.w, rayFarZ.y / rayFarZ.w, rayFarZ.z / rayFarZ.w) - rayGen.cameraLocation);

        // Trace the ray's path
        vec3f colorOut = tracePath(rayGen, ray, prd) * PROGRAM_EXPOSURE_FACTOR;

        // Clamp the output color
        colorOut.x = max(min(colorOut.x, 1.0f), 0.0f);
        colorOut.y = max(min(colorOut.y, 1.0f), 0.0f);
        colorOut.z = max(min(colorOut.z, 1.0f), 0.0f);

        color += colorOut;
    }

    color /= (float)rayGen.samples;

    if (prd.splatTouchesCamera) color = vec3f(1.0f) - color;

    // Assign frame buffer pixel color based on average of all samples
    rayGen.frameBuffer[pixel.x + rayGen.size.x * pixel.y] = make_rgba(color);
}

// Ray hit program
OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)() {
    PerRayData& prd = getPRD<PerRayData>();
    const WorldGeometry& world = getProgramData<WorldGeometry>();

    // Fetch data about the collision surface
    const unsigned int indexPrimitive = optixGetPrimitiveIndex();
    const vec3i index = world.triangles[indexPrimitive];

    // Calculate the normal of the surface
    const vec3f normalSurface = normalize(cross(world.vertices[index.y] - world.vertices[index.x],
                                                world.vertices[index.z] - world.vertices[index.x]));

    const vec3f rayOrigin = optixGetWorldRayOrigin();
    const vec3f rayDir = optixGetWorldRayDirection();

    vec2f uv = optixGetTriangleBarycentrics();
    vec2f textureCoord = (1.0f - uv.x - uv.y) * world.textureCoords[indexPrimitive * 3] +
                         uv.x * world.textureCoords[indexPrimitive * 3 + 1] +
                         uv.y * world.textureCoords[indexPrimitive * 3 + 2];
    vec4f colorTexture = vec4f(tex2D<float4>(world.textures[TEXTURE_DIFFUSE], textureCoord.x, 1.0f - textureCoord.y));

    // Calculate the position of the collision
    prd.hitOrigin = rayOrigin + optixGetRayTmax() * rayDir;

    if (colorTexture.w > prd.random()) { // Ray collided with the material
        // Calculate reflected direction
        vec3f directionReflect = rayDir - 2.0f * dot(rayDir, normalSurface) * normalSurface;
        if(prd.random() > MATERIAL_REFLECTIVITY){ // Scattering for lambertians
            directionReflect = normalSurface + randomUnitSphere(prd.random);
        }

        // Assign final ray data based on all the above calculations
        prd.bounceDirection = directionReflect;

        prd.color = vec3f(colorTexture.x, colorTexture.y, colorTexture.z);
        prd.reflected = true;
    } else { // Ray passed through the material
        prd.bounceDirection = rayDir;
        prd.color = vec3f(1.0f);
    }

    prd.shouldTerminate = false;
}

// Ray miss program
OPTIX_MISS_PROGRAM(miss)() {
    PerRayData& prd = getPRD<PerRayData>();

    vec3f rayDirNormal = normalize(vec3f(optixGetWorldRayDirection()));

    // Generic white/gray sky color function TODO add support for sky sphere textures or similar
    prd.color = vec3f(min(1.0f, 1.0f + rayDirNormal.y));

    prd.shouldTerminate = true;
}
