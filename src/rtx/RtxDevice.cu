#include <optix_device.h>

#include "RtxDevice.cuh"

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
    vec3f attenuation = vec3f(1.0f);
    // Loop as long as we haven't reached the maximum bounce depth
    for(int i = 0; i < 50; i++){
        prd.hitDetected = false;

        // Launch the ray
        traceRay(rayGen.worldHandle, ray, prd);

        attenuation *= prd.color;

        if(i == 0) {
            for (int c = 0; c < rayGen.splatCamerasCount; c++) {
                if(prd.hitDetected && dot(ray.direction, rayGen.splatCameras[c] - ray.origin) > dot(ray.direction, prd.hitOrigin - ray.origin)) continue;

                vec3f rayClosest = ray.origin + ray.direction * dot(ray.direction, rayGen.splatCameras[c] - ray.origin);
                vec3f rayDelta = rayGen.splatCameras[c] - rayClosest;
                if (dot(rayDelta, rayDelta) < SPLAT_CAMERA_DOT_SIZE * SPLAT_CAMERA_DOT_SIZE) {
                    prd.splatTouchesCamera = true;
                    break;
                }
            }
        }

        // The ray hit the sky or a light source
        if(!prd.hitDetected) return i == 0 ? vec3f(0.0f, 0.0f, 0.0f) : attenuation;

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
    for(int i = 0; i < PROGRAM_SAMPLES; i++){
        // Create ray from camera
        Ray ray;
        ray.origin = rayGen.camera.location;

        // Optional: link the ray's random seed to the pixel position. This is good for static images, but makes
        // real-time renders look like there's dirt on the screen.
        // prd.random.init(pixel.x + self.size.x * i,
        //                 pixel.y + self.size.y * i);

        // Set the ray's position and direction based on the current pixel
        const vec2f screen = (vec2f(pixel) + vec2f(prd.random(), prd.random()) + vec2f(0.5f)) / vec2f(rayGen.size);
        ray.direction = normalize(rayGen.camera.originPixel + screen.u * rayGen.camera.dirRight + screen.v * rayGen.camera.dirUp);

        // Trace the ray's path
        vec3f colorOut = tracePath(rayGen, ray, prd) * PROGRAM_EXPOSURE_FACTOR;

        // Clamp the output color
        colorOut.x = max(min(colorOut.x, 1.0f), 0.0f);
        colorOut.y = max(min(colorOut.y, 1.0f), 0.0f);
        colorOut.z = max(min(colorOut.z, 1.0f), 0.0f);

        color += colorOut;
    }

    color /= (float)PROGRAM_SAMPLES;

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
    const Material material = {false, 0.0f, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, 0};

    // Calculate the normal of the surface
    const vec3f normalSurface = normalize(cross(world.vertices[index.y] - world.vertices[index.x],
                                                world.vertices[index.z] - world.vertices[index.x]));

    const vec3f ro = optixGetWorldRayOrigin();
    const vec3f rd = optixGetWorldRayDirection();
    const vec3f rdn = normalize(rd);

    // Calculate the position of the collision
    prd.hitOrigin = ro + optixGetRayTmax() * rd;

    // Calculate reflected direction
    vec3f directionReflect = rd - 2.0f * dot(rd, normalSurface) * normalSurface;
    if(prd.random() > material.reflectivity){ // Scattering for lambertians
        directionReflect = normalSurface + randomUnitSphere(prd.random);
    }

    // Assign final ray data based on all the above calculations
    prd.bounceDirection = directionReflect;

    // Diffuse material scattering
    prd.bounceDirection += material.diffuse * randomUnitSphere(prd.random);

    bool glossyBounce = material.gloss > 0.0f && prd.random() < material.gloss;

    vec2f uv = optixGetTriangleBarycentrics();
    vec2f textureCoord = (1.0f - uv.x - uv.y) * world.textureCoords[indexPrimitive * 3] +
                         uv.x * world.textureCoords[indexPrimitive * 3 + 1] +
                         uv.y * world.textureCoords[indexPrimitive * 3 + 2];
    vec3f colorTexture = material.textureDiffuse > -1 ? vec3f(tex2D<float4>(world.textures[material.textureDiffuse], textureCoord.x, 1.0f - textureCoord.y)) : vec3f(1.0f);

    prd.color = glossyBounce ? vec3f(1.0f) : (material.color * colorTexture);

    prd.hitDetected = !material.fullbright;
}

// Ray miss program
OPTIX_MISS_PROGRAM(miss)() {
    PerRayData& prd = getPRD<PerRayData>();

    prd.hitDetected = false;

    vec3f rayNormal = normalize(vec3f(optixGetWorldRayDirection()));
    prd.color = vec3f(min(1.0f, 1.0f + rayNormal.y));
    //prd.color = vec3f(0.0f, 0.0f, 0.0f);
}
