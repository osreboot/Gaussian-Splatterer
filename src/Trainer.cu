#include "Trainer.cuh"

#include "OpenGLIncludes.h"

#include <unordered_set>
#include <cmath>

#include <rasterizer.h>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/type_ptr.hpp>

#include "rtx/RtxHost.h"
#include "Camera.h"
#include "Config.h"
#include "ModelSplatsDevice.h"
#include "ModelSplatsHost.h"
#include "Project.h"

// CUDA kernel to convert a float-based (3-channel) image into a 32-bit integer (4-channel, fully opaque) image
__global__ void imageFloatToInt(const float* source, uint32_t* frameBuffer, const int step, const int w, const int h) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += step){
        int x = i % w;
        int y = i / w;

        frameBuffer[y * w + x] =
                (min(255, max(0, (int)(source[(y * w + x)] * 256.0f))) << 0) +
                (min(255, max(0, (int)(source[(y * w + x) + w * h] * 256.0f))) << 8) +
                (min(255, max(0, (int)(source[(y * w + x) + 2 * w * h] * 256.0f))) << 16) + (0xFF << 24);
    }
}

// CUDA kernel to calculate the loss between a 32-bit integer (4-channel, alpha omitted) image and a float-based
// (3-channel) image. Result is written to a float-based (3-channel) image (with values between 0.0 - 1.0).
__global__ void imageIntToLoss(const uint32_t* truth, const float* rasterized, float* loss, const int step, const int w, const int h) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += step){
        int x = i % w;
        int y = i / w;

        uint32_t truthRgba = truth[y * w + x];

        loss[y * w + x] = ((float)(truthRgba & 0xFF) / 255.0f) - rasterized[y * w + x];
        loss[(y * w + x) + w * h] = ((float)((truthRgba >> 8) & 0xFF) / 255.0f) - rasterized[(y * w + x) + w * h];
        loss[(y * w + x) + 2 * w * h] = ((float)((truthRgba >> 16) & 0xFF) / 255.0f) - rasterized[(y * w + x) + 2 * w * h];
    }
}

// CUDA kernel that takes splat gradient data for one training sample and adds it to an averaged buffer.
__global__ void accumulateGradients(float* varLocations,
                                    float* avgLocations, float* avgShs, float* avgScales, float* avgOpacities, float* avgRotations,
                                    const float* locations, const float* shs, const float* scales, const float* opacities, const float* rotations,
                                    const float samples, const int shCoeffs, const int step, const int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step){
        varLocations[i] += sqrtf((locations[i * 3] * locations[i * 3]) +
                                 (locations[i * 3 + 1] * locations[i * 3 + 1]) +
                                 (locations[i * 3 + 2] * locations[i * 3 + 2])) / samples;

        avgLocations[i * 3] += locations[i * 3] / samples;
        avgLocations[i * 3 + 1] += locations[i * 3 + 1] / samples;
        avgLocations[i * 3 + 2] += locations[i * 3 + 2] / samples;

        for(int s = 0; s < shCoeffs; s++) {
            avgShs[(i * 3 * shCoeffs) + 3 * s] += shs[(i * 3 * shCoeffs) + 3 * s] / samples;
            avgShs[(i * 3 * shCoeffs) + 3 * s + 1] += shs[(i * 3 * shCoeffs) + 3 * s + 1] / samples;
            avgShs[(i * 3 * shCoeffs) + 3 * s + 2] += shs[(i * 3 * shCoeffs) + 3 * s + 2] / samples;
        }

        avgScales[i * 3] += scales[i * 3] / samples;
        avgScales[i * 3 + 1] += scales[i * 3 + 1] / samples;
        avgScales[i * 3 + 2] += scales[i * 3 + 2] / samples;

        avgOpacities[i] += opacities[i] / samples;

        avgRotations[i * 4] += rotations[i * 4] / samples;
        avgRotations[i * 4 + 1] += rotations[i * 4 + 1] / samples;
        avgRotations[i * 4 + 2] += rotations[i * 4 + 2] / samples;
        avgRotations[i * 4 + 3] += rotations[i * 4 + 3] / samples;
    }
}

// CUDA kernel that takes averaged splat gradient data and applies it to existing splats (based on feature-specific
// learning rates)
__global__ void applyGradients(float* locations, float* shs, float* scales, float* opacities, float* rotations,
                               const float* gradLocations, const float* gradShs, const float* gradScales, const float* gradOpacities, const float* gradRotations,
                               const float lr, const float lrSh, const float lrScale, const float lrOpacity, const float lrRotation,
                               const float maxScale, const int shCoeffs, const int step, const int count) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += step){
        for(int f = 0; f < 3; f++) {
            locations[i * 3 + f] += gradLocations[i * 3 + f] * lr;
        }
        for(int f  = 0; f < shCoeffs * 3; f++) {
            shs[i * 3 * shCoeffs + f] += gradShs[i * 3 * shCoeffs + f] * lrSh;
        }
        for(int f = 0; f < 3; f++) {
            scales[i * 3 + f] += gradScales[i * 3 + f] * lrScale;
            scales[i * 3 + f] = min(maxScale, max(0.0f, scales[i * 3 + f]));
        }
        opacities[i] = min(1.0f, max(0.0f, opacities[i] + gradOpacities[i] * lrOpacity));
        for(int f = 0; f < 4; f++) {
            rotations[i * 4 + f] += gradRotations[i * 4 + f] * lrRotation;
        }
    }
}

Trainer::Trainer() {
    // Reserve memory used by all models/projects
    cudaMalloc(&devBackground, 3 * sizeof(float));
    cudaMalloc(&devMatView, 16 * sizeof(float));
    cudaMalloc(&devMatProjView, 16 * sizeof(float));
    cudaMalloc(&devCameraLocation, 3 * sizeof(float));
    cudaMalloc(&devRasterized, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * 3 * sizeof(float));

    // Temporary model so we don't crash. This is replaced by the UiFrame on project load.
    model = new ModelSplatsDevice(ModelSplatsHost(0, 0, 1));
}

Trainer::~Trainer() {
    cudaFree(devBackground);
    cudaFree(devMatView);
    cudaFree(devMatProjView);
    cudaFree(devCameraLocation);
    cudaFree(devRasterized);

    cudaFree(devVarLocations);
    cudaFree(devAvgGradLocations);
    cudaFree(devAvgGradShs);
    cudaFree(devAvgGradScales);
    cudaFree(devAvgGradOpacities);
    cudaFree(devAvgGradRotations);

    cudaFree(devLossPixels);

    cudaFree(devGradLocations);
    cudaFree(devGradShs);
    cudaFree(devGradScales);
    cudaFree(devGradOpacities);
    cudaFree(devGradRotations);

    cudaFree(devGradMean2D);
    cudaFree(devGradConic);
    cudaFree(devGradColor);
    cudaFree(devGradCov3D);

    delete model;

    for (uint32_t* frameBuffer : truthFrameBuffersW) cudaFree(frameBuffer);
    for (uint32_t* frameBuffer : truthFrameBuffersB) cudaFree(frameBuffer);
}

void Trainer::render(uint32_t* frameBuffer, int sizeX, int sizeY, float splatScale, const Camera& camera) {

    // Copy scene data to the GPU
    std::vector<float> background = {0.0f, 0.0f, 0.0f};
    cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    glm::mat4 matView = camera.getView();
    glm::mat4 matProjView = camera.getProjection((float)sizeX / (float)sizeY) * matView;
    cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devMatProjView, glm::value_ptr(matProjView), 16 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(devCameraLocation, &camera.location[0], 3 * sizeof(float), cudaMemcpyHostToDevice);

    // If the required render resolution matches the buffer we've already reserved for training purposes, use that instead
    const bool useBuiltinColorBuffer = sizeX == RENDER_RESOLUTION_X && sizeY == RENDER_RESOLUTION_Y;

    float* devColorBuffer = nullptr;
    if (useBuiltinColorBuffer) {
        devColorBuffer = devRasterized;
    } else cudaMalloc(&devColorBuffer, sizeX * sizeY * 3 * sizeof(float));

    // Pointers used by the rasterizer
    char* geomBuffer;
    char* binningBuffer;
    char* imgBuffer;

    // Forward rasterizer call. This renders all splats to the float-based FBO at 'devColorBuffer'
    CudaRasterizer::Rasterizer::forward(
            [&](size_t N) { cudaMalloc(&geomBuffer, N); return geomBuffer; },
            [&](size_t N) { cudaMalloc(&binningBuffer, N); return binningBuffer; },
            [&](size_t N) { cudaMalloc(&imgBuffer, N); return imgBuffer; },
            model->count,
            model->shDegree,
            model->shCoeffs,
            devBackground,
            sizeX,
            sizeY,
            model->devLocations,
            model->devShs,
            nullptr,
            model->devOpacities,
            model->devScales,
            splatScale,
            model->devRotations,
            nullptr,
            devMatView,
            devMatProjView,
            devCameraLocation,
            tan(glm::radians((float)sizeX * camera.fovDegY / (float)sizeY) * 0.5f),
            tan(glm::radians(camera.fovDegY) * 0.5f),
            false,
            devColorBuffer,
            nullptr,
            true);

    // Rasterizer outputs a float-based image, so convert it to integers
    imageFloatToInt<<<256, 256>>>(devColorBuffer, frameBuffer, 256 * 256, sizeX, sizeY);

    // Free memory used by the rasterizer
    cudaFree(geomBuffer);
    cudaFree(binningBuffer);
    cudaFree(imgBuffer);

    if (!useBuiltinColorBuffer) cudaFree(devColorBuffer);

    // Synchronize is necessary because we need to wait for the above kernel to finish, and we aren't making any
    // cudaMemcpy (blocking) calls
    cudaDeviceSynchronize();
}

void Trainer::captureTruths(const Project& project, RtxHost& rtx) {
    if (lastTruthCount != Camera::getCamerasCount(project)) { // The number of truth frames has changed, so re-allocate FBOs
        lastTruthCount = Camera::getCamerasCount(project);

        for (uint32_t* frameBuffer : truthFrameBuffersW) cudaFree(frameBuffer);
        for (uint32_t* frameBuffer : truthFrameBuffersB) cudaFree(frameBuffer);
        truthFrameBuffersW.clear();
        truthFrameBuffersB.clear();

        truthCameras.clear();

        for (const Camera& camera : Camera::getCameras(project)) {
            uint32_t* frameBufferW;
            cudaMalloc(&frameBufferW, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t));
            rtx.render(frameBufferW, {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y}, camera, {1.0f, 1.0f, 1.0f}, project.rtSamples, {});
            truthFrameBuffersW.push_back(frameBufferW);

            uint32_t* frameBufferB;
            cudaMalloc(&frameBufferB, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t));
            rtx.render(frameBufferB, {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y}, camera, {0.0f, 0.0f, 0.0f}, project.rtSamples, {});
            truthFrameBuffersB.push_back(frameBufferB);

            truthCameras.push_back(camera);
        }
    } else { // The number of truth frames is the same, so we can simply re-use them
        std::vector<Camera> cameras = Camera::getCameras(project);
        for (int i = 0; i < cameras.size(); i++) {
            rtx.render(truthFrameBuffersW.at(i), {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y}, cameras.at(i), {1.0f, 1.0f, 1.0f}, project.rtSamples, {});
            rtx.render(truthFrameBuffersB.at(i), {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y}, cameras.at(i), {0.0f, 0.0f, 0.0f}, project.rtSamples, {});
            truthCameras.at(i) = cameras.at(i);
        }
    }
}

void Trainer::train(Project& project, bool densify) {
    if (truthFrameBuffersW.empty()) throw std::runtime_error("Can't run training iteration, no truth data available!");

    project.iterations++;

    // Check if the number of splats in the model has changed. If it has, we need to re-allocate all data with
    // splat-wise dimensionality. This is also used on initialization (splat count starts at -1).
    if (model->count != lastCount) {
        lastCount = model->count;

        cudaFree(devVarLocations);
        cudaFree(devAvgGradLocations);
        cudaFree(devAvgGradShs);
        cudaFree(devAvgGradScales);
        cudaFree(devAvgGradOpacities);
        cudaFree(devAvgGradRotations);

        cudaFree(devLossPixels);

        cudaFree(devGradLocations);
        cudaFree(devGradShs);
        cudaFree(devGradScales);
        cudaFree(devGradOpacities);
        cudaFree(devGradRotations);

        cudaFree(devGradMean2D);
        cudaFree(devGradConic);
        cudaFree(devGradColor);
        cudaFree(devGradCov3D);

        cudaMalloc(&devVarLocations, model->count * sizeof(float));
        cudaMalloc(&devAvgGradLocations, model->count * 3 * sizeof(float));
        cudaMalloc(&devAvgGradShs, model->count * 3 * model->shCoeffs * sizeof(float));
        cudaMalloc(&devAvgGradScales, model->count * 3 * sizeof(float));
        cudaMalloc(&devAvgGradOpacities, model->count * sizeof(float));
        cudaMalloc(&devAvgGradRotations, model->count * 4 * sizeof(float));

        cudaMalloc(&devLossPixels, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * 3 * sizeof(float));

        cudaMalloc(&devGradLocations, model->count * 3 * sizeof(float));
        cudaMalloc(&devGradShs, model->count * 3 * model->shCoeffs * sizeof(float));
        cudaMalloc(&devGradScales, model->count * 3 * sizeof(float));
        cudaMalloc(&devGradOpacities, model->count * sizeof(float));
        cudaMalloc(&devGradRotations, model->count * 4 * sizeof(float));

        cudaMalloc(&devGradMean2D, model->count * 3 * sizeof(float));
        cudaMalloc(&devGradConic, model->count * 4 * sizeof(float));
        cudaMalloc(&devGradColor, model->count * 3 * sizeof(float));
        cudaMalloc(&devGradCov3D, model->count * 6 * sizeof(float));
    }

    // Variance and averages need to be cleared every iteration
    cudaMemset(devVarLocations, 0, model->count * sizeof(float));
    cudaMemset(devAvgGradLocations, 0, model->count * 3 * sizeof(float));
    cudaMemset(devAvgGradShs, 0, model->count * 3 * model->shCoeffs * sizeof(float));
    cudaMemset(devAvgGradScales, 0, model->count * 3 * sizeof(float));
    cudaMemset(devAvgGradOpacities, 0, model->count * sizeof(float));
    cudaMemset(devAvgGradRotations, 0, model->count * 4 * sizeof(float));

    for (int i = 0; i < truthFrameBuffersW.size() * 2; i++) {
        bool backgroundWhite = i < truthFrameBuffersW.size();
        uint32_t* truthFrameBuffer = backgroundWhite ? truthFrameBuffersW[i] : truthFrameBuffersB[i - truthFrameBuffersW.size()];
        Camera camera = truthCameras[i % truthFrameBuffersW.size()];

        // Copy scene data to the GPU
        std::vector<float> background;
        for(int c = 0; c < 3; c++) background.emplace_back(backgroundWhite ? 1.0f : 0.0f);
        cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        glm::mat4 matView = camera.getView();
        glm::mat4 matProjView = camera.getProjection((float)RENDER_RESOLUTION_X / (float)RENDER_RESOLUTION_Y) * matView;
        cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devMatProjView, glm::value_ptr(matProjView), 16 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(devCameraLocation, &camera.location[0], 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Pointers used by the rasterizer
        char* geomBuffer;
        char* binningBuffer;
        char* imgBuffer;

        // Forward rasterizer call. This renders all splats to the float-based FBO at 'devRasterized'
        int countRendered = CudaRasterizer::Rasterizer::forward(
                [&](size_t N) { cudaMalloc(&geomBuffer, N); return geomBuffer; },
                [&](size_t N) { cudaMalloc(&binningBuffer, N); return binningBuffer; },
                [&](size_t N) { cudaMalloc(&imgBuffer, N); return imgBuffer; },
                model->count,
                model->shDegree,
                model->shCoeffs,
                devBackground,
                RENDER_RESOLUTION_X,
                RENDER_RESOLUTION_Y,
                model->devLocations,
                model->devShs,
                nullptr,
                model->devOpacities,
                model->devScales,
                1.0f,
                model->devRotations,
                nullptr,
                devMatView,
                devMatProjView,
                devCameraLocation,
                tan(glm::radians(camera.fovDegY) * 0.5f),
                tan(glm::radians(camera.fovDegY) * 0.5f),
                false,
                devRasterized,
                nullptr,
                true);

        // Convert float-based image at 'devRasterized' to int-based loss values
        imageIntToLoss<<<256, 256>>>(truthFrameBuffer, devRasterized, devLossPixels, 256 * 256, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

        // The rasterizer doesn't clear gradients & frame-specific data, so we have to
        cudaMemset(devGradLocations, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradShs, 0, model->count * 3 * model->shCoeffs * sizeof(float));
        cudaMemset(devGradScales, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradOpacities, 0, model->count * sizeof(float));
        cudaMemset(devGradRotations, 0, model->count * 4 * sizeof(float));

        cudaMemset(devGradMean2D, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradConic, 0, model->count * 4 * sizeof(float));
        cudaMemset(devGradColor, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradCov3D, 0, model->count * 6 * sizeof(float));

        // Backward rasterizer call. This converts loss values into splat feature gradients that we can use for training
        CudaRasterizer::Rasterizer::backward(
                model->count,
                model->shDegree,
                model->shCoeffs,
                countRendered,
                devBackground,
                RENDER_RESOLUTION_X,
                RENDER_RESOLUTION_Y,
                model->devLocations,
                model->devShs,
                nullptr,
                model->devScales,
                1.0f,
                model->devRotations,
                nullptr,
                devMatView,
                devMatProjView,
                devCameraLocation,
                tan(glm::radians(camera.fovDegY) * 0.5f),
                tan(glm::radians(camera.fovDegY) * 0.5f),
                nullptr, // TODO pass in from forward rasterizer to increase speed
                geomBuffer,
                binningBuffer,
                imgBuffer,
                devLossPixels,
                devGradMean2D,
                devGradConic,
                devGradOpacities,
                devGradColor,
                devGradLocations,
                devGradCov3D,
                devGradShs,
                devGradScales,
                devGradRotations,
                true);

        // Add gradients to an averaged pool. Once all frame gradients are accumulated, we'll have a single value that
        // we can use as the training gradient
        accumulateGradients<<<256, 256>>>(devVarLocations,
                                          devAvgGradLocations, devAvgGradShs, devAvgGradScales, devAvgGradOpacities,devAvgGradRotations,
                                          devGradLocations, devGradShs, devGradScales, devGradOpacities, devGradRotations,
                                          (float)truthFrameBuffersW.size() * 2.0f, model->shCoeffs, 256 * 256, model->count);

        // Free memory used by the rasterizer
        cudaFree(geomBuffer);
        cudaFree(binningBuffer);
        cudaFree(imgBuffer);
    }

    // Adjust splat features based on averaged gradients collected above
    applyGradients<<<256, 256>>>(model->devLocations, model->devShs, model->devScales, model->devOpacities, model->devRotations,
                                 devAvgGradLocations, devAvgGradShs, devAvgGradScales, devAvgGradOpacities, devAvgGradRotations,
                                 project.lrLocation, project.lrSh, project.lrScale, project.lrOpacity, project.lrRotation,
                                 project.paramScaleMax, model->shCoeffs, 256 * 256, model->count);

    // 'Densify' iterations examine all splats and split/clone volatile splats and cull undesirable splats. This is done
    // on the CPU instead of GPU because the operations are complicated not easily parallelizable (it can be done, but
    // it'd require a substantial effort). This is slower, but it's only about 1/200 iterations so the tradeoff is worth
    // it.
    if(densify) {
        // Transfer splats to host memory
        ModelSplatsHost modelHost(*model);

        // Transfer variance and gradient data to host memory so we can sample it for splitting/cloning/culling splats
        float* varLocations = new float[model->count];
        float* gradLocations = new float[model->count * 3];
        cudaMemcpy(varLocations, devVarLocations, model->count * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradLocations, devAvgGradLocations, model->count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Examine splats and determine which ones need to be culled, and which ones need to be split/cloned
        std::unordered_set<int> toSplit, toClone, toRemove;
        for(int i = 0; i < modelHost.count; i++) {
            float sizeMagnitude = glm::length(glm::vec3(modelHost.scales[i * 3], modelHost.scales[i * 3 + 1], modelHost.scales[i * 3 + 2]));
            if (modelHost.opacities[i] <= project.paramCullOpacity || sizeMagnitude < project.paramCullSize) {
                toRemove.insert(i);
            } else if (varLocations[i] - glm::length(glm::vec3(gradLocations[i * 3], gradLocations[i * 3 + 1], gradLocations[i * 3 + 2])) > project.paramDensifyVariance) {
                if (sizeMagnitude > project.paramSplitSize) toSplit.insert(i); else toClone.insert(i);
            }
        }

        // Split larger volatile splats
        for (int i : toSplit) {
            if (modelHost.count < modelHost.capacity) {
                glm::vec3 locPre(modelHost.locations[i * 3], modelHost.locations[i * 3 + 1], modelHost.locations[i * 3 + 2]);
                glm::vec3 scalePre(modelHost.scales[i * 3], modelHost.scales[i * 3 + 1], modelHost.scales[i * 3 + 2]);
                glm::quat rotPre(modelHost.rotations[i * 4], modelHost.rotations[i * 4 + 1], modelHost.rotations[i * 4 + 2], modelHost.rotations[i * 4 + 3]);

                // Identify the splat's largest scale axis
                glm::vec4 scaleOffset4(scalePre.x, scalePre.y, scalePre.z, 1.0f);
                if (scalePre.x > scalePre.y && scalePre.x > scalePre.z) {
                    scaleOffset4 *= glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                } else if (scalePre.y > scalePre.z) {
                    scaleOffset4 *= glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                } else {
                    scaleOffset4 *= glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
                }

                scaleOffset4 = (glm::mat4)rotPre * scaleOffset4;

                // Calculate new locations (separated along the original's largest scale axis)
                glm::vec3 locNew1 = locPre + glm::vec3(scaleOffset4.x / scaleOffset4.w, scaleOffset4.y / scaleOffset4.w, scaleOffset4.z / scaleOffset4.w) * project.paramSplitDistance * 0.5f;
                glm::vec3 locNew2 = locPre - glm::vec3(scaleOffset4.x / scaleOffset4.w, scaleOffset4.y / scaleOffset4.w, scaleOffset4.z / scaleOffset4.w) * project.paramSplitDistance * 0.5f;

                glm::vec3 scaleNew = glm::vec3(modelHost.scales[i * 3], modelHost.scales[i * 3 + 1], modelHost.scales[i * 3 + 2]) * project.paramSplitScale;

                // Create a duplicate splat at the end of the array
                int i2 = modelHost.count;
                modelHost.count++;
                modelHost.copy(i2, i);

                // Copy splat features and set new locations
                memcpy(&modelHost.locations[i * 3], &locNew1[0], 3 * sizeof(float));
                memcpy(&modelHost.locations[i2 * 3], &locNew2[0], 3 * sizeof(float));
                memcpy(&modelHost.scales[i * 3], &scaleNew[0], 3 * sizeof(float));
                memcpy(&modelHost.scales[i2 * 3], &scaleNew[0], 3 * sizeof(float));
                memcpy(&modelHost.rotations[i * 4], &rotPre[0], 4 * sizeof(float));
                memcpy(&modelHost.rotations[i2 * 4], &rotPre[0], 4 * sizeof(float));
            }
        }

        // Clone smaller volatile splats
        for (int i : toClone) {
            if (modelHost.count < modelHost.capacity) {
                glm::vec3 loc(modelHost.locations[i * 3], modelHost.locations[i * 3 + 1], modelHost.locations[i * 3 + 2]);
                glm::vec3 scale(modelHost.scales[i * 3], modelHost.scales[i * 3 + 1], modelHost.scales[i * 3 + 2]);
                glm::quat rot(modelHost.rotations[i * 4], modelHost.rotations[i * 4 + 1], modelHost.rotations[i * 4 + 2], modelHost.rotations[i * 4 + 3]);

                // Calculate the direction that the splat is being pulled in
                glm::vec3 dirGradient = glm::normalize(glm::vec3(gradLocations[i * 3], gradLocations[i * 3 + 1], gradLocations[i * 3 + 2]));

                glm::vec4 offset4 = (glm::mat4)rot * glm::vec4(scale.x, scale.y, scale.z, 1.0f);

                // 'loc' is a location for the clone that's offset (by a fixed amount) in the direction of the original's location gradient
                loc += glm::vec3(offset4.x / offset4.w, offset4.y / offset4.w, offset4.z / offset4.w) * dirGradient * project.paramCloneDistance;

                // Create a duplicate splat at the end of the array
                int i2 = modelHost.count;
                modelHost.count++;
                modelHost.copy(i2, i);

                // Set the new splat's location
                memcpy(&modelHost.locations[i2 * 3], &loc[0], 3 * sizeof(float));
            }
        }

        // Prune small/transparent splats
        if (!toRemove.empty()) {
            // Iterate over the entire splat list and collapse any voids left by removed splats
            int indexPreserved = 0;
            for (int indexScan = 0; indexScan < modelHost.count; indexScan++) {
                if (!toRemove.count(indexScan)) {
                    if (indexPreserved != indexScan) modelHost.copy(indexPreserved, indexScan);
                    indexPreserved++;
                }
            }
            modelHost.count -= (int)toRemove.size();
        }

        delete[] varLocations;
        delete[] gradLocations;

        // Transfer splats back to the GPU
        delete model;
        model = new ModelSplatsDevice(modelHost);
    }
}
