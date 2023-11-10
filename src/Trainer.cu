#include <unordered_set>
#include <cmath>

#include "ui/UiPanelInput.h"

#include "Trainer.cuh"
#include "TruthCameras.h"

#include <rasterizer.h>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/type_ptr.hpp>

__global__ void convertIntKernel(const float* source, uint32_t* frameBuffer, int step, int w, int h) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += step){
        int x = i % w;
        int y = i / w;

        frameBuffer[y * w + x] =
                (min(255, max(0, (int)(source[(y * w + x)] * 256.0f))) << 0) +
                (min(255, max(0, (int)(source[(y * w + x) + w * h] * 256.0f))) << 8) +
                (min(255, max(0, (int)(source[(y * w + x) + 2 * w * h] * 256.0f))) << 16) + (0xFF << 24);
    }
}

__global__ void lossKernel(const uint32_t* truth, const float* rasterized, float* loss, int step, int w, int h) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += step){
        int x = i % w;
        int y = i / w;

        uint32_t truthRgba = truth[y * w + x];

        loss[y * w + x] = ((float)(truthRgba & 0xFF) / 255.0f) - rasterized[y * w + x];
        loss[(y * w + x) + w * h] = ((float)((truthRgba >> 8) & 0xFF) / 255.0f) - rasterized[(y * w + x) + w * h];
        loss[(y * w + x) + 2 * w * h] = ((float)((truthRgba >> 16) & 0xFF) / 255.0f) - rasterized[(y * w + x) + 2 * w * h];
    }
}

__global__ void gradientSumKernel(float* avgLocations, float* avgShs, float* avgScales, float* avgOpacities, float* avgRotations,
                                      const float* locations, const float* shs, const float* scales, const float* opacities, const float* rotations,
                                      float samples, int shCoeffs, int step, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step){
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

__global__ void locationVarianceKernel(float* varLocations, const float* locations, float samples, int step, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step){
        varLocations[i] += sqrtf((locations[i * 3] * locations[i * 3]) +
                (locations[i * 3 + 1] * locations[i * 3 + 1]) +
                (locations[i * 3 + 2] * locations[i * 3 + 2])) / samples;
    }
}

Trainer::Trainer() {
    cudaMalloc(&devBackground, 3 * sizeof(float));
    cudaMalloc(&devMatView, 16 * sizeof(float));
    cudaMalloc(&devMatProjView, 16 * sizeof(float));
    cudaMalloc(&devCameraLocation, 3 * sizeof(float));
    cudaMalloc(&devRasterized, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * 3 * sizeof(float));

    model = new ModelSplats(1000000, 1, 4);

    for(int x = 0; x < 5; x++){
        for(int y = 0; y < 5; y++){
            for(int z = 0; z < 5; z++){
                model->pushBack({(float)x - 2.0f, (float)y - 2.0f, (float)z - 2.0f},
                                {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {0.2f, 0.2f, 0.2f},
                                1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
            }
        }
    }

    /*
    model->pushBack({0.0f, 4.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.1f},
                    1.0f, glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)));

    model->pushBack({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 2.0f},
                    1.0f, glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)));*/
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

    delete[] varLocations;
    delete[] avgGradLocations;
    delete[] avgGradShs;
    delete[] avgGradScales;
    delete[] avgGradOpacities;
    delete[] avgGradRotations;

    delete model;

    for (uint32_t* frameBuffer : truthFrameBuffers) {
        cudaFree(frameBuffer);
    }
}

void Trainer::render(uint32_t* frameBuffer, const Camera& camera) {
    model->deviceBuffer();

    std::vector<float> background = {0.0f, 0.0f, 0.0f};
    cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    glm::mat4 matView = -glm::lookAt(TruthCameras::toGlmVec(camera.location),
                                     TruthCameras::toGlmVec(camera.target), {0.0f, 1.0f, 0.0f});
    cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);

    glm::mat4 matProjView = glm::perspective(glm::radians(camera.degFovY),
                                             (float)RENDER_RESOLUTION_X / (float)RENDER_RESOLUTION_Y, 0.1f, 100.0f) * matView;
    cudaMemcpy(devMatProjView, glm::value_ptr(matProjView), 16 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(devCameraLocation, &camera.location[0], 3 * sizeof(float), cudaMemcpyHostToDevice);

    char* geomBuffer;
    char* binningBuffer;
    char* imgBuffer;

    CudaRasterizer::Rasterizer::forward(
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
            tan(glm::radians(camera.degFovX) * 0.5f),
            tan(glm::radians(camera.degFovY) * 0.5f),
            false,
            devRasterized,
            nullptr,
            true);

    convertIntKernel<<<256, 256>>>(devRasterized, frameBuffer, 256 * 256, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

    cudaFree(geomBuffer);
    cudaFree(binningBuffer);
    cudaFree(imgBuffer);
}

void Trainer::captureTruths(const TruthCameras& cameras, RtxHost& rtx) {
    for (uint32_t* frameBuffer : truthFrameBuffers) {
        cudaFree(frameBuffer);
    }
    truthFrameBuffers.clear();
    truthCameras.clear();

    for (int i = 0; i < cameras.getCount(); i++) {
        uint32_t* frameBuffer;
        Camera camera = cameras.getCamera(i);
        cudaMalloc(&frameBuffer, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t));
        rtx.render(frameBuffer, camera, nullptr);
        truthFrameBuffers.push_back(frameBuffer);
        truthCameras.push_back(camera);
    }
}

void Trainer::train(int iterations) {
    for(int i = 0; i < iterations; i++) train(false);
}

void Trainer::train(bool densify) {
    assert(truths.size() > 0);

    iterations++;

    model->deviceBuffer();

    bool dirty = model->count != lastCount;
    lastCount = model->count;

    if(dirty) {
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

        delete[] varLocations;
        delete[] avgGradLocations;
        delete[] avgGradShs;
        delete[] avgGradScales;
        delete[] avgGradOpacities;
        delete[] avgGradRotations;

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

        varLocations = new float[model->count];
        avgGradLocations = new float[model->count * 3];
        avgGradShs = new float[model->count * 3 * model->shCoeffs];
        avgGradScales = new float[model->count * 3];
        avgGradOpacities = new float[model->count];
        avgGradRotations = new float[model->count * 4];
    }

    cudaMemset(devVarLocations, 0, model->count * sizeof(float));
    cudaMemset(devAvgGradLocations, 0, model->count * 3 * sizeof(float));
    cudaMemset(devAvgGradShs, 0, model->count * 3 * model->shCoeffs * sizeof(float));
    cudaMemset(devAvgGradScales, 0, model->count * 3 * sizeof(float));
    cudaMemset(devAvgGradOpacities, 0, model->count * sizeof(float));
    cudaMemset(devAvgGradRotations, 0, model->count * 4 * sizeof(float));

    for (int i = 0; i < truthFrameBuffers.size(); i++) {
        uint32_t* truthFrameBuffer = truthFrameBuffers[i];
        Camera camera = truthCameras[i];

        std::vector<float> background = {0.0f, 0.0f, 0.0f};
        cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        glm::mat4 matView = -glm::lookAt(TruthCameras::toGlmVec(camera.location),
                                         TruthCameras::toGlmVec(camera.target), {0.0f, 1.0f, 0.0f});
        cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);

        glm::mat4 matProjView = glm::perspective(glm::radians(camera.degFovY),
                                                 (float)RENDER_RESOLUTION_X / (float)RENDER_RESOLUTION_Y, 0.1f, 100.0f) * matView;
        cudaMemcpy(devMatProjView, glm::value_ptr(matProjView), 16 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(devCameraLocation, &camera.location[0], 3 * sizeof(float), cudaMemcpyHostToDevice);

        char* geomBuffer;
        char* binningBuffer;
        char* imgBuffer;

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
                tan(glm::radians(camera.degFovX) * 0.5f),
                tan(glm::radians(camera.degFovY) * 0.5f),
                false,
                devRasterized,
                nullptr,
                true);

        lossKernel<<<256, 256>>>(truthFrameBuffer, devRasterized, devLossPixels, 256 * 256, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

        cudaMemset(devGradLocations, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradShs, 0, model->count * 3 * model->shCoeffs * sizeof(float));
        cudaMemset(devGradScales, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradOpacities, 0, model->count * sizeof(float));
        cudaMemset(devGradRotations, 0, model->count * 4 * sizeof(float));

        cudaMemset(devGradMean2D, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradConic, 0, model->count * 4 * sizeof(float));
        cudaMemset(devGradColor, 0, model->count * 3 * sizeof(float));
        cudaMemset(devGradCov3D, 0, model->count * 6 * sizeof(float));

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
                tan(glm::radians(camera.degFovX) * 0.5f),
                tan(glm::radians(camera.degFovY) * 0.5f),
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

        gradientSumKernel<<<256, 256>>>(devAvgGradLocations, devAvgGradShs, devAvgGradScales, devAvgGradOpacities, devAvgGradRotations,
                                        devGradLocations, devGradShs, devGradScales, devGradOpacities, devGradRotations,
                                        (float)truthFrameBuffers.size(), model->shCoeffs, 256 * 256, model->count);

        locationVarianceKernel<<<256, 256>>>(devVarLocations, devGradLocations, (float)truthFrameBuffers.size(), 256 * 256, model->count);

        cudaFree(geomBuffer);
        cudaFree(binningBuffer);
        cudaFree(imgBuffer);
    }

    cudaMemcpy(varLocations, devVarLocations, model->count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(avgGradLocations, devAvgGradLocations, model->count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(avgGradShs, devAvgGradShs, model->count * 3 * model->shCoeffs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(avgGradScales, devAvgGradScales, model->count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(avgGradOpacities, devAvgGradOpacities, model->count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(avgGradRotations, devAvgGradRotations, model->count * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    static const float learningRate = 0.0001f;
    static const float learningRateRot = 0.00002f;

    // Apply gradients
    std::unordered_set<int> toSplit, toRemove;
    for(int i = 0; i < model->count; i++) {
        for(int f = 0; f < 3; f++) {
            model->locations[i * 3 + f] += avgGradLocations[i * 3 + f] * learningRate;
        }
        for(int f  = 0; f < model->shCoeffs * 3; f++) {
            model->shs[i * 3 * model->shCoeffs + f] += avgGradShs[i * 3 * model->shCoeffs + f] * learningRate;
        }
        for(int f = 0; f < 3; f++) {
            model->scales[i * 3 + f] += avgGradScales[i * 3 + f] * learningRate;
            model->scales[i * 3 + f] = std::max(0.0f, model->scales[i * 3 + f]);
        }
        model->opacities[i] = std::min(1.0f, std::max(0.0f, model->opacities[i] + avgGradOpacities[i] * learningRate));
        for(int f = 0; f < 4; f++) {
            model->rotations[i * 4 + f] += avgGradRotations[i * 4 + f] * learningRateRot;
        }

        if(varLocations[i] > 10.0f) toSplit.insert(i);

        if(model->opacities[i] <= 0.001f) toRemove.insert(i);
        if(glm::length(glm::vec3(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2])) < 0.01f) toRemove.insert(i);
    }

    if(densify) {
        // Split volatile splats
        for (int i : toSplit) {
            if (model->count < model->capacity) {
                glm::vec3 locPre(model->locations[i * 3], model->locations[i * 3 + 1], model->locations[i * 3 + 2]);
                glm::vec4 scalePre4(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2], 1.0f);
                glm::quat rotPre(model->rotations[i * 4], model->rotations[i * 4 + 1], model->rotations[i * 4 + 2], model->rotations[i * 4 + 3]);

                scalePre4 = (glm::mat4)rotPre * scalePre4;
                glm::vec3 scalePre(scalePre4.x / scalePre4.w, scalePre4.y / scalePre4.w, scalePre4.z / scalePre4.w);

                glm::quat rotZero = glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f));

                glm::vec3 locNew1 = locPre + scalePre * 0.5f;
                glm::vec3 locNew2 = locPre - scalePre * 0.5f;

                glm::vec3 scaleNew = glm::vec3(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2]) / 1.6f;

                int i2 = model->count;
                model->count++;
                model->copy(i2, i); // Create a duplicate splat at the end of the array

                memcpy(&model->locations[i * 3], &locNew1[0], 3 * sizeof(float));
                memcpy(&model->locations[i2 * 3], &locNew2[0], 3 * sizeof(float));
                memcpy(&model->scales[i * 3], &scaleNew[0], 3 * sizeof(float));
                memcpy(&model->scales[i2 * 3], &scaleNew[0], 3 * sizeof(float));
                memcpy(&model->rotations[i * 4], &rotPre[0], 4 * sizeof(float));
                memcpy(&model->rotations[i2 * 4], &rotPre[0], 4 * sizeof(float));
            }
        }

        // Prune small/transparent splats
        if (!toRemove.empty()) {
            int indexPreserved = 0;
            for(int indexScan = 0; indexScan < model->count; indexScan++) {
                if (!toRemove.count(indexScan)) {
                    if (indexPreserved != indexScan) model->copy(indexPreserved, indexScan);
                    indexPreserved++;
                }
            }
            model->count -= (int)toRemove.size();
        }
    }
}
