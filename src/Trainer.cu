#include <unordered_set>
#include <cmath>
#include <fstream>

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

__global__ void locationVarianceKernel(float* varLocations, const float* gradLocations, float samples, int step, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step){
        varLocations[i] += sqrtf((gradLocations[i * 3] * gradLocations[i * 3]) +
                                 (gradLocations[i * 3 + 1] * gradLocations[i * 3 + 1]) +
                                 (gradLocations[i * 3 + 2] * gradLocations[i * 3 + 2])) / samples;
    }
}

Trainer::Trainer() {
    cudaMalloc(&devBackground, 3 * sizeof(float));
    cudaMalloc(&devMatView, 16 * sizeof(float));
    cudaMalloc(&devMatProjView, 16 * sizeof(float));
    cudaMalloc(&devCameraLocation, 3 * sizeof(float));
    cudaMalloc(&devRasterized, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * 3 * sizeof(float));

    // Scene-sized cube initialization
    model = new ModelSplats(1000000, 1, 4);

    static const float dim = 4.0f;
    static const float step = 0.5f;

    for(float x = -dim; x <= dim; x += step){
        for(float y = -dim; y <= dim; y += step){
            for(float z = -dim; z <= dim; z += step){
                model->pushBack({x, y, z}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {step * 0.1f, step * 0.1f, step * 0.1f},
                                1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
                //model->pushBack({x, y, z}, {0.0f, 0.0f, 0.0f}, {step * 0.1f, step * 0.1f, step * 0.1f},
                //                1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
            }
        }
    }

    //model->pushBack({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.1f, 0.1f, 0.3f},
    //                1.0f, glm::angleAxis(45.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
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

    for (uint32_t* frameBuffer : truthFrameBuffersW) cudaFree(frameBuffer);
    for (uint32_t* frameBuffer : truthFrameBuffersB) cudaFree(frameBuffer);
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
    for (uint32_t* frameBuffer : truthFrameBuffersW) cudaFree(frameBuffer);
    for (uint32_t* frameBuffer : truthFrameBuffersB) cudaFree(frameBuffer);
    truthFrameBuffersW.clear();
    truthFrameBuffersB.clear();
    truthCameras.clear();

    for (int i = 0; i < cameras.getCount(); i++) {
        Camera camera = cameras.getCamera(i);

        uint32_t* frameBufferW;
        cudaMalloc(&frameBufferW, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t));
        rtx.render(frameBufferW, camera, {1.0f, 1.0f, 1.0f}, nullptr);
        truthFrameBuffersW.push_back(frameBufferW);

        uint32_t* frameBufferB;
        cudaMalloc(&frameBufferB, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t));
        rtx.render(frameBufferB, camera, {0.0f, 0.0f, 0.0f}, nullptr);
        truthFrameBuffersB.push_back(frameBufferB);

        truthCameras.push_back(camera);
    }
}

void Trainer::train(int iter) {
    for(int i = 0; i < iter; i++) train(false);
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

    for (int i = 0; i < truthFrameBuffersW.size() * 2; i++) {
        bool backgroundWhite = i < truthFrameBuffersW.size();
        uint32_t* truthFrameBuffer = backgroundWhite ? truthFrameBuffersW[i] : truthFrameBuffersB[i - truthFrameBuffersW.size()];
        Camera camera = truthCameras[i % truthFrameBuffersW.size()];

        std::vector<float> background;
        for(int c = 0; c < 3; c++) background.emplace_back(backgroundWhite ? 1.0f : 0.0f);
        cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        glm::mat4 matView = camera.getView();
        glm::mat4 matProjView = camera.getProjection() * matView;
        cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);
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
                                        (float)truthFrameBuffersW.size() * 2.0f, model->shCoeffs, 256 * 256, model->count);

        locationVarianceKernel<<<256, 256>>>(devVarLocations, devGradLocations, (float)truthFrameBuffersW.size() * 2.0f, 256 * 256, model->count);

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

    static const float learningRate = 0.00005f;
    static const float learningRateShs = learningRate * 2.0f;
    static const float learningRateOpacity = learningRate * 2.0f;
    static const float learningRateRotation = learningRate / 2.0f;

    // Apply gradients
    std::unordered_set<int> toSplit, toClone, toRemove;
    for(int i = 0; i < model->count; i++) {
        for(int f = 0; f < 3; f++) {
            model->locations[i * 3 + f] += avgGradLocations[i * 3 + f] * learningRate;
        }
        for(int f  = 0; f < model->shCoeffs * 3; f++) {
            model->shs[i * 3 * model->shCoeffs + f] += avgGradShs[i * 3 * model->shCoeffs + f] * learningRateShs;
        }
        for(int f = 0; f < 3; f++) {
            model->scales[i * 3 + f] += avgGradScales[i * 3 + f] * learningRate;
            model->scales[i * 3 + f] = std::min(0.3f, std::max(0.0f, model->scales[i * 3 + f]));
        }
        model->opacities[i] = std::min(1.0f, std::max(0.0f, model->opacities[i] + avgGradOpacities[i] * learningRateOpacity));
        for(int f = 0; f < 4; f++) {
            model->rotations[i * 4 + f] += avgGradRotations[i * 4 + f] * learningRateRotation;
        }

        if (model->opacities[i] <= 0.005f || glm::length(glm::vec3(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2])) < 0.0001f) {
            toRemove.insert(i);
        } else if (varLocations[i] > 2.0f) {
            if (glm::length(glm::vec3(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2])) > 0.02f) toSplit.insert(i);
            else toClone.insert(i);
        }
    }

    if(densify) {
        for (int i : toSplit) {
            if (model->count < model->capacity) {
                glm::vec3 locPre(model->locations[i * 3], model->locations[i * 3 + 1], model->locations[i * 3 + 2]);
                glm::vec3 scalePre(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2]);
                glm::quat rotPre(model->rotations[i * 4], model->rotations[i * 4 + 1], model->rotations[i * 4 + 2], model->rotations[i * 4 + 3]);

                glm::vec4 scaleOffset4(scalePre.x, scalePre.y, scalePre.z, 1.0f);
                if (scalePre.x > scalePre.y && scalePre.x > scalePre.z) {
                    scaleOffset4 *= glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                } else if (scalePre.y > scalePre.z) {
                    scaleOffset4 *= glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                } else {
                    scaleOffset4 *= glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
                }

                scaleOffset4 = (glm::mat4)rotPre * scaleOffset4;

                glm::vec3 locNew1 = locPre + glm::vec3(scaleOffset4.x, scaleOffset4.y, scaleOffset4.z) * 0.75f;
                glm::vec3 locNew2 = locPre - glm::vec3(scaleOffset4.x, scaleOffset4.y, scaleOffset4.z) * 0.75f;

                glm::vec3 scaleNew = glm::vec3(model->scales[i * 3], model->scales[i * 3 + 1], model->scales[i * 3 + 2]) / 2.0f;

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

        for (int i : toClone) {
            if (model->count < model->capacity) {
                glm::vec3 loc(model->locations[i * 3], model->locations[i * 3 + 1], model->locations[i * 3 + 2]);
                loc += glm::vec3(avgGradLocations[i * 3] * learningRate,
                                 avgGradLocations[i * 3 + 1] * learningRate,
                                 avgGradLocations[i * 3 + 2] * learningRate);

                int i2 = model->count;
                model->count++;
                model->copy(i2, i); // Create a duplicate splat at the end of the array

                memcpy(&model->locations[i2 * 3], &loc[0], 3 * sizeof(float));
            }
        }

        // Prune small/transparent splats
        if (!toRemove.empty()) {
            int indexPreserved = 0;
            for (int indexScan = 0; indexScan < model->count; indexScan++) {
                if (!toRemove.count(indexScan)) {
                    if (indexPreserved != indexScan) model->copy(indexPreserved, indexScan);
                    indexPreserved++;
                }
            }
            model->count -= (int)toRemove.size();
        }
    }
}
