#include <unordered_set>
#include <cmath>
#include <fstream>

#include "ui/UiPanelInput.h"

#include "Trainer.cuh"
#include "TruthCameras.h"
#include "ModelSplatsHost.h"

#include <rasterizer.h>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/type_ptr.hpp>

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

__global__ void accumulateGradients(float* avgLocations, float* avgShs, float* avgScales, float* avgOpacities, float* avgRotations,
                                    const float* locations, const float* shs, const float* scales, const float* opacities, const float* rotations,
                                    const float samples, const int shCoeffs, const int step, const int n) {
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

__global__ void accumulateVariance(float* varLocations, const float* gradLocations, const float samples, const int step, const int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step){
        varLocations[i] += sqrtf((gradLocations[i * 3] * gradLocations[i * 3]) +
                                 (gradLocations[i * 3 + 1] * gradLocations[i * 3 + 1]) +
                                 (gradLocations[i * 3 + 2] * gradLocations[i * 3 + 2])) / samples;
    }
}

__device__ float length(float x, float y, float z) {
    return sqrt(x * x + y * y + z * z);
}

__global__ void applyGradients(float* locations, float* shs, float* scales, float* opacities, float* rotations,
                               const float* gradLocations, const float* gradShs, const float* gradScales, const float* gradOpacities, const float* gradRotations,
                               const float* varLocations,
                               const float lr, const float lrSh, const float lrOpacity, const float lrRotation,
                               int* flexIndices, int shCoeffs, const int step, const int count) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += step){
        for(int f = 0; f < 3; f++) {
            locations[i * 3 + f] += gradLocations[i * 3 + f] * lr;
        }
        for(int f  = 0; f < shCoeffs * 3; f++) {
            shs[i * 3 * shCoeffs + f] += gradShs[i * 3 * shCoeffs + f] * lrSh;
        }
        for(int f = 0; f < 3; f++) {
            scales[i * 3 + f] += gradScales[i * 3 + f] * lr;
            scales[i * 3 + f] = min(0.3f, max(0.0f, scales[i * 3 + f]));
        }
        opacities[i] = min(1.0f, max(0.0f, opacities[i] + gradOpacities[i] * lrOpacity));
        for(int f = 0; f < 4; f++) {
            rotations[i * 4 + f] += gradRotations[i * 4 + f] * lrRotation;
        }

        if (flexIndices) {
            flexIndices[i] = 0;
            if (opacities[i] <= 0.005f || length(scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]) < 0.0001f) {
                flexIndices[i] = -1; // delete splat
            } else if (varLocations[i] > 2.0f) flexIndices[i] = 1; // split/clone splat
        }
    }
}

__global__ void flexBlockOffsets(int* flexIndices, int* flexSizes, const int step, const int count) {
    for(int t = blockIdx.x * blockDim.x + threadIdx.x; t <= count / 256; t += step){
        int size = 0;
        for(int i = t * 256; i < min((t + 1) * 256, count); i++) {
            int old = flexIndices[i];
            flexIndices[i] = old == -1 ? -1 : size;
            size += old + 1;
        }
        flexSizes[t] = size;
    }
}

__global__ void flexBlockAccumulate(int* flexSizes, const int count, int* countUpdated) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        int last = flexSizes[0];
        flexSizes[0] = 0;
        for(int i = 1; i <= count / 256; i++) {
            int original = flexSizes[i];
            flexSizes[i] = last;
            last += original;
        }
        *countUpdated = last;
    }
}

__global__ void applyDensify(float* toLocations, float* toShs, float* toScales, float* toOpacities, float* toRotations,
                             const float* fromLocations, const float* fromShs, const float* fromScales, const float* fromOpacities, const float* fromRotations,
                             const float* gradLocations, const float lr,
                             const int* flexIndices, const int* flexSizes, const int shCoeffs, const int step, const int count) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += step){
        if (flexIndices[i] != -1) { // if the splat hasn't been removed

            //if (length(scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]) > 0.02f) split
            //else clone

            int t1 = flexIndices[i] + flexSizes[i / 256];
            int t2 = flexIndices[i] + flexSizes[i / 256] + 1;

            // TODO determine new positions here

            for(int f = 0; f < 3; f++){
                toLocations[t1 * 3 + f] = fromLocations[i * 3 + f];
                toLocations[t2 * 3 + f] = fromLocations[i * 3 + f] + (gradLocations[i * 3 + f] * lr);
            }
            for(int f = 0; f < 3 * shCoeffs; f++){
                toShs[t1 * 3 * shCoeffs + f] = fromShs[i * 3 * shCoeffs + f];
                toShs[t2 * 3 * shCoeffs + f] = fromShs[i * 3 * shCoeffs + f];
            }
            for(int f = 0; f < 3; f++){
                toScales[t1 * 3 + f] = fromScales[i * 3 + f];
                toScales[t2 * 3 + f] = fromScales[i * 3 + f];
            }
            toOpacities[t1] = fromOpacities[i];
            toOpacities[t2] = fromOpacities[i];
            for(int f = 0; f < 4; f++){
                toRotations[t1 * 4 + f] = fromRotations[i * 4 + f];
                toRotations[t2 * 4 + f] = fromRotations[i * 4 + f];
            }
        }
    }
}

Trainer::Trainer() {
    cudaMalloc(&devBackground, 3 * sizeof(float));
    cudaMalloc(&devMatView, 16 * sizeof(float));
    cudaMalloc(&devMatProjView, 16 * sizeof(float));
    cudaMalloc(&devCameraLocation, 3 * sizeof(float));
    cudaMalloc(&devRasterized, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * 3 * sizeof(float));

    // Scene-sized cube initialization
    ModelSplatsHost modelHost(1000000, 1, 4);

    static const float dim = 4.0f;
    static const float step = 0.5f;

    for(float x = -dim; x <= dim; x += step){
        for(float y = -dim; y <= dim; y += step){
            for(float z = -dim; z <= dim; z += step){
                modelHost.pushBack({x, y, z}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {step * 0.1f, step * 0.1f, step * 0.1f},
                                1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
                //modelHost.pushBack({x, y, z}, {0.0f, 0.0f, 0.0f}, {step * 0.1f, step * 0.1f, step * 0.1f},
                //                1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
            }
        }
    }

    //modelHost.pushBack({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.1f, 0.1f, 0.3f},
    //                1.0f, glm::angleAxis(45.0f, glm::vec3(0.0f, 1.0f, 0.0f)));

    model = new ModelSplatsDevice(modelHost);
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

    cudaFree(devFlexIndices);
    cudaFree(devFlexSizes);

    delete model;

    for (uint32_t* frameBuffer : truthFrameBuffersW) cudaFree(frameBuffer);
    for (uint32_t* frameBuffer : truthFrameBuffersB) cudaFree(frameBuffer);
}

void Trainer::render(uint32_t* frameBuffer, const Camera& camera) {
    std::vector<float> background = {0.0f, 0.0f, 0.0f};
    cudaMemcpy(devBackground, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    glm::mat4 matView = camera.getView();
    glm::mat4 matProjView = camera.getProjection() * matView;
    cudaMemcpy(devMatView, glm::value_ptr(matView), 16 * sizeof(float), cudaMemcpyHostToDevice);
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

    imageFloatToInt<<<256, 256>>>(devRasterized, frameBuffer, 256 * 256, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

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

    if (model->count != lastCount) {
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

        lastCount = model->count;
    }

    if (lastCapacity != model->capacity) {
        cudaFree(devFlexIndices);
        cudaMalloc(&devFlexIndices, model->capacity * sizeof(int));

        cudaFree(devFlexSizes);
        cudaMalloc(&devFlexSizes, ((model->capacity / 256) + 1) * sizeof(int));

        lastCapacity = model->capacity;
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

        imageIntToLoss<<<256, 256>>>(truthFrameBuffer, devRasterized, devLossPixels, 256 * 256, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

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

        accumulateGradients<<<256, 256>>>(devAvgGradLocations, devAvgGradShs, devAvgGradScales, devAvgGradOpacities,
                                          devAvgGradRotations,
                                          devGradLocations, devGradShs, devGradScales, devGradOpacities, devGradRotations,
                                          (float)truthFrameBuffersW.size() * 2.0f, model->shCoeffs, 256 * 256, model->count);

        accumulateVariance<<<256, 256>>>(devVarLocations, devGradLocations, (float)truthFrameBuffersW.size() * 2.0f,
                                       256 * 256, model->count);

        cudaFree(geomBuffer);
        cudaFree(binningBuffer);
        cudaFree(imgBuffer);
    }

    static const float learningRate = 0.00005f;
    static const float learningRateShs = learningRate * 2.0f;
    static const float learningRateOpacity = learningRate * 2.0f;
    static const float learningRateRotation = learningRate / 2.0f;

    applyGradients<<<256, 256>>>(model->devLocations, model->devShs, model->devScales, model->devOpacities, model->devRotations,
                                 devAvgGradLocations, devAvgGradShs, devAvgGradScales, devAvgGradOpacities, devAvgGradRotations,
                                 devVarLocations,
                                 learningRate, learningRateShs, learningRateOpacity, learningRateRotation,
                                 densify ? devFlexIndices : nullptr, model->shCoeffs, 256 * 256, model->count);

    if (densify) {
        flexBlockOffsets<<<256, 1>>>(devFlexIndices, devFlexSizes, 256, model->count);

        int* devCountUpdated;
        cudaMalloc(&devCountUpdated, sizeof(int));
        flexBlockAccumulate<<<1, 1>>>(devFlexSizes, model->count, devCountUpdated);
        int* countUpdated = new int;
        cudaMemcpy(countUpdated, devCountUpdated, sizeof(int), cudaMemcpyDeviceToHost);
        //wxLogMessage(wxString(std::to_string(*countUpdated)));

        ModelSplatsDevice* model2 = new ModelSplatsDevice(*model);
        model2->count = std::min(*countUpdated, model2->capacity);
        delete countUpdated;

        applyDensify<<<256, 256>>>(model2->devLocations, model2->devShs, model2->devScales, model2->devOpacities, model2->devRotations,
                                   model->devLocations, model->devShs, model->devScales, model->devOpacities, model->devRotations,
                                   devAvgGradLocations, learningRate,
                                   devFlexIndices, devFlexSizes, model->shCoeffs, 256 * 256, model->count);

        delete model;
        model = model2;
    }

    /*
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
    }*/
}
