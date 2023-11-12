#include "ModelSplats.h"
#include "ui/UiPanelInput.h"

ModelSplats::ModelSplats(int capacity, int shDegree, int shCoeffs) :
    capacity(capacity), shDegree(shDegree), shCoeffs(shCoeffs) {

    locations = new float[capacity * 3];
    shs = new float[capacity * 3 * shCoeffs];
    scales = new float[capacity * 3];
    opacities = new float[capacity];
    rotations = new float[capacity * 4];

    cudaMalloc(&devLocations, capacity * 3 * sizeof(float));
    cudaMalloc(&devShs, capacity * 3 * shCoeffs * sizeof(float));
    cudaMalloc(&devScales, capacity * 3 * sizeof(float));
    cudaMalloc(&devOpacities, capacity * sizeof(float));
    cudaMalloc(&devRotations, capacity * 4 * sizeof(float));
}

ModelSplats::~ModelSplats() {
    delete[] locations;
    delete[] shs;
    delete[] scales;
    delete[] opacities;
    delete[] rotations;

    cudaFree(devLocations);
    cudaFree(devShs);
    cudaFree(devScales);
    cudaFree(devOpacities);
    cudaFree(devRotations);
}

void ModelSplats::pushBack(glm::vec3 location, std::vector<float> sh, glm::vec3 scale, float opacity, glm::quat rotation) {
    memcpy(&locations[count * 3], &location, 3 * sizeof(float));
    for (int i  = 0; i < shCoeffs * 3; i++) {
        shs[count * 3 * shCoeffs + i] = sh.at(i);
    }
    memcpy(&scales[count * 3], &scale, 3 * sizeof(float));
    opacities[count] = opacity;
    memcpy(&rotations[count * 4], &rotation, 4 * sizeof(float));

    count++;
}

void ModelSplats::copy(int indexTo, int indexFrom) {
    assert(indexTo >= 0 && indexTo < count && indexFrom > 0 && indexFrom < count);

    memcpy(&locations[indexTo * 3], &locations[indexFrom * 3], 3 * sizeof(float));
    for (int i = 0; i < shCoeffs * 3; i++) {
        shs[indexTo * 3 * shCoeffs + i] = shs[indexFrom * 3 * shCoeffs + i];
    }
    memcpy(&scales[indexTo * 3], &scales[indexFrom * 3], 3 * sizeof(float));
    opacities[indexTo] = opacities[indexFrom];
    memcpy(&rotations[indexTo * 4], &rotations[indexFrom * 4], 4 * sizeof(float));
}

void ModelSplats::deviceBuffer() const {
    cudaMemcpy(devLocations, locations, capacity * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devShs, shs, capacity * 3 * shCoeffs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devScales, scales, capacity * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devOpacities, opacities, capacity * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devRotations, rotations, capacity * 4 * sizeof(float), cudaMemcpyHostToDevice);
}
