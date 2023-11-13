#include "ModelSplatsHost.h"
#include "ModelSplatsDevice.h"
#include "ui/UiPanelInput.h"

ModelSplatsHost::ModelSplatsHost(int capacity, int shDegree, int shCoeffs) :
        capacity(capacity), shDegree(shDegree), shCoeffs(shCoeffs) {

    locations = new float[capacity * 3];
    shs = new float[capacity * 3 * shCoeffs];
    scales = new float[capacity * 3];
    opacities = new float[capacity];
    rotations = new float[capacity * 4];
}

ModelSplatsHost::ModelSplatsHost(const ModelSplatsDevice& device) :
        ModelSplatsHost(device.capacity, device.shDegree, device.shCoeffs) {

    count = device.count;

    cudaMemcpy(locations, device.devLocations, count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(shs, device.devShs, count * 3 * shCoeffs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(scales, device.devScales, count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(opacities, device.devOpacities, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rotations, device.devRotations, count * 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

ModelSplatsHost::~ModelSplatsHost() {
    delete[] locations;
    delete[] shs;
    delete[] scales;
    delete[] opacities;
    delete[] rotations;
}

void ModelSplatsHost::pushBack(glm::vec3 location, std::vector<float> sh, glm::vec3 scale, float opacity, glm::quat rotation) {
    assert(count < capacity);

    memcpy(&locations[count * 3], &location, 3 * sizeof(float));
    for (int i  = 0; i < shCoeffs * 3; i++) {
        shs[count * 3 * shCoeffs + i] = sh.at(i);
    }
    memcpy(&scales[count * 3], &scale, 3 * sizeof(float));
    opacities[count] = opacity;
    memcpy(&rotations[count * 4], &rotation, 4 * sizeof(float));

    count++;
}

void ModelSplatsHost::copy(int indexTo, int indexFrom) {
    assert(indexTo >= 0 && indexTo < count && indexFrom > 0 && indexFrom < count);

    memcpy(&locations[indexTo * 3], &locations[indexFrom * 3], 3 * sizeof(float));
    for (int i = 0; i < shCoeffs * 3; i++) {
        shs[indexTo * 3 * shCoeffs + i] = shs[indexFrom * 3 * shCoeffs + i];
    }
    memcpy(&scales[indexTo * 3], &scales[indexFrom * 3], 3 * sizeof(float));
    opacities[indexTo] = opacities[indexFrom];
    memcpy(&rotations[indexTo * 4], &rotations[indexFrom * 4], 4 * sizeof(float));
}
