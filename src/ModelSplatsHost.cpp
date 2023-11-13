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

    cudaMemcpy(locations, device.devLocations, capacity * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(shs, device.devShs, capacity * 3 * shCoeffs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(scales, device.devScales, capacity * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(opacities, device.devOpacities, capacity * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rotations, device.devRotations, capacity * 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

ModelSplatsHost::~ModelSplatsHost() {
    delete[] locations;
    delete[] shs;
    delete[] scales;
    delete[] opacities;
    delete[] rotations;
}

void ModelSplatsHost::pushBack(glm::vec3 location, std::vector<float> sh, glm::vec3 scale, float opacity, glm::quat rotation) {
    memcpy(&locations[count * 3], &location, 3 * sizeof(float));
    for (int i  = 0; i < shCoeffs * 3; i++) {
        shs[count * 3 * shCoeffs + i] = sh.at(i);
    }
    memcpy(&scales[count * 3], &scale, 3 * sizeof(float));
    opacities[count] = opacity;
    memcpy(&rotations[count * 4], &rotation, 4 * sizeof(float));

    count++;
}
