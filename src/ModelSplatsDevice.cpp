#include "ModelSplatsDevice.h"
#include "ModelSplatsHost.h"
#include "ui/UiPanelInput.h"

ModelSplatsDevice::ModelSplatsDevice(const ModelSplatsDevice& device) :
        capacity(device.capacity), shDegree(device.shDegree), shCoeffs(device.shCoeffs) {

    count = device.count;

    cudaMalloc(&devLocations, capacity * 3 * sizeof(float));
    cudaMalloc(&devShs, capacity * 3 * shCoeffs * sizeof(float));
    cudaMalloc(&devScales, capacity * 3 * sizeof(float));
    cudaMalloc(&devOpacities, capacity * sizeof(float));
    cudaMalloc(&devRotations, capacity * 4 * sizeof(float));

    cudaMemcpy(devLocations, device.devLocations, capacity * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devShs, device.devShs, capacity * 3 * shCoeffs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devScales, device.devScales, capacity * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devOpacities, device.devOpacities, capacity * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devRotations, device.devRotations, capacity * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
}

ModelSplatsDevice::ModelSplatsDevice(const ModelSplatsHost& host) :
        capacity(host.capacity), shDegree(host.shDegree), shCoeffs(host.shCoeffs) {

    count = host.count;

    cudaMalloc(&devLocations, capacity * 3 * sizeof(float));
    cudaMalloc(&devShs, capacity * 3 * shCoeffs * sizeof(float));
    cudaMalloc(&devScales, capacity * 3 * sizeof(float));
    cudaMalloc(&devOpacities, capacity * sizeof(float));
    cudaMalloc(&devRotations, capacity * 4 * sizeof(float));

    cudaMemcpy(devLocations, host.locations, capacity * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devShs, host.shs, capacity * 3 * shCoeffs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devScales, host.scales, capacity * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devOpacities, host.opacities, capacity * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devRotations, host.rotations, capacity * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

ModelSplatsDevice::~ModelSplatsDevice() {
    cudaFree(devLocations);
    cudaFree(devShs);
    cudaFree(devScales);
    cudaFree(devOpacities);
    cudaFree(devRotations);
}
