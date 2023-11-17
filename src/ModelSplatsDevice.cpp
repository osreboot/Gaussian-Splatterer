#include "ModelSplatsDevice.h"

#include "ModelSplatsHost.h"
#include "ui/UiPanelViewInput.h"

ModelSplatsDevice::ModelSplatsDevice(const ModelSplatsDevice& device) :
        capacity(device.capacity), shDegree(device.shDegree), shCoeffs(device.shCoeffs) {

    count = device.count;

    cudaMalloc(&devLocations, count * 3 * sizeof(float));
    cudaMalloc(&devShs, count * 3 * shCoeffs * sizeof(float));
    cudaMalloc(&devScales, count * 3 * sizeof(float));
    cudaMalloc(&devOpacities, count * sizeof(float));
    cudaMalloc(&devRotations, count * 4 * sizeof(float));

    cudaMemcpy(devLocations, device.devLocations, count * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devShs, device.devShs, count * 3 * shCoeffs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devScales, device.devScales, count * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devOpacities, device.devOpacities, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(devRotations, device.devRotations, count * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
}

ModelSplatsDevice::ModelSplatsDevice(const ModelSplatsHost& host) :
        capacity(host.capacity), shDegree(host.shDegree), shCoeffs(host.shCoeffs) {

    count = host.count;

    cudaMalloc(&devLocations, count * 3 * sizeof(float));
    cudaMalloc(&devShs, count * 3 * shCoeffs * sizeof(float));
    cudaMalloc(&devScales, count * 3 * sizeof(float));
    cudaMalloc(&devOpacities, count * sizeof(float));
    cudaMalloc(&devRotations, count * 4 * sizeof(float));

    cudaMemcpy(devLocations, host.locations, count * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devShs, host.shs, count * 3 * shCoeffs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devScales, host.scales, count * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devOpacities, host.opacities, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devRotations, host.rotations, count * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

ModelSplatsDevice::~ModelSplatsDevice() {
    cudaFree(devLocations);
    cudaFree(devShs);
    cudaFree(devScales);
    cudaFree(devOpacities);
    cudaFree(devRotations);
}
