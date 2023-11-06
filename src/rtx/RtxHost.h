#pragma once

#include <owl/owl.h>
#include <owl/common.h>

extern "C" char RtxDevice_ptx[];

class RtxHost {

private:
    OWLRayGen rayGen;
    OWLContext context;

    OWLGeomType geomType;

    const owl::vec2i size;

    std::vector<owl::vec3f> splatCameras;

    bool initialized;
    float timer;

public:
    RtxHost(const owl::vec2i size);

    void setSplatModel(const std::string& pathModel, const std::string& pathTexture);
    void setSplatCameras(int count, float distance);

    void update(float delta, uint64_t frameBuffer, float cameraDistance);

};
