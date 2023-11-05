#pragma once

#include "owl/owl.h"
#include "owl/common.h"

extern "C" char RtxDevice_ptx[];

class RtxHost {

private:
    OWLRayGen rayGen;
    OWLContext context;

    OWLGeomType geomType;

    std::vector<owl::vec3f> splatCameras;

    bool initialized;
    float timer;

public:
    RtxHost();

    void setSplatModel(const std::string& pathModel, const std::string& pathTexture);
    void setSplatCameras();

    void update(float delta, int width, int height, uint64_t frameBuffer);

};
