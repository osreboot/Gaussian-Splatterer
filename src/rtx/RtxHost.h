#pragma once

#include <owl/owl.h>
#include <owl/common.h>

#include "TruthCameras.h"

extern "C" char RtxDevice_ptx[];

class RtxHost {

private:
    OWLRayGen rayGen;
    OWLContext context;

    OWLGeomType geomType;

    const owl::vec2i size;

    bool initialized;
    float timer;

public:
    RtxHost(const owl::vec2i size);

    void load(const std::string& pathModel, const std::string& pathTexture);

    void render(float delta, uint64_t frameBuffer, TruthCameras& cameras);

};
