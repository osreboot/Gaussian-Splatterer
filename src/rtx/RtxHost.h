#pragma once

#include <owl/owl.h>
#include <owl/common.h>

#include "Project.h"

class Camera;

extern "C" char RtxDevice_ptx[];

class RtxHost {

private:
    OWLRayGen rayGen;
    OWLContext context;

    OWLGeomType geomType;

    const owl::vec2i size;

    bool initialized;

public:
    explicit RtxHost(owl::vec2i size);

    void load(const Project& project);

    void render(uint32_t* frameBuffer, const Camera& camera, owl::vec3f background, const std::vector<Camera>& cameras);

};
