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

    bool initialized;

public:
    explicit RtxHost();

    void load(const Project& project);

    void render(uint32_t* frameBuffer, owl::vec2i size, const Camera& camera, owl::vec3f background, const std::vector<Camera>& cameras);

};
