#pragma once

#include <owl/owl.h>
#include <owl/common.h>

#include "Project.h"
#include "RtxDevice.cuh"

class Camera;

extern "C" char RtxDevice_ptx[];

class RtxHost {

private:
    OWLRayGen rayGen;
    OWLContext context;

    OWLGeomType geomType;

    std::vector<owl::vec3f> bufferedVertices;
    std::vector<owl::vec3i> bufferedTriangles;
    std::vector<owl::vec2f> bufferedTextureCoords;

    bool modelPresent = false;
    std::array<OWLTexture, TEXTURE_COUNT> bufferedTextures;

    void bufferGeometry();

public:
    explicit RtxHost();

    void reset();
    void loadModel(const std::string& pathModel, const std::function<void()>& progressCallback);
    void loadTextureDiffuse(const std::string& pathTexture);

    void render(uint32_t* frameBuffer, owl::vec2i size, const Camera& camera, owl::vec3f background, int samples, const std::vector<Camera>& cameras);

};
