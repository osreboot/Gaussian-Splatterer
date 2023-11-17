#include "RtxHost.h"

#include <fstream>
#include <vector>

#include <stb/stb_image.h>
#include <diff-gaussian-rasterization/third_party/glm/glm/glm.hpp>

#include "RtxDevice.cuh"
#include "Camera.h"

using namespace owl;
using namespace std;

RtxHost::RtxHost(vec2i size) : size(size) {
    // Initialize OWL context so we can start loading resources
    context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(context, RtxDevice_ptx);

    // Initialize OWL data structures and parameters with our world geometry and materials
    OWLVarDecl trianglesGeomVars[] = {
            {"vertices", OWL_BUFPTR, OWL_OFFSETOF(WorldGeometry, vertices)},
            {"triangles", OWL_BUFPTR, OWL_OFFSETOF(WorldGeometry, triangles)},
            {"textureCoords", OWL_BUFPTR, OWL_OFFSETOF(WorldGeometry, textureCoords)},
            {"textures", OWL_BUFPTR, OWL_OFFSETOF(WorldGeometry, textures)},
            {nullptr}
    };
    geomType = owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(WorldGeometry), trianglesGeomVars, -1);
    owlGeomTypeSetClosestHit(geomType, 0, module, "TriangleMesh");

    owlMissProgCreate(context, module, "miss", 0, nullptr, 0);

    OWLVarDecl rayGenVars[] = {
            {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenerator, frameBuffer)},
            {"size", OWL_INT2, OWL_OFFSETOF(RayGenerator, size)},
            {"background", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, background)},
            {"worldHandle", OWL_GROUP, OWL_OFFSETOF(RayGenerator, worldHandle)},
            {"cameraLocation", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, cameraLocation)},
            {"cameraMatrix", OWL_BUFPTR, OWL_OFFSETOF(RayGenerator, cameraMatrix)},
            {"splatCamerasCount", OWL_INT, OWL_OFFSETOF(RayGenerator, splatCamerasCount)},
            {"splatCameras", OWL_BUFPTR, OWL_OFFSETOF(RayGenerator, splatCameras)},
            {}
    };
    rayGen = owlRayGenCreate(context, module, "rayGenProgram", sizeof(RayGenerator), rayGenVars, -1);

    // Build everything
    owlBuildPrograms(context);
    owlBuildPipeline(context);

    initialized = false;
}

static OWLTexture loadTexture(OWLContext context, const string& path) {
    int width, height;
    unsigned char* image = stbi_load(path.c_str(), &width, &height, nullptr, STBI_rgb_alpha);
    OWLTexture texture = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, width, height, image,
                                            OWL_TEXTURE_NEAREST, OWL_TEXTURE_WRAP);
    stbi_image_free(image);
    return texture;
}

void RtxHost::load(const string& pathModel, const string& pathTexture) {
    vector<vec3f> vertices;
    vector<vec3i> triangles;
    vector<vec2f> textureCoordsRefVec;
    vector<int> textureCoordsVec;

    ifstream ifs(pathModel);
    string line;
    while(getline(ifs, line)) {
        istringstream iss(line);
        string prefix;
        iss >> prefix;

        if(prefix == "v") { // Vertex definition
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }else if(prefix == "vt") { // Vertex definition
            float u, v;
            iss >> u >> v;
            textureCoordsRefVec.emplace_back(u, v);
        }else if(prefix == "f") { // Face definition
            // Space-separated vertices list, with the syntax: vertexIndex/vertexTextureIndex/vertexNormalIndex
            vector<vec3i> faceVertices;

            string sFaceVertex;
            while(iss >> sFaceVertex) {
                vec3i faceVertex = {0, 0, 0};
                int charIndex = 0;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.x = faceVertex.x * 10 + (sFaceVertex[charIndex++] - '0');
                }
                charIndex++;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.y = faceVertex.y * 10 + (sFaceVertex[charIndex++] - '0');
                }
                charIndex++;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.z = faceVertex.z * 10 + (sFaceVertex[charIndex++] - '0');
                }
                faceVertices.push_back(faceVertex);
            }
            if(faceVertices.size() == 4) {
                triangles.push_back(vec3i(faceVertices[0].x, faceVertices[1].x, faceVertices[2].x) - vec3i(1));
                triangles.push_back(vec3i(faceVertices[0].x, faceVertices[2].x, faceVertices[3].x) - vec3i(1));
                textureCoordsVec.emplace_back(faceVertices[0].y);
                textureCoordsVec.emplace_back(faceVertices[1].y);
                textureCoordsVec.emplace_back(faceVertices[2].y);
                textureCoordsVec.emplace_back(faceVertices[0].y);
                textureCoordsVec.emplace_back(faceVertices[2].y);
                textureCoordsVec.emplace_back(faceVertices[3].y);
            }else if(faceVertices.size() == 3) {
                triangles.push_back(vec3i(faceVertices[0].x, faceVertices[1].x, faceVertices[2].x) - vec3i(1));
                textureCoordsVec.emplace_back(faceVertices[0].y);
                textureCoordsVec.emplace_back(faceVertices[1].y);
                textureCoordsVec.emplace_back(faceVertices[2].y);
            }else throw runtime_error("Unexpected vertex count in face list!" + to_string(faceVertices.size()));
        }
    }

    int numTriangles = (int)triangles.size();
    vector<vec2f> textureCoords;
    for(int i = 0; i < numTriangles; i++){
        if(textureCoordsVec[i * 3] > 0 && textureCoordsVec[i * 3 + 1] > 0 && textureCoordsVec[i * 3 + 2] > 0) {
            textureCoords.push_back(textureCoordsRefVec[textureCoordsVec[i * 3] - 1]);
            textureCoords.push_back(textureCoordsRefVec[textureCoordsVec[i * 3 + 1] - 1]);
            textureCoords.push_back(textureCoordsRefVec[textureCoordsVec[i * 3 + 2] - 1]);
        }else{
            textureCoords.emplace_back(0.0f, 0.0f);
            textureCoords.emplace_back(0.0f, 0.0f);
            textureCoords.emplace_back(0.0f, 0.0f);
        }
    }

    vector<OWLTexture> textures;
    textures.push_back(loadTexture(context, pathTexture));

    OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer triangleBuffer = owlDeviceBufferCreate(context, OWL_INT3, triangles.size(), triangles.data());
    OWLBuffer textureCoordBuffer = owlDeviceBufferCreate(context, OWL_FLOAT2, textureCoords.size(), textureCoords.data());
    OWLBuffer textureBuffer = owlDeviceBufferCreate(context, OWL_TEXTURE, textures.size(), textures.data());

    OWLGeom worldGeometry = owlGeomCreate(context, geomType);
    owlTrianglesSetVertices(worldGeometry, vertexBuffer, vertices.size(), sizeof(vertices[0]), 0);
    owlTrianglesSetIndices(worldGeometry, triangleBuffer, triangles.size(), sizeof(triangles[0]), 0);

    owlGeomSetBuffer(worldGeometry, "vertices", vertexBuffer);
    owlGeomSetBuffer(worldGeometry, "triangles", triangleBuffer);
    owlGeomSetBuffer(worldGeometry, "textureCoords", textureCoordBuffer);
    owlGeomSetBuffer(worldGeometry, "textures", textureBuffer);

    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &worldGeometry);
    owlGroupBuildAccel(trianglesGroup);
    OWLGroup worldGroup = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(worldGroup);

    owlRayGenSetGroup(rayGen, "worldHandle", worldGroup);

    initialized = true;
}

void RtxHost::render(uint32_t* frameBuffer, const Camera& camera, vec3f background, const vector<Camera>& cameras) {
    owlRayGenSet1i(rayGen, "splatCamerasCount", (int)cameras.size());

    if (!cameras.empty()) {
        vector<glm::vec3> cameraLocations = cameras
                | ranges::views::transform([](Camera camera){ return camera.location; })
                | ranges::to<vector>();

        OWLBuffer splatCamerasBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, cameraLocations.size(), cameraLocations.data());
        owlRayGenSetBuffer(rayGen, "splatCameras", splatCamerasBuffer);
    }

    if(initialized) {
        // Send camera parameters to the ray tracer
        owlRayGenSet1ul(rayGen, "frameBuffer", (uint64_t)frameBuffer);
        owlRayGenSet2i(rayGen, "size", size.x, size.y);
        owlRayGenSet3f(rayGen, "background", (const owl3f&)background);
        owlRayGenSet3f(rayGen, "cameraLocation", (const owl3f&)camera.location);
        glm::mat4 cameraMatrix = glm::inverse(camera.getProjection() * camera.getView());
        owlRayGenSetBuffer(rayGen, "cameraMatrix", owlDeviceBufferCreate(context, OWL_FLOAT, 16, &cameraMatrix[0]));

        // Run ray tracer
        owlBuildSBT(context);
        owlRayGenLaunch2D(rayGen, size.x, size.y);
    }
}
