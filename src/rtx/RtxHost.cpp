#define CAMERA_COS_FOVY 0.8f // wide = 0.9 | original = 0.66 | narrow = 0.5
#define CAMERA_LOOK_UP vec3f{0.0f, 1.0f, 0.0f}

#include <fstream>
#include <vector>

#include <owl/owl.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "RtxHost.h"
#include "RtxDevice.cuh"

using namespace owl;
using namespace std;

RtxHost::RtxHost() {
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
            {"worldHandle", OWL_GROUP, OWL_OFFSETOF(RayGenerator, worldHandle)},
            {"camera.location", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, camera.location)},
            {"camera.originPixel", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, camera.originPixel)},
            {"camera.dirRight", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, camera.dirRight)},
            {"camera.dirUp", OWL_FLOAT3, OWL_OFFSETOF(RayGenerator, camera.dirUp)},
            {"splatCamerasCount", OWL_INT, OWL_OFFSETOF(RayGenerator, splatCamerasCount)},
            {"splatCameras", OWL_BUFPTR, OWL_OFFSETOF(RayGenerator, splatCameras)},
            {}
    };
    rayGen = owlRayGenCreate(context, module, "rayGenProgram", sizeof(RayGenerator), rayGenVars, -1);

    // Build everything
    owlBuildPrograms(context);
    owlBuildPipeline(context);

    initialized = false;
    timer = 0.0f;

    setSplatCameras();
}

static OWLTexture loadTexture(OWLContext context, const string& path) {
    int width, height;
    unsigned char* image = stbi_load(path.c_str(), &width, &height, nullptr, STBI_rgb_alpha);
    OWLTexture texture = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, width, height, image,
                                            OWL_TEXTURE_NEAREST, OWL_TEXTURE_WRAP);
    stbi_image_free(image);
    return texture;
}

void RtxHost::setSplatModel(const string& pathModel, const string& pathTexture) {
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

void RtxHost::setSplatCameras() {
    splatCameras.clear();

    splatCameras.emplace_back(1.0f, 1.0f, 1.0f);
    splatCameras.emplace_back(1.0f, 1.0f, -1.0f);
    splatCameras.emplace_back(1.0f, -1.0f, 1.0f);
    splatCameras.emplace_back(1.0f, -1.0f, -1.0f);
    splatCameras.emplace_back(-1.0f, 1.0f, 1.0f);
    splatCameras.emplace_back(-1.0f, 1.0f, -1.0f);
    splatCameras.emplace_back(-1.0f, -1.0f, 1.0f);
    splatCameras.emplace_back(-1.0f, -1.0f, -1.0f);

    owlRayGenSet1i(rayGen, "splatCamerasCount", (int)splatCameras.size());

    OWLBuffer splatCamerasBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, splatCameras.size(), splatCameras.data());
    owlRayGenSetBuffer(rayGen, "splatCameras", splatCamerasBuffer);
}

void RtxHost::update(float delta, int width, int height, uint64_t frameBuffer) {
    static const vec3f cameraTarget = {0.0f, 0.0f, 0.0f};

    timer += delta;

    const vec3f cameraLocation = {cos(timer / 2.0f) * 8.0f, 3.0f, sin(timer / 2.0f) * 8.0f};

    if(initialized) {
        // Calculate camera parameters
        vec3f cameraDir = normalize(cameraTarget - cameraLocation);
        float aspect = (float)width / (float)height;
        vec3f cameraDirRight = CAMERA_COS_FOVY * aspect * normalize(cross(cameraDir, CAMERA_LOOK_UP));
        vec3f cameraDirUp = CAMERA_COS_FOVY * normalize(cross(cameraDirRight, cameraDir));
        vec3f cameraOriginPixel = cameraDir - 0.5f * cameraDirRight - 0.5f * cameraDirUp;

        // Send camera parameters to the ray tracer
        owlRayGenSet1ul(rayGen, "frameBuffer", frameBuffer);
        owlRayGenSet2i(rayGen, "size", width, height);
        owlRayGenSet3f(rayGen, "camera.location", (const owl3f&)cameraLocation);
        owlRayGenSet3f(rayGen, "camera.originPixel", (const owl3f&)cameraOriginPixel);
        owlRayGenSet3f(rayGen, "camera.dirRight", (const owl3f&)cameraDirRight);
        owlRayGenSet3f(rayGen, "camera.dirUp", (const owl3f&)cameraDirUp);

        // Run ray tracer
        owlBuildSBT(context);
        owlRayGenLaunch2D(rayGen, width, height);
    }
}
