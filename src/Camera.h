#pragma once

#include <vector>

#include <diff-gaussian-rasterization/third_party/glm/glm/glm.hpp>

class Project;

class Camera {

public:
    static int getCamerasCount(const Project& project);
    static std::vector<Camera> getCameras(const Project& project);
    static Camera getPreviewCamera(const Project& project);

    glm::vec3 location;
    glm::vec3 target;

    float fovDegY;

    Camera(const glm::vec3& location, const glm::vec3& target, float fovDegY);

    glm::mat4 getView() const;
    glm::mat4 getProjection() const;

};
