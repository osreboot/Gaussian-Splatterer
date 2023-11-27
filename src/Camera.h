#pragma once

#include <vector>

#include <diff-gaussian-rasterization/third_party/glm/glm/glm.hpp>

class Project;

class Camera {

public:
    static int getCamerasCount(const Project& project); // Returns the number of cameras present in the scene
    static std::vector<Camera> getCameras(const Project& project); // Returns all cameras in the scene
    static Camera getPreviewCamera(const Project& project); // Returns the camera currently being used by the preview views

    glm::vec3 location; // Origin of the camera
    glm::vec3 target; // Point that the camera is looking at

    float fovDegY; // Vertical FOV, in degrees

    Camera(const glm::vec3& location, const glm::vec3& target, float fovDegY);

    glm::mat4 getView() const; // The camera's view transformation matrix
    glm::mat4 getProjection(float aspect) const; // The camera's projection transformation matrix

};
