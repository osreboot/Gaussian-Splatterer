#include "Camera.h"

#include <diff-gaussian-rasterization/third_party/glm/glm/glm.hpp>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/quaternion.hpp>

#include "Config.h"
#include "Project.h"

std::vector<glm::vec3> getFibonacciSphere(int count, float distance) {
    std::vector<glm::vec3> out;

    // Algorithm source for Fibonacci sphere placement: https://youtu.be/lctXaT9pxA0?si=xJQE1KXCH92s5tne&t=66
    static const float goldenRatio = (1.0f + sqrtf(5.0f)) / 2.0f;
    static const float angleStep = 2.0f * (float)M_PI * goldenRatio;

    for(int i = 0; i < count; i++) {
        float t = (float)i / (float)count;
        float angle1 = acosf(1.0f - 2.0f * t);
        float angle2 = angleStep * (float)i;

        out.emplace_back(sinf(angle1) * cosf(angle2) * distance,
                         sinf(angle1) * sinf(angle2) * distance,
                         cosf(angle1) * distance);
    }

    return out;
}

int Camera::getCamerasCount(const Project& project) {
    return project.sphere1.count + project.sphere2.count;
}

std::vector<Camera> Camera::getCameras(const Project& project) {
    std::vector<Camera> out;
    out.reserve(getCamerasCount(project));

    static const glm::vec3 target = {0.0f, 0.0f, 0.0f};

    glm::mat4 rot1 = (glm::mat4)glm::angleAxis(glm::radians(project.sphere1.rotY), glm::vec3(0.0f, 1.0f, 0.0f)) *
                     (glm::mat4)glm::angleAxis(glm::radians(project.sphere1.rotX), glm::vec3(1.0f, 0.0f, 0.0f));

    for(glm::vec3 loc3 : getFibonacciSphere(project.sphere1.count, project.sphere1.distance)) {
        glm::vec4 loc4 = rot1 * glm::vec4(loc3.x, loc3.y, loc3.z, 1.0f);
        out.emplace_back(glm::vec3(loc4.x / loc4.w, loc4.y / loc4.w, loc4.z / loc4.w), target, project.sphere1.fovDeg);
    }

    glm::mat4 rot2 = (glm::mat4)glm::angleAxis(glm::radians(project.sphere2.rotY), glm::vec3(0.0f, 1.0f, 0.0f)) *
                     (glm::mat4)glm::angleAxis(glm::radians(project.sphere2.rotX), glm::vec3(1.0f, 0.0f, 0.0f));

    for(glm::vec3 loc3 : getFibonacciSphere(project.sphere2.count, project.sphere2.distance)) {
        glm::vec4 loc4 = rot2 * glm::vec4(loc3.x, loc3.y, loc3.z, 1.0f);
        out.emplace_back(glm::vec3(loc4.x / loc4.w, loc4.y / loc4.w, loc4.z / loc4.w), target, project.sphere2.fovDeg);
    }

    return out;
}

Camera Camera::getPreviewCamera(const Project& project) {
    static const glm::vec3 target = {0.0f, 0.0f, 0.0f};

    if(project.previewIndex == -1) {
        return {glm::vec3(cos(project.previewTimer / 2.0f), 0.4f, sin(project.previewTimer / 2.0f)) * glm::vec3(10.0f),
                target, PREVIEW_FOV_Y};
    } else return getCameras(project).at(project.previewIndex);
}

Camera::Camera(const glm::vec3& location, const glm::vec3& target, float fovDegY) :
    location(location), target(target), fovDegY(fovDegY) {}

glm::mat4 Camera::getView() const {
    static const glm::vec3 up(0.0f, 1.0f, 0.0f);
    return -glm::lookAt(location, target, up);
}

glm::mat4 Camera::getProjection() const {
    return glm::perspective(glm::radians(fovDegY), (float)RENDER_RESOLUTION_X / (float)RENDER_RESOLUTION_Y, 0.1f, 100.0f);
}
