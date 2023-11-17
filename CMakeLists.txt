cmake_minimum_required(VERSION 3.25)
project(GaussianSplatterer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 14)

add_definitions(-DNDEBUG)

add_executable(${PROJECT_NAME} WIN32 ${SOURCES})

include_directories(lib src)

add_subdirectory(src)

set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED OPTIONAL_COMPONENTS OpenGL)
if(TARGET OpenGL::OpenGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::OpenGL)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL)
endif()

add_subdirectory(lib/owl EXCLUDE_FROM_ALL)
embed_ptx(
        OUTPUT_TARGET ${PROJECT_NAME}-ptx
        PTX_LINK_LIBRARIES owl::owl
        SOURCES "src/rtx/RtxDevice.cuh" "src/rtx/RtxDevice.cu"
)

set(wxBUILD_SHARED OFF)
set(wxUSE_REGEX OFF)
add_subdirectory(lib/wxWidgets)

add_subdirectory(lib/diff-gaussian-rasterization)

target_link_libraries(${PROJECT_NAME} PRIVATE owl::owl ${PROJECT_NAME}-ptx wx::net wx::core wx::base wx::gl wx::html CudaRasterizer)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)