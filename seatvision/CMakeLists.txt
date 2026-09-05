cmake_minimum_required(VERSION 3.22)
project(SeatVision VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.8 REQUIRED COMPONENTS core dnn highgui imgproc videoio)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found")
endif()

option(SEATVISION_BUILD_LEGACY_DEMO "Build the retired hard-coded MVP demo" OFF)
if(SEATVISION_BUILD_LEGACY_DEMO)
    add_executable(seatvision
        src/main.cpp
        src/argus_camera.cpp
        src/async_detector.cpp
        src/seat_state_engine.cpp
    )
    target_include_directories(seatvision PRIVATE include)
    target_link_libraries(seatvision PRIVATE ${OpenCV_LIBS} PkgConfig::GSTREAMER)
    target_compile_options(seatvision PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(seatvisiond
    src/product/main.cpp
    src/product/config.cpp
    src/product/semantics.cpp
    src/product/camera_stream.cpp
    src/product/opencv_yolo_detector.cpp
    src/product/tensorrt_yolo_detector.cpp
    src/product/detector_factory.cpp
    src/product/tracker.cpp
    src/product/scene_reasoner.cpp
    src/product/renderer.cpp
    src/product/event_sink.cpp
    src/product/runtime.cpp
    src/argus_camera.cpp
)
target_include_directories(seatvisiond PRIVATE include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(seatvisiond PRIVATE ${OpenCV_LIBS} PkgConfig::GSTREAMER ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} CUDA::cudart)
target_compile_options(seatvisiond PRIVATE -Wall -Wextra -Wpedantic)

# A deliberately small diagnostic executable for validating Argus-to-appsink
# capture independently from OpenCV, TensorRT, tracking, and rendering.
add_executable(seatvision_capture_probe tools/capture_probe.cpp)
target_link_libraries(seatvision_capture_probe PRIVATE PkgConfig::GSTREAMER)
target_compile_options(seatvision_capture_probe PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS seatvisiond RUNTIME DESTINATION bin)
if(SEATVISION_BUILD_LEGACY_DEMO)
    install(TARGETS seatvision RUNTIME DESTINATION bin)
endif()

if(BUILD_TESTING)
    add_executable(seatvision_scene_test
        tests/scene_reasoner_test.cpp
        src/product/semantics.cpp
        src/product/tracker.cpp
        src/product/scene_reasoner.cpp
    )
    target_include_directories(seatvision_scene_test PRIVATE include)
    target_link_libraries(seatvision_scene_test PRIVATE ${OpenCV_LIBS})
    target_compile_options(seatvision_scene_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME scene_reasoner COMMAND seatvision_scene_test)

endif()
