cmake_minimum_required(VERSION 3.22)
project(SeatVisionDataset VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.8 REQUIRED COMPONENTS core dnn imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found")
endif()

add_library(seatvision_dataset_core STATIC
    src/config.cpp
    src/detector.cpp
    src/evaluation.cpp
    src/batch.cpp
)
target_include_directories(seatvision_dataset_core
    PUBLIC include
    PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(seatvision_dataset_core
    PUBLIC ${OpenCV_LIBS} ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} CUDA::cudart
)
target_compile_options(seatvision_dataset_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(seatvision_dataset src/main.cpp)
set_target_properties(seatvision_dataset PROPERTIES OUTPUT_NAME "seatvision-dataset")
target_link_libraries(seatvision_dataset PRIVATE seatvision_dataset_core)
target_compile_options(seatvision_dataset PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS seatvision_dataset RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_executable(seatvision_dataset_evaluation_test
        tests/evaluation_test.cpp
        src/evaluation.cpp
    )
    target_include_directories(seatvision_dataset_evaluation_test PRIVATE include)
    target_link_libraries(seatvision_dataset_evaluation_test PRIVATE ${OpenCV_LIBS})
    target_compile_options(seatvision_dataset_evaluation_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME dataset_evaluation COMMAND seatvision_dataset_evaluation_test)
endif()
