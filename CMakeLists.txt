cmake_minimum_required(VERSION 2.6)

project(mmdeploy_plugins)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /O2")
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

find_package(CUDA REQUIRED)

#if(WIN32)
#enable_language(CUDA)
#endif(WIN32)

# cuda
set(cuda_inc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")
set(cuda_lib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64")
include_directories(${cuda_inc})
link_directories(${cuda_lib})
#cub
set(CUB_ROOT_DIR "third_party/cub")
include_directories(${CUB_ROOT_DIR})
# tensorrt
set(tensorrt_inc "D:/TensorRT-8.4.3.1/include")
set(tensorrt_lib "D:/TensorRT-8.4.3.1/lib")
include_directories(${tensorrt_inc})
link_directories(${tensorrt_lib})
# opencv
include_directories("${PROJECT_SOURCE_DIR}/third_party/CV460_64/include")
set(opencv_lib "${PROJECT_SOURCE_DIR}/third_party/CV460_64/lib/opencv_world460.lib")

# common files
include_directories(common)

file(GLOB trtbatchednms_src ${PROJECT_SOURCE_DIR}/trtbatchednms/*.cpp ${PROJECT_SOURCE_DIR}/trtbatchednms/*.cu)
cuda_add_library(trtbatchednms SHARED ${trtbatchednms_src})
#cuda_add_library(trtbatchednms STATIC ${trtbatchednms_src})
target_link_libraries(trtbatchednms nvinfer cudart)


file(GLOB multilevelroialign_src ${PROJECT_SOURCE_DIR}/multi_level_roi_align/*.cpp ${PROJECT_SOURCE_DIR}/multi_level_roi_align/*.cu)
cuda_add_library(mmcvmultilevelroialign SHARED ${multilevelroialign_src})
#cuda_add_library(mmcvmultilevelroialign STATIC ${multilevelroialign_src})
target_link_libraries(mmcvmultilevelroialign nvinfer cudart)


file(GLOB maskrcnn_src ${PROJECT_SOURCE_DIR}/maskrcnn_test/*.cpp)
add_executable(maskrcnn ${maskrcnn_src})
target_link_libraries(maskrcnn nvinfer cudart nvinfer_plugin)
target_link_libraries(maskrcnn trtbatchednms)
target_link_libraries(maskrcnn mmcvmultilevelroialign)
target_link_libraries(maskrcnn ${opencv_lib})

if(UNIX)
    add_definitions(-O2 -pthread)
endif(UNIX)