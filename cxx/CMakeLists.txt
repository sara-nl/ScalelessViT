cmake_minimum_required(VERSION 3.19)
project(cxx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)


set(default_build_type "Release")
if (EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "Debug")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# CUDA flags
set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PythonLibs REQUIRED)


include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -fopenmp")

add_executable(crop_interpolate main.cpp)
set_target_properties(crop_interpolate PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#
# Copmile with CUDA
#
add_library(ci_kernels SHARED kernels.cu kernels.cuh)
set_target_properties(
        ci_kernels
        PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

target_compile_options(
        ci_kernels
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler
        -fPIC
        -shared
        -dc
        >)
target_link_libraries(ci_kernels CUDA::cudart_static)

target_link_libraries(crop_interpolate ci_kernels ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
set_target_properties(crop_interpolate PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
set_target_properties(crop_interpolate PROPERTIES INSTALL_RPATH "/home/duncank/local")

install(TARGETS crop_interpolate
        CONFIGURATIONS Release
        RUNTIME DESTINATION bin)
