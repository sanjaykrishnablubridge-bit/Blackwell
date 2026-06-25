cmake_minimum_required(VERSION 3.22)
project(BLACKWELL CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Toolkit
find_package(CUDAToolkit REQUIRED)

# Language standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# ---- GPU architecture selection -------------------------------------------
# Compute capability of the target GPU as a 3-digit number, e.g.
#   120 = sm_120 (Blackwell, RTX 5080),  103 = sm_103.
# Auto-detected from the installed GPU via nvidia-smi, but can be overridden
# from the command line:  cmake -DGPU_CC=103 ..
if(NOT DEFINED GPU_CC)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _GPU_CC_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    # nvidia-smi reports one line per GPU (e.g. "12.0"); take the first.
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _GPU_CC_FIRST "${_GPU_CC_RAW}")
    # "12.0" -> "120", "10.3" -> "103"
    string(REPLACE "." "" GPU_CC "${_GPU_CC_FIRST}")
endif()

if(NOT GPU_CC)
    message(FATAL_ERROR
        "Could not detect the GPU compute capability. "
        "Pass it explicitly, e.g.  cmake -DGPU_CC=120 ..")
endif()

message(STATUS "Target GPU compute capability: sm_${GPU_CC}")
if(GPU_CC EQUAL 103)
    # tcgen05 (Tensor Memory) is a family-specific feature: it only assembles for
    # the "a" virtual arch. NOTE: -arch=sm_103a is NOT enough — nvcc 13.0 lowers the
    # PTX .target back to plain sm_103 and ptxas then rejects tcgen05.{alloc,mma,ld}.
    # The virtual arch must carry the "a", so spell it out with -gencode.
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_103a,code=sm_103a")
else()
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_120a,code=sm_120a")
endif()

# Compiler flags
# -lineinfo  : maps profiler samples back to source lines (Nsight Compute)
# -O3        : full optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3 -lcudart")

# Auto-discover the .cu files to build. CONFIGURE_DEPENDS re-scans when files
# are added or removed.
#   * On an sm_103 GPU, build ONLY the kernels under src/SM103.
#   * On any other GPU, build everything except the sm_103-specific kernels
#     (which use features unavailable on other architectures).
if(GPU_CC EQUAL 103)
    file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS src/SM103/*.cu)
else()
    file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS src/*.cu)
    list(FILTER KERNEL_SOURCES EXCLUDE REGEX "/src/SM103/")
endif()

find_library(NVML_LIB nvidia-ml
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64/stubs)

foreach(SOURCE ${KERNEL_SOURCES})
    get_filename_component(NAME ${SOURCE} NAME_WE)
    add_executable(${NAME} ${SOURCE})
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${NAME} PRIVATE CUDA::cuda_driver)
    if(NAME STREQUAL "deviceSpecs" AND NVML_LIB)
        target_link_libraries(${NAME} PRIVATE ${NVML_LIB})
    endif()
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
