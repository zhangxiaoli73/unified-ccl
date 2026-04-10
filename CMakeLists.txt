# Unified-CCL: Intel GPU Collective Communication Library
cmake_minimum_required(VERSION 3.20)
project(unified-ccl LANGUAGES CXX)

# Require Intel DPC++ compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler should be set via toolchain file or -DCMAKE_CXX_COMPILER=icpx
# Do not hardcode CMAKE_CXX_COMPILER inside CMakeLists.txt
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(WARNING "Recommended compiler is icpx (Intel DPC++). "
                    "Set -DCMAKE_CXX_COMPILER=icpx for SYCL support.")
endif()

# SYCL flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

# Options
option(UCCL_BUILD_TESTS "Build tests" ON)
option(UCCL_BUILD_PYTHON "Build Python bindings" ON)
option(UCCL_BUILD_UCX_PLUGIN "Build external UCX plugin" ON)

# Third-party dependencies
add_subdirectory(third_party)

# Dependencies
find_package(MPI REQUIRED)

# pybind11 (optional)
if(UCCL_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(WARNING "pybind11 not found, Python bindings will not be built")
        set(UCCL_BUILD_PYTHON OFF)
    endif()
endif()

# Level Zero
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(L0 QUIET level-zero)
endif()

# Core library
add_subdirectory(src)

# Python bindings
if(UCCL_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

# External UCX plugin
if(UCCL_BUILD_UCX_PLUGIN)
    add_subdirectory(plugins/ucx)
endif()

# Tests
if(UCCL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
