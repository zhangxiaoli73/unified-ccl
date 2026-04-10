# Third-party dependencies for Unified-CCL

# ---- UCX (Unified Communication X) ----
# Build UCX from source via FetchContent.
# UCX provides RDMA transports (InfiniBand, RoCE, etc.) and GPU Direct support.

include(FetchContent)

option(UCCL_UCX_TAG "UCX version tag to fetch" "v1.17.0")

FetchContent_Declare(
    ucx
    GIT_REPOSITORY https://github.com/openucx/ucx.git
    GIT_TAG        ${UCCL_UCX_TAG}
    GIT_SHALLOW    TRUE
)

FetchContent_GetProperties(ucx)
if(NOT ucx_POPULATED)
    FetchContent_Populate(ucx)
endif()

# UCX uses autotools, so we drive the build via ExternalProject-style custom commands.
include(ExternalProject)

set(UCX_SOURCE_DIR ${ucx_SOURCE_DIR})
set(UCX_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/ucx/install)

ExternalProject_Add(ucx_build
    SOURCE_DIR      ${UCX_SOURCE_DIR}
    PREFIX          ${CMAKE_BINARY_DIR}/third_party/ucx
    INSTALL_DIR     ${UCX_INSTALL_DIR}

    CONFIGURE_COMMAND
        cd ${UCX_SOURCE_DIR} && ./autogen.sh &&
        ${UCX_SOURCE_DIR}/configure
            --prefix=${UCX_INSTALL_DIR}
            --enable-shared
            --disable-static
            --without-java
            --without-go

    BUILD_COMMAND     make -j$ENV{NPROC}
    INSTALL_COMMAND   make install
    BUILD_IN_SOURCE   FALSE
    BUILD_BYPRODUCTS
        ${UCX_INSTALL_DIR}/lib/libucp.so
        ${UCX_INSTALL_DIR}/lib/libuct.so
        ${UCX_INSTALL_DIR}/lib/libucs.so
)

# Create imported targets for UCX libraries
add_library(ucx::ucp SHARED IMPORTED GLOBAL)
set_target_properties(ucx::ucp PROPERTIES
    IMPORTED_LOCATION ${UCX_INSTALL_DIR}/lib/libucp.so
    INTERFACE_INCLUDE_DIRECTORIES ${UCX_INSTALL_DIR}/include
)
add_dependencies(ucx::ucp ucx_build)

add_library(ucx::uct SHARED IMPORTED GLOBAL)
set_target_properties(ucx::uct PROPERTIES
    IMPORTED_LOCATION ${UCX_INSTALL_DIR}/lib/libuct.so
    INTERFACE_INCLUDE_DIRECTORIES ${UCX_INSTALL_DIR}/include
)
add_dependencies(ucx::uct ucx_build)

add_library(ucx::ucs SHARED IMPORTED GLOBAL)
set_target_properties(ucx::ucs PROPERTIES
    IMPORTED_LOCATION ${UCX_INSTALL_DIR}/lib/libucs.so
    INTERFACE_INCLUDE_DIRECTORIES ${UCX_INSTALL_DIR}/include
)
add_dependencies(ucx::ucs ucx_build)
