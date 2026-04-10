# Unified-CCL core library
set(UCCL_SOURCES
    init.cc
    collectives.cc
    enqueue.cc
    group.cc
    bootstrap.cc
    proxy.cc
    channel.cc
    transport.cc

    # Topology
    topo/topo_pci.cc
    topo/topo_upi.cc
    topo/topo_ring.cc

    # Algorithms
    algorithms/ring.cc
    algorithms/one_shot.cc
    algorithms/symmetric.cc
    algorithms/copy_engine.cc

    # Protocols
    protocols/simple.cc
    protocols/ll128.cc

    # Transport
    transport/p2p.cc
    transport/net.cc

    # Plugin
    plugin/net_ucx.cc

    # RMA (one-sided remote memory access)
    rma/rma.cc

    # Misc
    misc/debug.cc
    misc/utils.cc
)

add_library(uccl SHARED ${UCCL_SOURCES})

target_include_directories(uccl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# SYCL flags
target_compile_options(uccl PRIVATE -fsycl)
target_link_options(uccl PRIVATE -fsycl)

# MPI
target_link_libraries(uccl PUBLIC MPI::MPI_CXX)

# Level Zero (optional)
if(L0_FOUND)
    target_include_directories(uccl PRIVATE ${L0_INCLUDE_DIRS})
    target_link_libraries(uccl PRIVATE ${L0_LIBRARIES})
    target_compile_definitions(uccl PRIVATE UCCL_HAS_LEVEL_ZERO=1)
endif()

# UCX (built from third_party)
target_link_libraries(uccl PRIVATE ucx::ucp ucx::uct ucx::ucs)
target_compile_definitions(uccl PRIVATE UCCL_HAS_UCX=1)

# dl for dynamic plugin loading
if(UNIX)
    target_link_libraries(uccl PRIVATE dl)
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(uccl PRIVATE Threads::Threads)

# Install
install(TARGETS uccl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    include/uccl.h
    include/uccl_net.h
    include/uccl_common.h
    include/hw_resources.h
    DESTINATION include/uccl
)
