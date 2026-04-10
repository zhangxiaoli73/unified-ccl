# External UCX Network Plugin
# Compiles as a standalone libuccl-net-ucx.so

add_library(uccl-net-ucx SHARED plugin.cc)

target_include_directories(uccl-net-ucx PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(uccl-net-ucx PRIVATE
    ucx::ucp ucx::uct ucx::ucs
)

install(TARGETS uccl-net-ucx
    LIBRARY DESTINATION lib
)
