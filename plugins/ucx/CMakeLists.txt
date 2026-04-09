# External UCX Network Plugin
# Compiles as a standalone libuccl-net-ucx.so

add_library(uccl-net-ucx SHARED plugin.cc)

target_include_directories(uccl-net-ucx PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${UCX_INCLUDE_DIRS}
)

target_link_libraries(uccl-net-ucx PRIVATE
    ${UCX_LIBRARIES}
)

install(TARGETS uccl-net-ucx
    LIBRARY DESTINATION lib
)
