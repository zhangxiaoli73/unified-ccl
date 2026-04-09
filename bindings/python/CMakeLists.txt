# Python bindings CMakeLists.txt
pybind11_add_module(_uccl_bindings uccl_bindings.cc)

target_include_directories(_uccl_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)

target_link_libraries(_uccl_bindings PRIVATE uccl)

# Install Python module
install(TARGETS _uccl_bindings
    LIBRARY DESTINATION lib/python/uccl
)

install(FILES
    uccl/__init__.py
    uccl/uccl.py
    DESTINATION lib/python/uccl
)
