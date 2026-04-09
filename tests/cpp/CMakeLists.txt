# C++ tests

# AllReduce correctness test
add_executable(test_allreduce test_allreduce.cc)
target_link_libraries(test_allreduce PRIVATE uccl MPI::MPI_CXX)
target_compile_options(test_allreduce PRIVATE -fsycl)
target_link_options(test_allreduce PRIVATE -fsycl)
add_test(NAME test_allreduce COMMAND test_allreduce)

# Transport layer test
add_executable(test_transport test_transport.cc)
target_link_libraries(test_transport PRIVATE uccl MPI::MPI_CXX)
target_compile_options(test_transport PRIVATE -fsycl)
target_link_options(test_transport PRIVATE -fsycl)
add_test(NAME test_transport COMMAND test_transport)
