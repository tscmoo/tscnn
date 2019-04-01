cmake_minimum_required(VERSION 3.1)
project(tscnn-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

find_package(CUDA REQUIRED)

cuda_add_executable(tscnn-test ./test.cpp)

set_property(TARGET tscnn-test PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tscnn-test Threads::Threads ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARY})


