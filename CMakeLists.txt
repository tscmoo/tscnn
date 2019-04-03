cmake_minimum_required(VERSION 3.1)
project(tscnn-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

find_package(CUDA REQUIRED)

if (WIN32)
	get_filename_component(CUDA_LIBRARY_DIR "${CUDA_CUDA_LIBRARY}" PATH)
	find_library(CUDA_nvrtc_LIBRARY nvrtc ${CUDA_LIBRARY_DIR})
	find_library(CUDA_LIBRARY cuda ${CUDA_LIBRARY_DIR})
endif()

cuda_add_executable(tscnn-test ./test.cpp)

set_property(TARGET tscnn-test PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tscnn-test Threads::Threads ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARY})


