#cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_30” ..
cmake_minimum_required(VERSION 2.8)

project(reduce)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDA REQUIRED) 
list(APPEND HEADER_REP ${ROOT_DIR}/ ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc")

#file(GLOB CPP_FILES ${ROOT_DIR}/src/CPP/*.cpp)
#file(GLOB CU_FILES ${ROOT_DIR}/src/CPP/*.cu)
set(CU_FILES ${ROOT_DIR}/Reduce.cu)
#file(GLOB_RECURSE HEADER_FILES ${ROOT_DIR}/src/CUDA/*.cuh ${ROOT_DIR}/inc/*.h)


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "-lineinfo;-I\"${ROOT_DIR}/inc\" -I\"${ROOT_DIR}/\" -I\"/usr/local/cuda/samples/common/inc/\" -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

cuda_add_executable(reduce ${CPP_FILES} ${CU_FILES} ${HEADER_FILES})
set_target_properties(reduce PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(reduce PUBLIC ${ROOT_DIR}/inc/ /usr/local/cuda/samples/common/inc/)









