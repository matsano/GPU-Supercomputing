#cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_30” ..
cmake_minimum_required(VERSION 3.8)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(Matrice_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EXE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

list(APPEND HEADER_REP ${ROOT_DIR}/inc  ${ROOT_DIR}/src/CUDA ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc")

file(GLOB CPP_FILES ${ROOT_DIR}/src/CPP/*.cpp)
file(GLOB CU_FILES ${ROOT_DIR}/src/CPP/*.cu)
file(GLOB_RECURSE HEADER_FILES ${ROOT_DIR}/src/CUDA/*.cuh ${ROOT_DIR}/inc/*.h)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

#COMPILATION LIBRAIRIE PUIS EXECUTABLE
#add_library(matrice ${CPP_FILES} ${CU_FILES} ${HEADER_FILES})

#set_target_properties(matrice PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_compile_features(matrice PUBLIC cxx_std_11)
#target_include_directories(matrice PUBLIC "${HEADER_REP}")

#add_executable(matrice_exe "${EXE_DIR}/src/CPP/matrixMul_main.cu")
#target_link_libraries(matrice_exe PUBLIC matrice)

#COMPILATION EXECUTABLE (sans generer avant une librairie => tous les fichers sources sont pris en compte)
add_executable(matrice_exe ${CPP_FILES} ${CU_FILES} ${HEADER_FILES})

set_target_properties(matrice_exe PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(matrice_exe PUBLIC cxx_std_11)
target_include_directories(matrice_exe PUBLIC "${HEADER_REP}")
target_link_libraries(matrice_exe PUBLIC cublas)

