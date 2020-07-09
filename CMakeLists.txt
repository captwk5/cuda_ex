CMAKE_MINIMUM_REQUIRED ( VERSION 3.8.1 )

SET ( MODEL "cuda_app" )

enable_language(CUDA)

INCLUDE (FindCUDA)

SET ( CMAKE_INSTALL_PREFIX /usr/local/cuda-10.2/bin )

SET ( SRC
        main.cpp
        cuda_ex.cu
        )

#SET ( CMAKE_C_COMPILER "nvcc" )
#SET ( CMAKE_CXX_FLAGS "-std=c++11 -pthread")
#SET ( CUDA_NVCC_FLAG "${CUDA_NVCC_FLAGS} -std=c++11")

FIND_PACKAGE(CUDA REQUIRED)

include_directories( ${CUDA_INCLUDE_DIRS} )

ADD_EXECUTABLE( ${MODEL} ${SRC} )

set_property(TARGET ${MODEL} PROPERTY CUDA_STANDARD 11)