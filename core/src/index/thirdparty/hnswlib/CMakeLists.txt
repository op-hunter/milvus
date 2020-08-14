CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

PROJECT(hnswoy)

SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_C_FLAGS_RELEASE "-g -Wall -fopenmp")
SET(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -fopenmp")

INCLUDE_DIRECTORIES(../faiss)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)
SET(HNSWOY
        ${DIR_SRCS}
        ../faiss
        )

ADD_EXECUTABLE(${PROJECT_NAME} ${HNSWOY})

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                      hdf5_serial
                      pthread
                      /usr/local/openblas/lib/libopenblas.a
                      /home/zilliz/workspace/dev/milvus/milvus/core/cmake_build/src/index/knowhere/libknowhere.a
                      /home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/libfaiss.a
                      )
