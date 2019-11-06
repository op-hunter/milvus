export MILVUS_CORE_DIR=${TRAVIS_BUILD_DIR}/core
export MILVUS_BUILD_DIR=${TRAVIS_BUILD_DIR}/core/cmake_build
export MILVUS_INSTALL_PREFIX=/opt/milvus
export MILVUS_TRAVIS_COVERAGE=${MILVUS_TRAVIS_COVERAGE:=0}

if ["$MILVUS_TRAVIS_COVERAGE" == "1"]; then
  export MILVUS_CPP_COVERAGE_FILE=${TRAVIS_BUILD_DIR}/output_new.info
fi

export MILVUS_BUILD_TYPE=${MILVUS_BUILD_TYPE:=Release}
