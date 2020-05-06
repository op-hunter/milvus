#/bin/bash

cd src/index/thirdparty/faiss/
make clean
cd -
rm -rf cmake_build/*
./build.sh > build3.output
