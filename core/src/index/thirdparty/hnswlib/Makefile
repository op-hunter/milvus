
TARGETS = hnsw_faiss_sq8_test hnsw_sq8_small hnsw_small

CXX             = g++ -std=c++11
CXXCPP          = g++ -std=c++11 -E
CPPFLAGS        = -DFINTEGER=int -DUSE_CPU
CXXFLAGS        = -fPIC -m64 -Wno-sign-compare  -O3 -fPIC -DELPP_THREAD_SAFE -fopenmp -O3 -DNDEBUG -fPIC -mf16c -O3
LDFLAGS         = -L/usr/local/openblas/lib
LIBS            = -lhdf5_serial -lopenblas

all : hnsw_faiss_sq8_test hnsw_sq8_small hnsw_small

.PHONY : all

#hnsw_test : hnsw_test.cpp ../faiss/libfaiss.a
#	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ $(LIBS)

#hnsw_sq8_test : hnsw_sq8_test.cpp ../faiss/libfaiss.a
#	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ $(LIBS)

hnsw_faiss_sq8_test : hnsw_faiss_sq8_test.cpp ../faiss/libfaiss.so
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/cmake_build/src/index/faiss_ep-prefix/src/faiss_ep/include/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ -I/usr/local/openblas/lib/ $(LIBS)

hnsw_sq8_small : hnsw_sq8_small.cpp ../faiss/libfaiss.so
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/cmake_build/src/index/faiss_ep-prefix/src/faiss_ep/include/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ -I/usr/local/openblas/lib/ $(LIBS)

hnsw_small : hnsw_small.cpp ../faiss/libfaiss.so
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/cmake_build/src/index/faiss_ep-prefix/src/faiss_ep/include/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ -I/usr/local/openblas/lib/ $(LIBS)

#%: %.cpp ../faiss/libfaiss.a
	#$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ $(LIBS)

clean:
	rm -f $(TARGETS)

