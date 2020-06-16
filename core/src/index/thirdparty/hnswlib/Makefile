
TARGETS = hnsw_test

CXX             = g++ -std=c++11
CXXCPP          = g++ -std=c++11 -E
CPPFLAGS        = -DFINTEGER=int -DUSE_CPU
CXXFLAGS        = -fPIC -m64 -Wno-sign-compare  -O3 -fPIC -DELPP_THREAD_SAFE -fopenmp -O3 -DNDEBUG -fPIC -mf16c -O3
LDFLAGS         = -L/usr/local/openblas/lib
LIBS            = -lhdf5_serial

hnsw_test : hnsw_test.cpp ../faiss/libfaiss.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/knowhere/knowhere/index/vector_index/helpers/ -I/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/ $(LIBS)

.PHONY : clean
clean:
	rm -f $(TARGETS)

