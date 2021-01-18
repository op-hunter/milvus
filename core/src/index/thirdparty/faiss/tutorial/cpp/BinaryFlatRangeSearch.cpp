/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <faiss/IndexBinaryFlat.h>
#include <sys/time.h>
#include <unistd.h>

// #define TEST_HAMMING

long int getTime(timeval end, timeval start) {
	return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

float dis(const uint64_t *pa, const uint64_t *pb) {
    return __builtin_popcountl((*pa) ^ (*pb));
}

void BFS(size_t nb, size_t nq, const uint64_t *xb, const uint64_t *xq, size_t d, size_t radius, std::string file_path) {
    std::ofstream fout(file_path, std::ios::out);

    for (auto i = 0; i < nq; ++ i) {
        int res_cnt = 0;
        const uint64_t* pq = xq + i;
        for (auto j = 0; j < nb; ++ j) {
            auto dist = dis(pq, xb + j);
            if (dist < radius) {
                res_cnt ++;
                fout << j << " " << dist << std::endl;
            }
        }
        fout << "query " << i << " has " << res_cnt << " answers" << std::endl;
    }
    fout.close();
}

int main() {
    // freopen("0.txt", "w", stdout);

    size_t d = 64;                          // dimension
    size_t nb = 400000;                    // database size
    size_t nq = 20;                          // nb of queries
    size_t radius = 20;

    uint8_t *xb = new uint8_t[d * nb / sizeof(uint8_t)];
    uint8_t *xq = new uint8_t[d * nq / sizeof(uint8_t)];

    // skip 0
    lrand48();

    size_t size_to_long = d * nb / sizeof(int32_t);
    for(size_t i = 0; i < size_to_long; i++) {
        ((int32_t*)xb)[i] = lrand48();
    }

    size_to_long = d * nq / sizeof(long int);
    for(size_t i = 0; i < size_to_long; i++) {
        ((int32_t*)xq)[i] = lrand48();
    }

    printf("test haming\n");
    faiss::IndexBinaryFlat index(d, faiss::MetricType::METRIC_Hamming);

    index.add(nb, xb);
    printf("ntotal = %ld d = %d\n", index.ntotal, index.d);

    auto save_ans = [&](std::string file_path, std::vector<faiss::RangeSearchPartialResult*> &result) {
        std::ofstream fout(file_path, std::ios::out);
        for (auto &prspr : result) {
            size_t ofs = 0;
            for (auto &query : prspr->queries) {
                fout << "query " << query.qno << " has " << query.nres << " results" << std::endl;
                for (auto i = 0; i < query.nres; ++ i) {
                    auto bno = (ofs + i) / prspr->buffer_size;
                    auto pos = (ofs + i) % prspr->buffer_size;
                    fout << query.pres->buffers[bno].ids[pos] << " " << query.pres->buffers[bno].dis[pos] << std::endl;
                }
                ofs += query.nres;
            }
        }
        fout.close();
    };

    {
        std::vector<faiss::RangeSearchPartialResult*> results;
        index.range_search(nq, xq, radius, results, 1024 * 16, nullptr);
        std::cout << "size of results: " << results.size() << std::endl;
        std::cout << "save faiss::IndexBinaryFlat range search answer..." << std::endl;
        save_ans("/tmp/faiss_index_binary_flathamming.result", results);
    }

    {
        BFS(nb, nq, reinterpret_cast<uint64_t*>(xb), reinterpret_cast<uint64_t*>(xq), d, radius, "/tmp/bruteforce_binary_flat_hamming.result");
    }

    delete [] xb;
    delete [] xq;

    return 0;
}


