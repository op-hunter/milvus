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

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

float dis(const float *pa, const float *pb, size_t d) {
    float ret = 0.0;
    for (auto i = 0; i < d; ++ i) {
        auto diff = (pa[i] - pb[i]);
        ret += diff * diff;
    }
    return ret;
}

void BFS(size_t nb, size_t nq, const float *xb, const float *xq, size_t d, float radius, std::string file_path) {
    radius = radius * radius;
    std::ofstream fout(file_path, std::ios::out);
    for (auto i = 0; i < nq; ++ i) {
        int res_cnt = 0;
        const float *pq = xq + i * d;
        for (auto j = 0; j < nb; ++ j) {
            auto dist = dis(pq, xb + j * d, d);
            if (dist <= radius) {
                res_cnt ++;
                fout << j << " " << dist << std::endl;
            }
        }
        fout << "query " << i + 1 << " has " << res_cnt << " answers" << std::endl;
    }
    fout.close();
}

int main() {
    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 100;                        // nb of queries
    float radius = 3;

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    faiss::IndexFlatL2 index(d);           // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);                     // add vectors to the index
    printf("ntotal = %ld\n", index.ntotal);

    auto save_ans = [&](std::string file_path, std::vector<faiss::RangeSearchPartialResult*> &results) {
        std::ofstream fout(file_path, std::ios::out);
        for (auto &prspr : results) {
            size_t ofs = 0;
            for (auto &query : prspr->queries) {
                fout << "query " << query.qno << " has " << query.nres << " results" << std::endl;
                fout << "address of result[i] = " << prspr << ", check query.pres = " << query.pres << std::endl;
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

    {       // faiss::IndexFlat.range_search

        std::vector<faiss::RangeSearchPartialResult*> results;
        index.range_search(nq, xq, radius * radius, results, 1024 * 16, nullptr);
        std::cout << "size of results = " << results.size() << std::endl;
        std::cout << "save faiss:IndexFlat range search answer..." << std::endl;
        save_ans("/tmp/faiss_index_flatl2.result", results);
    }

    {
        BFS(nb, nq, xb, xq, d, radius, "/tmp/bruteforce_flatl2.result"); // bruteforce and save
    }

    delete [] xb;
    delete [] xq;

    return 0;
}
