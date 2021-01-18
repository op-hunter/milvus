/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

float dis(const float *pa, const float *pb, size_t d) {
    float ret = 0.0;
    for (auto i = 0; i < d; ++ i) {
        auto diff = (pa[i] - pb[i]);
        ret += diff * diff;
    }
    return ret;
}

void BFS(size_t nb, size_t nq, const float *xb, const float *xq, size_t d, float radius, std::string file_path, std::vector<std::vector<bool>> &idmap) {
    radius = radius * radius;
    std::ofstream fout(file_path, std::ios::out);
    float recall = 0.0;
    for (auto i = 0; i < nq; ++ i) {
        int res_cnt = 0;
        int recall_cnt = 0;
        const float *pq = xq + i * d;
        for (auto j = 0; j < nb; ++ j) {
            auto dist = dis(pq, xb + j * d, d);
            if (dist < radius) {
                res_cnt ++;
                if (i >= idmap.size()) {
                    std::cout << "fuck i = " << i << std::endl;
                    continue;
                }
                if (j >= idmap[i].size()) {
                    std::cout << "fuck i = " << i << ", j = " << j << std::endl;
                    continue;
                }
                if (idmap[i][j])
                    recall_cnt ++;
//                fout << j << " " << dist << std::endl;
            }
        }
        fout << "query " << i << " has " << res_cnt << " answers" << std::endl;
        fout << "query " << i << " recall = " << (double)recall_cnt / res_cnt << std::endl;
        recall += (double)recall_cnt / res_cnt;
    }
    fout << "total recall = " << recall << std::endl;
    fout.close();
}

int main() {
    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 20;                        // nb of queries
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


    int nlist = 100;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    index.add(nb, xb);

    auto save_ans = [&](std::string file_path, std::vector<faiss::RangeSearchPartialResult*> &results) {
        std::ofstream fout(file_path, std::ios::out);
        for (auto &prspr : results) {
            size_t ofs = 0;
            for (auto &query : prspr->queries) {
                fout << "query " << query.qno << " has " << query.nres << " results" << std::endl;
//                fout << "address of result[i] = " << prspr << ", check query.pres = " << query.pres << std::endl;
                for (auto i = 0; i < query.nres; ++ i) {
                    auto bno = (ofs + i) / prspr->buffer_size;
                    auto pos = (ofs + i) % prspr->buffer_size;
                    if (query.pres->buffers[bno].ids[pos] > nb) {
                        std::cout << "fuck the ans id = " << query.pres->buffers[bno].ids[pos] << " which is invalid, i = " << i << std::endl;
//                        continue;
                    }
                    fout << query.pres->buffers[bno].ids[pos] << " " << query.pres->buffers[bno].dis[pos] << std::endl;
                }
                ofs += query.nres;
            }
        }
        fout.close();
    };

    std::vector<std::vector<bool>> idmap(nq, std::vector<bool>(nb, false));

    {       // search xq
        std::vector<faiss::RangeSearchPartialResult*> results;
        index.range_search(nq, xq, radius * radius, results, 1024 * 16, nullptr);
        std::cout << "size of results = " << results.size() << std::endl;
        std::cout << "save faiss:IndexIVFFlat range search answer..." << std::endl;
        save_ans("/tmp/faiss_index_ivfflatl2.result", results);
        std::cout << "save ans done" << std::endl;
        for (auto &prspr: results) {
            size_t ofs = 0;
            for (auto &query: prspr->queries) {
                for (auto i = 0; i < query.nres; ++ i) {
                    auto bno = (ofs + i) / prspr->buffer_size;
                    auto pos = (ofs + i) % prspr->buffer_size;
                    if (query.qno >= idmap.size()) {
                        std::cout << "fuck query.qno = " << query.qno << ", which bigger than idmap.size() = " << idmap.size() << std::endl;
                        continue;
                    }
                    if (query.pres->buffers[bno].ids[pos] >= idmap[query.qno].size()) {
                        std::cout << "fuck query.pres->buffers[bno].ids[pos] = " << query.pres->buffers[bno].ids[pos] << ", which bigger than idmap[query.qno].size() = " << idmap[query.qno].size() << std::endl;
                        continue;
                    }
                    idmap[query.qno][query.pres->buffers[bno].ids[pos]] = true;
                }
                ofs += query.nres;
            }
        }
        std::cout << "mark ans done" << std::endl;
    }

    {
        BFS(nb, nq, xb, xq, d, radius, "/tmp/bruteforce_ivfflatl2.result", idmap);
    }


    delete [] xb;
    delete [] xq;

    return 0;
}
