/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>

#include <faiss/IndexFlat.h>

int main() {
    int d = 128;                            // dimension
    int nb = 1000000;                       // database size
    int nq = 5000;                        // nb of queries


    auto gen_data = [&](int gb, int gq, float *xb, float *xq) {
        for(int i = 0; i < gb; i++) {
            for(int j = 0; j < d; j++)
                xb[d * i + j] = drand48();
            xb[d * i] += i / 1000.;
        }

        for(int i = 0; i < gq; i++) {
            for(int j = 0; j < d; j++)
                xq[d * i + j] = drand48();
            xq[d * i] += i / 1000.;
        }
    };

    std::string save_file = "StrategyA.result";
    std::ofstream fout(save_file, std::ios::app);

    auto search = [&](int snb, int snq, int topk) {
        printf("args: snb = %d, snq = %d, topk = %d\n", snb, snq, topk);

        {       // search xq
            long *I = new long[topk * snq];
            float *D = new float[topk * snq];
            // gen data
            float *xb = new float[d * snb];
            float *xq = new float[d * snq];
            std::chrono::high_resolution_clock::time_point ts, te;
            float tot_cost = 0.0;
            std::vector<float> costs;
            costs.clear();

            for (auto i = 0; i < 5; ++ i) {
                gen_data(snb, snq, xb, xq);
                faiss::IndexFlatIP index(d);           // call constructor
                printf("is_trained = %s\n", index.is_trained ? "true" : "false");
                index.add(snb, xb);                     // add vectors to the index
                printf("ntotal = %ld\n", index.ntotal);
                ts = std::chrono::high_resolution_clock::now();
                index.search(snq, xq, topk, D, I);
                te = std::chrono::high_resolution_clock::now();
                auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
                std::cout << "the " << i + 1 << "th search costs " << search_duration << " ms." << std::endl;
                tot_cost += search_duration;
                costs.push_back(search_duration);
            }
            std::sort(costs.begin(), costs.end());
            fout << snb << " " << snq << " " << topk << " " << tot_cost / 5.0 << " (" << costs[2] << ")" << std::endl;

            // print results
            /*
            printf("I (5 first results)=\n");
            for(int i = 0; i < 5; i++) {
                for(int j = 0; j < k; j++)
                    printf("%5ld ", I[i * k + j]);
                printf("\n");
            }

            printf("I (5 last results)=\n");
            for(int i = nq - 5; i < nq; i++) {
                for(int j = 0; j < k; j++)
                    printf("%5ld ", I[i * k + j]);
                printf("\n");
            }
            */

            delete [] xb;
            delete [] xq;
            delete [] I;
            delete [] D;
        }
    };

//    std::vector<int> nbs = {1024, 4096, 16384, 65536, 100000};
//    std::vector<int> nqs = {1, 4, 6, 8, 12, 1024, 4096, 10000};
//    std::vector<int> topks = {1, 4, 6, 8, 12, 1024, 4096, 10000};
//    std::vector<int> nbs = {1, 128, 256, 512, 1024, 4096, 16384, 65536, 100000, 1000000};
//    std::vector<int> nqs = {1, 12, nq};
//    std::vector<int> topks = {128};
//    std::vector<int> nbs = {128, 256, 512, 1024, 2048, 4096, 16384};
    std::vector<int> nbs = {1000, 10000, 50000, 100000, 500000, 1000000, 10000000};
    std::vector<int> nqs = {1000};
    std::vector<int> topks = {1};

    for (auto &snb : nbs) {
        for (auto &snq : nqs) {
            for (auto &tpk : topks) {
                search(snb, snq, tpk);
            }
        }
    }

    fout.close();

    return 0;
}
