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
#include <cassert>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>


int main() {
    int d = 64;                            // dimension
    int nb = 5000000;                       // database size
    int nq = 1000;                        // nb of queries

    std::chrono::high_resolution_clock::time_point t0, t1;
    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "gen xb costs: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms " << std::endl;

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }
    t0 = std::chrono::high_resolution_clock::now();
    std::cout << "gen xq costs: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(t0 - t1).count() << "ms " << std::endl;

    faiss::gpu::StandardGpuResources res;

    // Using a flat index

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

    printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
    t0 = std::chrono::high_resolution_clock::now();
    index_flat.add(nb, xb);  // add vectors to the index
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "index add costs: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms " << std::endl;
    printf("ntotal = %ld\n", index_flat.ntotal);

//    int k = 4;
    int topk[5] = {32, 64, 128, 256, 1024};
//    std::string file_location = "/home/zilliz/workspace/test/faiss_k_selection/standard/";
    std::string file_location = "/home/zilliz/workspace/test/faiss_k_selection/upgrade/";

    for (int i = 0; i < 5; ++ i) {       // search xq
        int k = topk[i];
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        t0 = std::chrono::high_resolution_clock::now();
        index_flat.search(nq, xq, k, D, I);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "when topk = " << k << ", ";
        std::cout << "index search costs: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms " << std::endl;

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

        // save results
        std::string file_name = file_location + std::to_string(k) + ".txt";
        std::ofstream fout(file_name.c_str(), std::ios::out);
        for (int i = 0; i < nq; ++ i) {
            for (int j = 0; j < k; ++ j) {
                fout << I[i * k + j] << " ";
            }
            fout << std::endl;
        }
        fout.close();

        delete [] I;
        delete [] D;
    }

    delete [] xb;
    delete [] xq;

    return 0;
}
