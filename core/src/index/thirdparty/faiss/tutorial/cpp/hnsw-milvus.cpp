/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include "../../../hnswlib/hnswalg.h"
#include "/usr/include/hdf5/serial/hdf5.h"
#include "/usr/include/hdf5/serial/H5Cpp.h"


void LoadData(const std::string file_location, float *&data, const std::string data_name, int &dim, int &num_vets) {
    hid_t fd;
    herr_t status;
    fd = H5Fopen(file_location.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t dataset_id;
    dataset_id = H5Dopen2(fd, data_name.c_str(), H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    hid_t dspace = H5Dget_space(dataset_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    num_vets = dims[0];
    dim = dims[1];
    status = H5Dclose(dataset_id);
    status = H5Fclose(fd);
}

using namespace hnswlib;
int main() {
    int d = 128;                            // dimension
    int nb = 1000000;                       // database size
    int nq = 10000;                        // nb of queries
    int M = 16;
    int efConstruction = 200;
    int efSearch = 100;
    int topk = 1;

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    srand(12345);
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }
    /*
    std::string data_location = "/home/zilliz/workspace/data/sift-128-euclidean.hdf5";
    std::cout << "start load sift data ..." << std::endl;
    auto ts = std::chrono::high_resolution_clock::now();
    LoadData(data_location, xb, "train", d, nb);
    std::cout << "dim: " << d << ", rows: " << nb << std::endl;
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "load data costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    */

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }


    L2Space *space = new L2Space(d);
    std::shared_ptr<HierarchicalNSW<float>> hnsw = std::make_shared<HierarchicalNSW<float>>(space, nb, M, efConstruction);

    auto ts = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(xb, 0);
#pragma omp parallel for
    for (int i = 1; i < nb; ++ i) {
        hnsw->addPoint(xb + i * d, i);
    }
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "build index costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    hnsw->show_stats();
    hnsw->print_stats();

    {       // search xq
//        long *I = new long[k * nq];
//        float *D = new float[k * nq];

        using P = std::pair<float, int64_t>;
        auto compare = [](const P& v1, const P& v2) { return v1.first < v2.first; };
        hnsw->setEf(efSearch);

        ts = std::chrono::high_resolution_clock::now();
        int correct_cnt = 0;
#pragma omp parallel for
        for (int i = 0; i < nq; ++ i) {
            std::vector<P> ret;
            ret = hnsw->searchKnn((void*)(xb + i * d), topk, compare, nullptr);
//            {
//                if (i == ret[0].second || ret[0].first < 1e-5)
//                    correct_cnt ++;
//            }
        }
        te = std::chrono::high_resolution_clock::now();
        std::cout << "search " << nq << " times costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        std::cout << "correct query of top" << topk << " is " << correct_cnt << std::endl;

//        delete [] I;
//        delete [] D;
    }



    delete [] xb;
    delete [] xq;

    return 0;
}

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#include "../hnswlib/hnswalg.h"
#include "../hnswlib/space_ip.h"
#include "../hnswlib/space_l2.h"

long int getTime(timeval end, timeval start) {
    return 1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
}

using namespace hnswlib;

int main() {
    int M = 16;

    int d = 128;                            // dimension
    int nb = 1000000;                       // database size
    int nq = 1000;                        // nb of queries
    int k = 1;

    srand(12345);

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
    }


    hnswlib::SpaceInterface<float>* space = new hnswlib::L2Space(d);
    hnswlib::HierarchicalNSW<float> index(space, nb, M, 40);


    timeval t0, t1;

    gettimeofday(&t0, 0);

    index.addPoint(xb, 0);
#pragma omp parallel for
    for (int i=1;i<nb;i++){
        index.addPoint(xb + d * i, i);
    }

    gettimeofday(&t1, 0);
    printf("build time %d\n", getTime(t1,t0));


    index.setEf(40);

    using P = std::pair<float, int64_t>;
    auto compare = [](const P& v1, const P& v2) { return v1.first < v2.first; };


    gettimeofday(&t0, 0);

#pragma omp parallel for
    for (unsigned int i = 0; i < nq; ++i) {
        std::vector<P> ret;
        const float *single_query = (float *) xq + i * d;

        ret = index.searchKnn((float *) single_query, k, compare, nullptr);

//        for (int i=0;i<k;i++) {
//            printf("%lf %d\n", ret[i].first, ret[i].second);
//        }
    }

    gettimeofday(&t1, 0);
    printf("serach time %d\n", getTime(t1,t0));


    delete [] xb;
    delete [] xq;

    return 0;
}
*/

