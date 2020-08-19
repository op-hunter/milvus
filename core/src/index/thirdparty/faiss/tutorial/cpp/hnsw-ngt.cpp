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
#include <gperftools/profiler.h>

#include "../../../hnswlib/include/hnswalg.h"
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


void query(float *xb, int topk, int nq, int d, HierarchicalNSW<float> *hnsw) {
    std::cout << "query startiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii" << std::endl;
    {       // search xq
        long *I = new long[topk * nq];
        float *D = new float[topk * nq];

        using P = std::pair<float, int64_t>;
        auto compare = [](const P& v1, const P& v2) { return v1.first < v2.first; };

        auto ts = std::chrono::high_resolution_clock::now();
        int correct_cnt = 0;
#pragma omp parallel for
        for (int i = 0; i < nq; ++ i) {
            std::vector<P> ret;
            ret = hnsw->searchKnn((void*)(xb + i * d), topk, compare, nullptr, xb);
            //std::cout << "ret of query " << i << " is: " << std::endl;
            //for (auto fff = 0; fff < ret.size(); ++ fff) {
            //    std::cout << ret[fff].second << ", " << ret[fff].first << std::endl;
            //}
            for (auto j = 0; j < topk; ++ j) {
                I[i * topk + j] = ret[j].second;
                D[i * topk + j] = ret[j].first;
            }
            /*
            {
                for (auto j = 0; j < topk; ++ j) {
                    if (i == ret[j].second || ret[j].first < 1e-5) {
                        correct_cnt ++;
                        break;
                    }
                    std::cout << "query " << i << ", topk " << j << ": id = " << ret[j].second << ", dis = " << ret[j].first << std::endl;
                }
            }
            */
        }
        auto te = std::chrono::high_resolution_clock::now();
        std::cout << "search " << nq << " times costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        for (auto i = 0; i < nq; ++ i) {
//            std::cout << "query " << i + 1 << ": ";
            for (auto j = 0; j < topk; ++ j) {
                if (I[i * topk + j] == i)
                    correct_cnt ++;
//                std::cout << "(" << I[i * topk + j] << ", " << D[i * topk + j] << ")";
//                if (j == topk - 1)
//                    std::cout << std::endl;
//                else
//                    std::cout << ",";
            }
        }
        std::cout << "correct query of top" << topk << " is " << correct_cnt << std::endl;

        delete [] I;
        delete [] D;
    }
    std::cout << "query done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
}


int main(int argc, char **argv) {
    int d = 128;                            // dimension
    int nb = 1000000;                       // database size
//    int nb = 100000;                       // database size
    int nq = 10000;                        // nb of queries
    //int nq = 10;                        // nb of queries
    int M = 16;
    int efConstruction = 200;
    int efSearch = 100;
    int topk = 1;


    if (argc != 5) {
        std::cout << "invalid number of args, except 5 but only get " << argc << std::endl;
        return EXIT_FAILURE;
    }
    M = atoi(argv[1]);
    efConstruction = atoi(argv[2]);
    efSearch = atoi(argv[3]);
    topk = atoi(argv[4]);
    std::cout << "argvs: M = " << M << ", efConstruction = " << efConstruction << ", efSearch = " << efSearch << ", topk = " << topk << std::endl;

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    /*
    srand(12345);
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
    */

    std::string data_location = "/home/zilliz/workspace/data/sift-128-euclidean.hdf5";
    std::cout << "start load sift data ..." << std::endl;
    auto ts = std::chrono::high_resolution_clock::now();
    LoadData(data_location, xb, "train", d, nb);
    std::cout << "dim: " << d << ", rows: " << nb << std::endl;
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "load data costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;


    hnswlib::L2Space *space = new hnswlib::L2Space(d);
    std::shared_ptr<HierarchicalNSW<float>> hnsw = std::make_shared<HierarchicalNSW<float>>(space, nb, M, efConstruction);
    hnsw->show_l2_dis_invoke_times();

    ts = std::chrono::high_resolution_clock::now();
    //ProfilerStart("milvus.profile");
    hnsw->addPoint(xb, 0, 0, 0);
#pragma omp parallel for
    for (int i = 1; i < nb; ++ i) {
        hnsw->addPoint(xb, i, 0, i);
    }
    //ProfilerStop();
    te = std::chrono::high_resolution_clock::now();
    std::cout << "build index costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    hnsw->show_l2_dis_invoke_times();
    hnsw->setEf(efSearch);

    std::cout << "query original:" << std::endl;
    query(xb, topk, nq, d, hnsw.get());

    char *sa_memory = nullptr;
    int eo, ei;
    eo = (M << 2) / 3;
    ei = (M << 1) - eo;
    std::cout << "start static degree adjust, eo = " << eo << ", ei = " << ei << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    hnsw->ConstructAdjustedGraph(eo, ei, xb, sa_memory);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "static degree adjustment costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;

    char *ori_level0 = hnsw->data_level0_memory_;
    hnsw->data_level0_memory_ = sa_memory;
    std::cout << "query static degree adjustment:" << std::endl;
    query(xb, topk, nq, d, hnsw.get());

    std::cout << "static degree adjust + path adjust: " << std::endl;
    char *sap_memory = nullptr;
    ts = std::chrono::high_resolution_clock::now();
    hnsw->AdjustPath2(sa_memory, sap_memory, xb);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "static degree adjustment + path adjust costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    hnsw->data_level0_memory_ = sap_memory;
    std::cout << "query static degree adjust + path adjust: " << std::endl;
    query(xb, topk, nq, d, hnsw.get());
    free(sa_memory);
    free(sap_memory);

    char *sac_memory = nullptr;
    hnsw->data_level0_memory_ = ori_level0;
    std::cout << "start static degree adjust with constraint, eo = " << eo << ", ei = " << ei << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    hnsw->ConstructAdjustedGraphWithConstraint(eo, ei, xb, sac_memory);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "static degree adjustment with adjustment costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;

    hnsw->data_level0_memory_ = sac_memory;
    std::cout << "query static degree adjustment with constraint:" << std::endl;
    query(xb, topk, nq, d, hnsw.get());

    std::cout << "static degree adjust with constraint+ path adjust: " << std::endl;
    char *sacp_memory = nullptr;
    ts = std::chrono::high_resolution_clock::now();
    hnsw->AdjustPath2(sac_memory, sacp_memory, xb);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "static degree adjustment with constraint + path adjust costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    hnsw->data_level0_memory_ = sacp_memory;
    std::cout << "query static degree adjust with constraint + path adjust: " << std::endl;
    query(xb, topk, nq, d, hnsw.get());
    free(sac_memory);
    free(sacp_memory);

    hnsw->data_level0_memory_ = ori_level0;
    std::cout << "query origin: " << std::endl;
    query(xb, topk, nq, d, hnsw.get());

    std::cout << "origin graph~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hnsw->indegree_stats(0);

    char *op_memory = nullptr;
    ts = std::chrono::high_resolution_clock::now();
    hnsw->AdjustPath2(ori_level0, op_memory, xb);
    char *op_memory2 = nullptr;
    hnsw->AdjustPath2(op_memory, op_memory2, xb);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "path adjustment on original graph costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    hnsw->data_level0_memory_ = op_memory2;
    std::cout << "query path adjust on original graph: " << std::endl;
    query(xb, topk, nq, d, hnsw.get());
    std::cout << "path adjust grpath~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hnsw->indegree_stats(1);
    free(op_memory);
    free(op_memory2);

    hnsw->data_level0_memory_ = ori_level0;

    hnsw->show_l2_dis_invoke_times();

    delete [] xb;
    delete [] xq;

    return 0;
}
