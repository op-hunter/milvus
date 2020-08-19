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


void query(float *xb, int topk, int nq, int d, int offset, HierarchicalNSW<float> *hnsw, int &query_time, int &recall) {
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
            ret = hnsw->searchKnn((void*)(xb + (offset + i) * d), topk, compare, nullptr, xb);
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
        query_time = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        for (auto i = 0; i < nq; ++ i) {
//            std::cout << "query " << i + 1 << ": ";
            for (auto j = 0; j < topk; ++ j) {
                if (I[i * topk + j] == offset + i)
                    correct_cnt ++;
//                std::cout << "(" << I[i * topk + j] << ", " << D[i * topk + j] << ")";
//                if (j == topk - 1)
//                    std::cout << std::endl;
//                else
//                    std::cout << ",";
            }
        }
        std::cout << "correct query of top" << topk << " is " << correct_cnt << std::endl;
        recall = correct_cnt;

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

    std::vector<int>og1(8, 0);
    std::vector<int>og2(8, 0);
    std::vector<int>ogpa(8, 0);
    std::vector<std::vector<std::vector<int> > > tdarr(M * 2 + 1, std::vector<std::vector<int> >(4, std::vector<int>(8, 0)));
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
    og1[0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
//    hnsw->show_l2_dis_invoke_times();
    hnsw->setEf(efSearch);
    std::string error_file = "/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/tutorial/cpp/test/error.txt";
    std::ofstream errf(error_file, std::ios::out);
    errf << "check original graph:" << std::endl;
    hnsw->check_graph(errf);
    errf << "original graph check done." << std::endl;
    errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    errf << std::endl;

    std::cout << "original graph has " << hnsw->get_edges() << " edges." << std::endl;

    std::cout << "query original:" << std::endl;
    for (auto i = 0; i < 3; ++ i) {
        int query_time, recall;
        query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
        og1[2 + i * 2] = query_time;
        og1[3 + i * 2] = recall;
    }


    char *ori_level0 = hnsw->data_level0_memory_;
    int M0 = M << 1;
    std::cout << "start static degree adjust." << std::endl;
    for (int eo = 0; eo <= M0; ++ eo) {
        int ei = M0 - eo;
        char *sa_memory = nullptr;
        errf << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>(" << eo << " " << ei << ")<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        std::cout << "eo = " << eo << " ei = " << ei << std::endl;
        ts = std::chrono::high_resolution_clock::now();
        hnsw->ConstructAdjustedGraph(eo, ei, xb, sa_memory);
        te = std::chrono::high_resolution_clock::now();
        std::cout << "static degree adjustment costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        tdarr[eo][0][0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        hnsw->data_level0_memory_ = sa_memory;
        errf << "check sa graph:" << std::endl;
        hnsw->check_graph(errf);
        errf << "sa graph check done." << std::endl;
        errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        errf << std::endl;
        std::cout << "current graph has " << hnsw->get_edges() << " edges." << std::endl;
        std::cout << "query static degree adjustment:" << std::endl;
        for (auto i = 0; i < 3; ++ i) {
            int query_time, recall;
            query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
            tdarr[eo][0][2 + i * 2] = query_time;
            tdarr[eo][0][3 + i * 2] = recall;
        }

        std::cout << "apply path adjust:" << std::endl;
        char *sap_memory = nullptr;
        ts = std::chrono::high_resolution_clock::now();
        tdarr[eo][1][1] = hnsw->AdjustPath2(sa_memory, sap_memory, xb);
        te = std::chrono::high_resolution_clock::now();
        std::cout << "static degree adjustment + path adjust costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        tdarr[eo][1][0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        hnsw->data_level0_memory_ = sap_memory;
        errf << "check sapa graph:" << std::endl;
        hnsw->check_graph(errf);
        errf << "sapa graph check done." << std::endl;
        errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        errf << std::endl;
        std::cout << "current graph has " << hnsw->get_edges() << " edges." << std::endl;
        std::cout << "query static degree adjust + path adjust: " << std::endl;
        for (auto i = 0; i < 3; ++ i) {
            int query_time, recall;
            query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
            tdarr[eo][1][2 + i * 2] = query_time;
            tdarr[eo][1][3 + i * 2] = recall;
        }

        free(sa_memory);
        free(sap_memory);
        hnsw->data_level0_memory_ = ori_level0;
    }

    std::cout << "start static degree adjust with constraint." << std::endl;
    for (int eo = 0; eo <= M0; ++ eo) {
        int ei = M0 - eo;
        char *sac_memory = nullptr;
        std::cout << "eo = " << eo << ", ei = " << ei << std::endl;
        errf << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>(" << eo << " " << ei << ")<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        ts = std::chrono::high_resolution_clock::now();
        hnsw->ConstructAdjustedGraphWithConstraint(eo, ei, xb, sac_memory);
        te = std::chrono::high_resolution_clock::now();
        std::cout << "static degree adjustment with constraint costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        tdarr[eo][2][0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        hnsw->data_level0_memory_ = sac_memory;
        errf << "check sac graph:" << std::endl;
        hnsw->check_graph(errf);
        errf << "sac graph check done." << std::endl;
        errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        errf << std::endl;
        std::cout << "current graph has " << hnsw->get_edges() << " edges." << std::endl;
        std::cout << "query static degree adjustment with constraint:" << std::endl;
        for (auto i = 0; i < 3; ++ i) {
            int query_time, recall;
            query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
            tdarr[eo][2][2 + i * 2] = query_time;
            tdarr[eo][2][3 + i * 2] = recall;
        }

        std::cout << "apply path adjust on constraint:" << std::endl;
        char *sacp_memory = nullptr;
        ts = std::chrono::high_resolution_clock::now();
        tdarr[eo][3][1] = hnsw->AdjustPath2(sac_memory, sacp_memory, xb);
        te = std::chrono::high_resolution_clock::now();
        std::cout << "static degree adjustment with constraint + path adjust costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
        tdarr[eo][3][0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        hnsw->data_level0_memory_ = sacp_memory;
        errf << "check sacpa graph:" << std::endl;
        hnsw->check_graph(errf);
        errf << "sacpa graph check done." << std::endl;
        errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        errf << std::endl;
        std::cout << "current graph has " << hnsw->get_edges() << " edges." << std::endl;
        std::cout << "query static degree adjust with constraint + path adjust: " << std::endl;
        for (int i = 0; i < 3; ++ i) {
            int query_time, recall;
            query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
            tdarr[eo][3][2 + i * 2] = query_time;
            tdarr[eo][3][3 + i * 2] = recall;
        }

        free(sac_memory);
        free(sacp_memory);
        hnsw->data_level0_memory_ = ori_level0;
    }

    hnsw->data_level0_memory_ = ori_level0;
    std::cout << "query origin again: " << std::endl;
    for (int i = 0; i < 3; ++ i) {
        int query_time, recall;
        query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
        og2[2 + i * 2] = query_time;
        og2[3 + i * 2] = recall;
    }

    std::cout << "origin graph~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hnsw->indegree_stats(0);

    errf << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>(pa)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    std::cout << "only path adjust:" << std::endl;
    char *op_memory = nullptr;
    ts = std::chrono::high_resolution_clock::now();
    ogpa[1] = hnsw->AdjustPath2(ori_level0, op_memory, xb);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "path adjustment on original graph costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    ogpa[0] = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    hnsw->data_level0_memory_ = op_memory;
    errf << "check pa graph:" << std::endl;
    hnsw->check_graph(errf);
    errf << "pa graph check done." << std::endl;
    errf << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    errf << std::endl;
    std::cout << "query path adjust on original graph: " << std::endl;
    for (int i = 0; i < 3; ++ i) {
        int query_time, recall;
        query(xb, topk, nq, d, i * nq, hnsw.get(), query_time, recall);
        ogpa[2 + i * 2] = query_time;
        ogpa[3 + i * 2] = recall;
    }
    std::cout << "path adjust grpath~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hnsw->indegree_stats(1);
    free(op_memory);

    hnsw->data_level0_memory_ = ori_level0;

    hnsw->show_l2_dis_invoke_times();


    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "========================grogeous divide line=============================" << std::endl;
    std::string filename = "/home/zilliz/workspace/dev/milvus/milvus/core/src/index/thirdparty/faiss/tutorial/cpp/test/";
    filename += std::to_string(M) + "-" + std::to_string(efConstruction) + "-" + std::to_string(efSearch) + "-" + std::to_string(topk) + ".stats";
    std::ofstream ff(filename, std::ios::out);
    ff << "og1:" << std::endl;
    for (auto i = 0; i < og1.size(); ++ i)
        ff << og1[i] << " ";
    ff << std::endl;
    ff << std::endl;

    for (auto eo = 0; eo <= M0; ++ eo) {
        ff << "eo = " << eo << " ei = " << M0 - eo << std::endl;
        ff << "sa:" << std::endl;
        for (auto i = 0; i < tdarr[eo][0].size(); ++ i)
            ff << tdarr[eo][0][i] << " ";
        ff << std::endl;
        ff << "sapa:" << std::endl;
        for (auto i = 0; i < tdarr[eo][1].size(); ++ i)
            ff << tdarr[eo][1][i] << " ";
        ff << std::endl;
        ff << "sac:" << std::endl;
        for (auto i = 0; i < tdarr[eo][2].size(); ++ i)
            ff << tdarr[eo][2][i] << " ";
        ff << std::endl;
        ff << "sacpa:" << std::endl;
        for (auto i = 0; i < tdarr[eo][3].size(); ++ i)
            ff << tdarr[eo][3][i] << " ";
        ff << std::endl;
    }

    ff << "og2:" << std::endl;
    for (auto i = 0; i < og2.size(); ++ i)
        ff << og2[i] << " ";
    ff << std::endl;
    ff << "ogpa:" << std::endl;
    for (auto i = 0; i < ogpa.size(); ++ i)
        ff << ogpa[i] << " ";
    ff << std::endl;
    ff << std::endl;

    ff.close();
    errf.close();

    delete [] xb;
    delete [] xq;

    return 0;
}
