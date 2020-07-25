#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "/usr/include/hdf5/serial/hdf5.h"
#include "/usr/include/hdf5/serial/H5Cpp.h"
//#include "hnswalg2.h"
#include "hnswalg_faiss_sq8.h"
//#include "hnswalg.h"

using namespace hnswlib;
using namespace H5;

void LoadData(const std::string file_location, float *&data, const std::string data_name, int &dim, int &num_vets) {
    int cnt = 0;
    for (int i = 0; i < num_vets; ++ i) {
        for (int j = 0; j < dim; ++ j) {
            data[cnt] = float(cnt);
            cnt ++;
        }
    }
}

int main() {
    int dim = 3;
    int nb = 100;
    int nq = 1;
    int topk = 1;
    int M = 16;
    int efConstruction = 100;
    int ef= 32;

    std::chrono::high_resolution_clock::time_point ts, te;
    float *pdata = (float*)malloc(dim * nb * sizeof(float));
    assert(pdata != nullptr);
    std::string data_location = "/home/zilliz/workspace/data/sift-128-euclidean.hdf5";

    std::cout << "start load sift data ..." << std::endl;
    ts = std::chrono::high_resolution_clock::now();
//    LoadTxtData(data_location, pdata, dim, nb);
    LoadData(data_location, pdata, "train", dim, nb);
    std::cout << "dim: " << dim << ", rows: " << nb << std::endl;
    te = std::chrono::high_resolution_clock::now();
    std::cout << "load data costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;

    std::cout << "the first vector: " << std::endl;
    for (int i = 0; i < dim; ++ i) {
        std::cout << pdata[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "the last vector: " << std::endl;
    for (int i = 0; i < dim; ++ i) {
        std::cout << pdata[dim * (nb - 1) + i] << " ";
    }
    std::cout << std::endl;

    L2Space *space = new L2Space(dim);
    std::shared_ptr<HierarchicalNSW<float>> hnsw_index = std::make_shared<HierarchicalNSW<float>>(space, nb, M, efConstruction);

    uint8_t *p_sq8_data = (uint8_t *) malloc(sizeof(float) * dim * 2 + sizeof(uint8_t) * dim * nb);
    hnsw_index->SetSq8(true);
    ts = std::chrono::high_resolution_clock::now();
    hnsw_index->sq_train(nb, pdata, p_sq8_data);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "sq_train costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;

    // show sq8 codes:
    std::cout << "sq8 codes: " << std::endl;
    for (auto i = 0; i < nb; ++ i) {
        std::cout << i + 1 << "th: ";
        for (auto j = 0; j < dim; ++ j) {
            std::cout << (int)(p_sq8_data[i * dim + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "vmin: " << std::endl;
    for (auto i = 0; i < dim; ++ i) {
        std::cout << ((float*)(p_sq8_data + dim * nb))[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "vdiff: " << std::endl;
    for (auto i = 0; i < dim; ++ i) {
        std::cout << ((float*)(p_sq8_data + dim * nb))[dim + i] << " ";
    }
    std::cout << std::endl;

    ts = std::chrono::high_resolution_clock::now();
    hnsw_index->addPoint((void*)pdata, 0, 0, 0);
//#pragma omp parallel for
    for (int i = 1; i < nb; ++ i) {
        hnsw_index->addPoint((void*)pdata, i, 0, i);
//        hnsw_index->addPoint(pdata + i * dim, i);
    }
    te = std::chrono::high_resolution_clock::now();
    std::cout << "build index costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;

    size_t id_size = sizeof(int) * topk;
    size_t dist_size = sizeof(float) * topk;
    auto p_id = (int*) malloc(id_size * nq);
    auto p_dist = (float*) malloc(dist_size * nq);
    using P = std::pair<float, int64_t>;
    auto compare = [](const P& v1, const P& v2) { return v1.first < v2.first; };
    hnsw_index->setEf(ef);
//    hnsw_index->SetSq8(true);
//    Sq8(pdata, p_sq8_data, dim, nb);

    getchar();
    std::mutex print_lock;
    ts = std::chrono::high_resolution_clock::now();
    int correct_cnt = 0;
//#pragma omp parallel for
    for (int i = 0; i < nq; ++ i) {
        std::vector<P> ret;
        ret = hnsw_index->searchKnn((void*)(pdata + i * dim), topk, compare, nullptr, p_sq8_data);
//        ret = hnsw_index->searchKnn(pdata + i * dim, topk, compare, nullptr);
        {
            std::unique_lock <std::mutex> lock(print_lock);
            std::cout << "the " << i + 1 << "th query result: dist = " << ret[0].first << ", id = " << ret[0].second << std::endl;
            if (i == ret[0].second || ret[0].first < 1e-5)
                correct_cnt ++;
        }
    }
    te = std::chrono::high_resolution_clock::now();
    std::cout << "query " << nq << " times costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
    std::cout << "correct rate: " << correct_cnt << "% " << std::endl;
    free(p_sq8_data);
}
