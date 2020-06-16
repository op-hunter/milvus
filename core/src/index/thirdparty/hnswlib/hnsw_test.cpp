#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "/usr/include/hdf5/serial/hdf5.h"
#include "/usr/include/hdf5/serial/H5Cpp.h"
#include "hnswalg2.h"
//#include "hnswalg.h"

using namespace hnswlib;
using namespace H5;

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

void LoadTxtData(const std::string file_location, float *&data, int &dim, int &num_vets) {
    std::ifstream in(file_location.c_str(), std::ios::in);
    for (auto i = 0; i < num_vets; ++ i) {
        for (auto j = 0; j < dim; ++ j) {
            in >> *(data + i * dim + j);
        }
    }
    in.close();
}

int main() {
    int dim = 128;
    int nb = 1000000;
    int nq = 100;
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

    ts = std::chrono::high_resolution_clock::now();
    hnsw_index->addPoint(pdata, 0);
#pragma omp parallel for
    for (int i = 1; i < nb; ++ i) {
        hnsw_index->addPoint(pdata, i);
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

    std::mutex print_lock;
    ts = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for
    for (int i = 0; i < nq; ++ i) {
        std::vector<P> ret;
        ret = hnsw_index->searchKnn((void*)(pdata + i * dim), topk, compare, nullptr, pdata);
//        ret = hnsw_index->searchKnn(pdata + i * dim, topk, compare, nullptr);
        {
            std::unique_lock <std::mutex> lock(print_lock);
            std::cout << "the " << i + 1 << "th query result: dist = " << ret[0].first << ", id = " << ret[0].second << std::endl;
        }
    }
    te = std::chrono::high_resolution_clock::now();
    std::cout << "query " << nq << " times costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count() << "ms " << std::endl;
}
