// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.


#pragma once

#include <memory>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include "wrapper/VecIndex.h"
#include "wrapper/utils.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

class DataGenBase;

using DataGenPtr = std::shared_ptr<DataGenBase>;

constexpr int64_t DIM = 128;
constexpr int64_t NB = 100000;
constexpr int64_t NQ = 10;
constexpr int64_t DEVICEID = 0;
constexpr int64_t PINMEM = 1024 * 1024 * 200;
constexpr int64_t TEMPMEM = 1024 * 1024 * 300;
constexpr int64_t RESNUM = 2;

class DataGenBase {
 public:
    virtual void GenData(const int& dim, const int& nb, const int& nq, float* xb, float* xq, int64_t* ids,
                         const int& k, int64_t* gt_ids, float* gt_dis);

    virtual void GenData(const int& dim,
                         const int& nb,
                         const int& nq,
                         std::vector<float>& xb,
                         std::vector<float>& xq,
                         std::vector<int64_t>& ids,
                         const int& k,
                         std::vector<int64_t>& gt_ids,
                         std::vector<float>& gt_dis);

    void AssertResult(const std::vector<int64_t>& ids, const std::vector<float>& dis);

    int dim = DIM;
    int nb = NB;
    int nq = NQ;
    int k = 10;
    std::vector<float> xb;
    std::vector<float> xq;
    std::vector<int64_t> ids;

    // Ground Truth
    std::vector<int64_t> gt_ids;
    std::vector<float> gt_dis;
};

namespace {

namespace ms = milvus::engine;
namespace kw = knowhere;

} // namespace

class ParamGenerator {
 public:
    static ParamGenerator& GetInstance() {
        static ParamGenerator instance;
        return instance;
    }

    knowhere::Config Gen(const milvus::engine::IndexType& type) {
        switch (type) {
            case milvus::engine::IndexType::FAISS_IDMAP: {
                auto tempconf = std::make_shared<knowhere::Cfg>();
                tempconf->metric_type = knowhere::METRICTYPE::L2;
                return tempconf;
            }
            case milvus::engine::IndexType::FAISS_IVFFLAT_CPU:
            case milvus::engine::IndexType::FAISS_IVFFLAT_GPU:
            case milvus::engine::IndexType::FAISS_IVFFLAT_MIX: {
                auto tempconf = std::make_shared<knowhere::IVFCfg>();
                tempconf->nlist = 100;
                tempconf->nprobe = 16;
                tempconf->metric_type = knowhere::METRICTYPE::L2;
                return tempconf;
            }
            case milvus::engine::IndexType::FAISS_IVFSQ8_HYBRID:
            case milvus::engine::IndexType::FAISS_IVFSQ8_CPU:
            case milvus::engine::IndexType::FAISS_IVFSQ8_GPU:
            case milvus::engine::IndexType::FAISS_IVFSQ8_MIX: {
                auto tempconf = std::make_shared<knowhere::IVFSQCfg>();
                tempconf->nlist = 100;
                tempconf->nprobe = 16;
                tempconf->nbits = 8;
                tempconf->metric_type = knowhere::METRICTYPE::L2;
                return tempconf;
            }
            case milvus::engine::IndexType::FAISS_IVFPQ_CPU:
            case milvus::engine::IndexType::FAISS_IVFPQ_GPU: {
                auto tempconf = std::make_shared<knowhere::IVFPQCfg>();
                tempconf->nlist = 100;
                tempconf->nprobe = 16;
                tempconf->nbits = 8;
                tempconf->m = 8;
                tempconf->metric_type = knowhere::METRICTYPE::L2;
                return tempconf;
            }
            case milvus::engine::IndexType::NSG_MIX: {
                auto tempconf = std::make_shared<knowhere::NSGCfg>();
                tempconf->nlist = 100;
                tempconf->nprobe = 16;
                tempconf->search_length = 8;
                tempconf->knng = 200;
                tempconf->search_length = 40; // TODO(linxj): be 20 when search
                tempconf->out_degree = 60;
                tempconf->candidate_pool_size = 200;
                tempconf->metric_type = knowhere::METRICTYPE::L2;
                return tempconf;
            }
        }
    }
};


//class SanityCheck : public DataGenBase {
// public:
//    void GenData(const int &dim, const int &nb, const int &nq, float *xb, float *xq, long *ids,
//                 const int &k, long *gt_ids, float *gt_dis) override;
//};

