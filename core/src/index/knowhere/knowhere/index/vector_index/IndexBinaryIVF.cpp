// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/index/vector_index/IndexBinaryIVF.h"

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>

#include <chrono>
#include <string>

#include "knowhere/adapter/VectorAdapter.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"

namespace knowhere {

using stdclock = std::chrono::high_resolution_clock;

BinarySet
BinaryIVF::Serialize() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    std::lock_guard<std::mutex> lk(mutex_);
    return SerializeImpl();
}

void
BinaryIVF::Load(const BinarySet& index_binary) {
    std::lock_guard<std::mutex> lk(mutex_);
    LoadImpl(index_binary);
}

DatasetPtr
BinaryIVF::Search(const DatasetPtr& dataset, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GETBINARYTENSOR(dataset)

    try {
        auto elems = rows * config[meta::TOPK].get<int64_t>();

        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        auto p_id = (int64_t*)malloc(p_id_size);
        auto p_dist = (float*)malloc(p_dist_size);

        search_impl(rows, (uint8_t*)p_data, config[meta::TOPK].get<int64_t>(), p_dist, p_id, config);

        auto ret_ds = std::make_shared<Dataset>();
        if (index_->metric_type == faiss::METRIC_Hamming) {
            auto pf_dist = (float*)malloc(p_dist_size);
            int32_t* pi_dist = (int32_t*)p_dist;
            for (int i = 0; i < elems; i++) {
                *(pf_dist + i) = (float)(*(pi_dist + i));
            }
            ret_ds->Set(meta::IDS, p_id);
            ret_ds->Set(meta::DISTANCE, pf_dist);
            free(p_dist);
        } else {
            ret_ds->Set(meta::IDS, p_id);
            ret_ds->Set(meta::DISTANCE, p_dist);
        }
        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
BinaryIVF::search_impl(int64_t n, const uint8_t* data, int64_t k, float* distances, int64_t* labels,
                       const Config& cfg) {
    auto params = GenParams(cfg);
    auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    ivf_index->nprobe = params->nprobe;
    int32_t* pdistances = (int32_t*)distances;
    stdclock::time_point before = stdclock::now();

    // todo: remove static cast (zhiru)
    static_cast<faiss::IndexBinary*>(index_.get())->search(n, (uint8_t*)data, k, pdistances, labels, bitset_);

    stdclock::time_point after = stdclock::now();
    double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
    KNOWHERE_LOG_DEBUG << "IVF search cost: " << search_cost
                       << ", quantization cost: " << faiss::indexIVF_stats.quantization_time
                       << ", data search cost: " << faiss::indexIVF_stats.search_time;
    faiss::indexIVF_stats.quantization_time = 0;
    faiss::indexIVF_stats.search_time = 0;
}

std::shared_ptr<faiss::IVFSearchParameters>
BinaryIVF::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = config[IndexParams::nprobe];
    // params->max_codes = config["max_code"];
    return params;
}

IndexModelPtr
BinaryIVF::Train(const DatasetPtr& dataset, const Config& config) {
    GETBINARYTENSOR(dataset)
    auto p_ids = dataset->Get<const int64_t*>(meta::IDS);

    faiss::IndexBinary* coarse_quantizer =
        new faiss::IndexBinaryFlat(dim, GetMetricType(config[Metric::TYPE].get<std::string>()));
    auto index = std::make_shared<faiss::IndexBinaryIVF>(coarse_quantizer, dim, config[IndexParams::nlist],
                                                         GetMetricType(config[Metric::TYPE].get<std::string>()));
    index->train(rows, (uint8_t*)p_data);
    index->add_with_ids(rows, (uint8_t*)p_data, p_ids);
    index_ = index;
    return nullptr;
}

int64_t
BinaryIVF::Count() {
    return index_->ntotal;
}

int64_t
BinaryIVF::Dimension() {
    return index_->d;
}

void
BinaryIVF::Add(const DatasetPtr& dataset, const Config& config) {
    KNOWHERE_THROW_MSG("not support yet");
}

void
BinaryIVF::Seal() {
    // do nothing
}

DatasetPtr
BinaryIVF::GetVectorById(const DatasetPtr& dataset, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    //    GETBINARYTENSOR(dataset)
    // auto rows = dataset->Get<int64_t>(meta::ROWS);
    auto p_data = dataset->Get<const int64_t*>(meta::IDS);
    auto elems = dataset->Get<int64_t>(meta::DIM);

    try {
        size_t p_x_size = sizeof(uint8_t) * elems;
        auto p_x = (uint8_t*)malloc(p_x_size);

        index_->get_vector_by_id(1, p_data, p_x, bitset_);

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::TENSOR, p_x);
        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
BinaryIVF::SearchById(const DatasetPtr& dataset, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto rows = dataset->Get<int64_t>(meta::ROWS);
    auto p_data = dataset->Get<const int64_t*>(meta::IDS);

    try {
        auto elems = rows * config[meta::TOPK].get<int64_t>();

        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        auto p_id = (int64_t*)malloc(p_id_size);
        auto p_dist = (float*)malloc(p_dist_size);

        int32_t* pdistances = (int32_t*)p_dist;
        index_->search_by_id(rows, p_data, config[meta::TOPK].get<int64_t>(), pdistances, p_id, bitset_);

        auto ret_ds = std::make_shared<Dataset>();
        if (index_->metric_type == faiss::METRIC_Hamming) {
            auto pf_dist = (float*)malloc(p_dist_size);
            int32_t* pi_dist = (int32_t*)p_dist;
            for (int i = 0; i < elems; i++) {
                *(pf_dist + i) = (float)(*(pi_dist + i));
            }
            ret_ds->Set(meta::IDS, p_id);
            ret_ds->Set(meta::DISTANCE, pf_dist);
            free(p_dist);
        } else {
            ret_ds->Set(meta::IDS, p_id);
            ret_ds->Set(meta::DISTANCE, p_dist);
        }

        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
BinaryIVF::SetBlacklist(faiss::ConcurrentBitsetPtr list) {
    bitset_ = std::move(list);
}

void
BinaryIVF::GetBlacklist(faiss::ConcurrentBitsetPtr& list) {
    list = bitset_;
}

}  // namespace knowhere
