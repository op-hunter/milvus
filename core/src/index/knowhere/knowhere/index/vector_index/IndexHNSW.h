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

#pragma once

#include <memory>
#include <mutex>
#include <src/metrics/SystemInfo.h>
#include <src/cache/CpuCacheMgr.h>

#include "hnswlib/hnswlib.h"

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/VecIndex.h"

namespace milvus {
namespace knowhere {

class IndexHNSW : public VecIndex {
 public:
    IndexHNSW() {
        index_type_ = IndexEnum::INDEX_HNSW;
        printf("in IndexHNSW non-parameter constructor, current memory info:\n");
        auto used_memory = server::SystemInfo::GetInstance().GetProcessUsedMemory();
        printf("current process cost memory: %ld B, %.2f MB, %.2f GB.\n", used_memory, (double)used_memory/1024/1024, (double)used_memory/1024/1024/1024);
        cache::CpuCacheMgr::GetInstance()->PrintInfo();
        printf("------------------------gorgeous dividing line------------------------------\n");
//        printf("current Index size = %ld bytes, %ld MB\n", Size(), Size()/1024/1024);
    }

    ~IndexHNSW() {
        printf("IndexHNSW deconstructed.\n");
        printf("current memory info:\n");
        auto used_memory = server::SystemInfo::GetInstance().GetProcessUsedMemory();
        printf("current process cost memory: %ld B, %.2f MB, %.2f GB.\n", used_memory, (double)used_memory/1024/1024, (double)used_memory/1024/1024/1024);
        cache::CpuCacheMgr::GetInstance()->PrintInfo();
//        printf("current Index size = %ld bytes, %ld MB\n", Size(), Size()/1024/1024);
        printf("------------------------gorgeous dividing line------------------------------\n");
    }

    BinarySet
    Serialize(const Config& config = Config()) override;

    void
    Load(const BinarySet& index_binary) override;

    void
    Train(const DatasetPtr& dataset_ptr, const Config& config) override;

    void
    Add(const DatasetPtr& dataset_ptr, const Config& config) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override {
        KNOWHERE_THROW_MSG("Incremental index is not supported");
    }

    DatasetPtr
    Query(const DatasetPtr& dataset_ptr, const Config& config) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

 private:
    bool normalize = false;
    std::mutex mutex_;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

}  // namespace knowhere
}  // namespace milvus
