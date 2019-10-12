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

#include "ExecutionEngine.h"
#include "src/wrapper/VecIndex.h"

#include <memory>
#include <string>

namespace milvus {
namespace engine {

class ExecutionEngineImpl : public ExecutionEngine {
 public:
    ExecutionEngineImpl(uint16_t dimension, const std::string& location, EngineType index_type, MetricType metric_type,
                        int32_t nlist);

    ExecutionEngineImpl(VecIndexPtr index, const std::string& location, EngineType index_type, MetricType metric_type,
                        int32_t nlist);

    Status
    AddWithIds(int64_t n, const float* xdata, const int64_t* xids) override;

    size_t
    Count() const override;

    size_t
    Size() const override;

    size_t
    Dimension() const override;

    size_t
    PhysicalSize() const override;

    Status
    Serialize() override;

    Status
    Load(bool to_cache) override;

    Status
    CopyToGpu(uint64_t device_id, bool hybrid = false) override;

    Status
    CopyToIndexFileToGpu(uint64_t device_id) override;

    Status
    CopyToCpu() override;

    ExecutionEnginePtr
    Clone() override;

    Status
    Merge(const std::string& location) override;

    Status
    Search(int64_t n,
           const float* data,
           int64_t k,
           int64_t nprobe,
           float* distances,
           int64_t* labels,
           bool hybrid = false) const override;

    ExecutionEnginePtr
    BuildIndex(const std::string& location, EngineType engine_type) override;

    Status
    Cache() override;

    Status
    GpuCache(uint64_t gpu_id) override;

    Status
    Init() override;

    EngineType
    IndexEngineType() const override {
        return index_type_;
    }

    MetricType
    IndexMetricType() const override {
        return metric_type_;
    }

    std::string
    GetLocation() const override {
        return location_;
    }

 private:
    VecIndexPtr
    CreatetVecIndex(EngineType type);

    VecIndexPtr
    Load(const std::string& location);

    void
    HybridLoad() const;

    void
    HybridUnset() const;

 protected:
    VecIndexPtr index_ = nullptr;
    EngineType index_type_;
    MetricType metric_type_;

    int64_t dim_;
    std::string location_;

    int32_t nlist_ = 0;
    int32_t gpu_num_ = 0;
};

}  // namespace engine
}  // namespace milvus
