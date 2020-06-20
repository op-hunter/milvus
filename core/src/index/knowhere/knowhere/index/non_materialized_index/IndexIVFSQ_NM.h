// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <memory>
#include <utility>

#include "knowhere/index/non_materialized_index/IndexIVF_NM.h"

namespace milvus {
namespace knowhere {

class IVFSQ_NM : public IVF_NM {
 public:
    IVFSQ_NM() : IVF_NM() {
        index_type_ = IndexEnum::INDEX_FAISS_IVFSQ8;
    }

    explicit IVFSQ_NM(std::shared_ptr<faiss::Index> index) : IVF_NM(std::move(index)) {
        index_type_ = IndexEnum::INDEX_FAISS_IVFSQ8;
    }

    void
    Load(const BinarySet&, const void*, size_t) override;

    void
    Train(const DatasetPtr&, const Config&) override;

    VecIndexPtr
    CopyCpuToGpu(const int64_t, const Config&) override;
};

using IVFSQNMPtr = std::shared_ptr<IVFSQ_NM>;

}  // namespace knowhere
}  // namespace milvus
