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
#include <vector>

#include "knowhere/common/Config.h"
#include "knowhere/common/Dataset.h"
#include "knowhere/index/Index.h"
#include "knowhere/index/preprocessor/Preprocessor.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "segment/Types.h"

namespace knowhere {

class VectorIndex;
using VectorIndexPtr = std::shared_ptr<VectorIndex>;

class VectorIndex : public Index {
 public:
    virtual PreprocessorPtr
    BuildPreprocessor(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }

    virtual IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }

    virtual DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }

    virtual DatasetPtr
    SearchById(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }

    virtual void
    Add(const DatasetPtr& dataset, const Config& config) = 0;

    virtual void
    Seal() = 0;

    // TODO(linxj): Deprecated
    //    virtual VectorIndexPtr
    //    Clone() = 0;

    virtual int64_t
    Count() = 0;

    virtual int64_t
    Dimension() = 0;

    virtual const std::vector<milvus::segment::doc_id_t>&
    GetUids() const {
        return uids_;
    }

    virtual void
    SetUids(std::vector<milvus::segment::doc_id_t>& uids) {
        uids_.clear();
        uids_.swap(uids);
    }

 private:
    std::vector<milvus::segment::doc_id_t> uids_;
};

}  // namespace knowhere
