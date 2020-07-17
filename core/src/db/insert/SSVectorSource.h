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
#include <string>
#include <unordered_map>
#include <vector>

#include "db/IDGenerator.h"
#include "db/engine/ExecutionEngine.h"
#include "db/insert/SSMemManager.h"
#include "segment/SSSegmentWriter.h"
#include "segment/Segment.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {

// TODO(zhiru): this class needs to be refactored once attributes are added

class SSVectorSource {
 public:
    explicit SSVectorSource(const DataChunkPtr& chunk);

    Status
    Add(const segment::SSSegmentWriterPtr& segment_writer_ptr, const int64_t& num_attrs_to_add,
        int64_t& num_attrs_added);

    bool
    AllAdded();

 private:
    DataChunkPtr chunk_;

    int64_t current_num_added_ = 0;
};  // SSVectorSource

using SSVectorSourcePtr = std::shared_ptr<SSVectorSource>;

}  // namespace engine
}  // namespace milvus
