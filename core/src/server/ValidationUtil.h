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

#include "db/Types.h"
#include "db/meta/MetaTypes.h"
#include "utils/Json.h"
#include "utils/Status.h"

#include <string>
#include <vector>

namespace milvus {
namespace server {

constexpr int64_t QUERY_MAX_TOPK = 2048;

extern Status
ValidateCollectionName(const std::string& collection_name);

extern Status
ValidateTableDimension(int64_t dimension, int64_t metric_type);

extern Status
ValidateCollectionIndexType(int32_t index_type);

extern Status
ValidateIndexParams(const milvus::json& index_params, const engine::meta::CollectionSchema& collection_schema,
                    int32_t index_type);

extern Status
ValidateSearchParams(const milvus::json& search_params, const engine::meta::CollectionSchema& collection_schema,
                     int64_t topk);

extern Status
ValidateVectorData(const engine::VectorsData& vectors, const engine::meta::CollectionSchema& collection_schema);

extern Status
ValidateVectorDataSize(const engine::VectorsData& vectors, const engine::meta::CollectionSchema& collection_schema);

extern Status
ValidateCollectionIndexFileSize(int64_t index_file_size);

extern Status
ValidateCollectionIndexMetricType(int32_t metric_type);

extern Status
ValidateSearchTopk(int64_t top_k);

extern Status
ValidatePartitionName(const std::string& partition_name);

extern Status
ValidatePartitionTags(const std::vector<std::string>& partition_tags);
}  // namespace server
}  // namespace milvus
