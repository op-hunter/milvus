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

#include <string>
#include "knowhere/common/Dataset.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

#define GETTENSOR(dataset)                         \
    auto dim = dataset->Get<int64_t>(meta::DIM);   \
    auto rows = dataset->Get<int64_t>(meta::ROWS); \
    auto p_data = dataset->Get<const float*>(meta::TENSOR);

#define GETBINARYTENSOR(dataset)                   \
    auto dim = dataset->Get<int64_t>(meta::DIM);   \
    auto rows = dataset->Get<int64_t>(meta::ROWS); \
    auto p_data = dataset->Get<const uint8_t*>(meta::TENSOR);

}  // namespace knowhere
