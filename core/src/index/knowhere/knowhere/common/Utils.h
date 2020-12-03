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
#include "knowhere/index/vector_index/helpers/FaissIO.h"
#include "BinarySet.h"
#include "Config.h"
#include "Exception.h"

namespace milvus {
namespace knowhere {

extern const std::string INDEX_FILE_SLICE_SIZE_IN_MEGABYTE;
extern const std::string INDEX_FILE_SLICE_NUM;
extern const std::string INDEX_FILE_LEN;
extern const std::string INDEX_DATA_FILE_SLICE_NUM;
extern const std::string INDEX_DATA_FILE_LEN;

void Assemble(const BinarySet& index_binary, const std::string &prefix, MemoryIOReader &reader);
void Assemble(const BinarySet& index_binary, const std::string &prefix, uint8_t *p_data, int64_t &data_len);

void Disassemble(const int64_t &slice_size_in_byte, const MemoryIOWriter &writer, const std::string &prefix, BinarySet &bs);
void Disassemble(const int64_t &slice_size_in_byte, const std::string &prefix, const uint8_t *p_data, const int64_t &data_len, BinarySet &bs);

}  // namespace knowhere
}  // namespace milvus
