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

#include "Utils.h"


namespace milvus {
namespace knowhere {

const std::string INDEX_FILE_SLICE_SIZE_IN_MEGABYTE("slice_size");
const std::string INDEX_FILE_SLICE_NUM("index_slice_num");
const std::string INDEX_FILE_LEN("index_slice_len");
const std::string INDEX_DATA_FILE_SLICE_NUM("data_slice_num");
const std::string INDEX_DATA_FILE_LEN("data_slice_len");

const int SLICE_NUM_LEN = 13;
const int INDEX_FILE_LEN_IN_BYTE = 23;

void write_sizet(const std::string &key, const size_t &target, const int len, BinarySet &res_set) {
    uint8_t* target_p = (uint8_t*)malloc(len);
    auto ok = snprintf((char*)target_p, len, "%zu", target);
    if (ok < 0 || ok >= len) {
        KNOHWERE_ERROR_MSG("error occurs when write the meta info " + key + " in Serialize!");
    }
    std::shared_ptr<uint8_t[]> target_sp(target_p, std::default_delete<uint8_t[]>());
    res_set.Append(key, target_sp, ok + 1);
}

void read_sizet(const std::string &key, const BinarySet &data_src, size_t &res) {
    auto index_len_bin = data_src.GetByName(key);
    auto ok = sscanf((char*)(index_len_bin->data.get()), "%zu", &res);
    if (ok < 0) {
        KNOHWERE_ERROR_MSG("error occurs when read the meta info " + key + " in Load!");
    }
}

void Assemble(const BinarySet& index_binary, const std::string &prefix, MemoryIOReader &reader) {
    size_t index_len = 0;
    read_sizet(prefix + "_" + INDEX_FILE_LEN, index_binary, index_len);

    size_t slice_num = 0;
    read_sizet(prefix + "_" + INDEX_FILE_SLICE_NUM, index_binary, slice_num);

    reader.total = index_len + 1;
    reader.data_ = (uint8_t*) malloc(reader.total);
    reader.data_[index_len] = 0;
    int64_t pos = 0;
    for (auto i = 0; i < slice_num; ++ i) {
        auto slice_i_sp = index_binary.GetByName(prefix + "_" + std::to_string(i));
        memcpy(reader.data_ + pos, slice_i_sp->data.get(), (size_t)(slice_i_sp->size));
        pos += slice_i_sp->size;
    }
}

void Assemble(const BinarySet& index_binary, const std::string &prefix, uint8_t *p_data, int64_t &data_len) {
    size_t index_len = 0;
    read_sizet(prefix + "_" + INDEX_FILE_LEN, index_binary, index_len);

    size_t slice_num = 0;
    read_sizet(prefix + "_" + INDEX_FILE_SLICE_NUM, index_binary, slice_num);

    data_len = index_len + 1;
    p_data = (uint8_t*) malloc(index_len + 1);
    p_data[index_len] = 0;
    int64_t pos = 0;
    for (auto i = 0; i < slice_num; ++ i) {
        auto slice_i_sp = index_binary.GetByName(prefix + "_" + std::to_string(i));
        memcpy(p_data + pos, slice_i_sp->data.get(), (size_t)(slice_i_sp->size));
        pos += slice_i_sp->size;
    }
}

void Disassemble(const int64_t &slice_size_in_byte, const MemoryIOWriter &writer, const std::string &prefix, BinarySet &res_set) {
    size_t slice_idx = 0;
    for (int64_t i = 0; i < writer.rp; ++ slice_idx) {
        int64_t ri = std::min(i + slice_size_in_byte, (int64_t)writer.rp);
        uint8_t* slice_i = (uint8_t*) malloc((size_t)(ri - i));
        memcpy(slice_i, writer.data_ + i, (size_t)(ri - i));
        std::shared_ptr<uint8_t[]> slice_i_sp(slice_i, std::default_delete<uint8_t[]>());
        res_set.Append(prefix + "_" + std::to_string(slice_idx), slice_i_sp, ri - i);
        i = ri;
    }

    write_sizet(prefix + "_" + INDEX_FILE_SLICE_NUM, slice_idx, SLICE_NUM_LEN, res_set);
    write_sizet(prefix + "_" + INDEX_FILE_LEN, writer.rp + 1, INDEX_FILE_LEN_IN_BYTE, res_set);
}

void Disassemble(const int64_t &slice_size_in_byte, const std::string &prefix, const uint8_t *p_data, const int64_t &data_len, BinarySet &res_set) {
    size_t slice_idx = 0;
    for (int64_t i = 0; i < data_len; ++ slice_idx) {
        int64_t ri = std::min(i + slice_size_in_byte, (int64_t)data_len);
        uint8_t* slice_i = (uint8_t*) malloc((size_t)(ri - i));
        memcpy(slice_i, p_data + i, (size_t)(ri - i));
        std::shared_ptr<uint8_t[]> slice_i_sp(slice_i, std::default_delete<uint8_t[]>());
        res_set.Append(prefix + "_" + std::to_string(slice_idx), slice_i_sp, ri - i);
        i = ri;
    }

    write_sizet(prefix + "_" + INDEX_FILE_SLICE_NUM, slice_idx, SLICE_NUM_LEN, res_set);
    write_sizet(prefix + "_" + INDEX_FILE_LEN, data_len + 1, INDEX_FILE_LEN_IN_BYTE, res_set);
}

}  // namespace knowhere
}  // namespace milvus
