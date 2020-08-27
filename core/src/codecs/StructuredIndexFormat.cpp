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

#include "codecs/StructuredIndexFormat.h"

#include <unistd.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <memory>
#include <unordered_map>

#include "db/Types.h"
#include "db/Utils.h"
#include "knowhere/index/structured_index/StructuredIndexSort.h"

#include "storage/ExtraFileInfo.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace codec {

const char* STRUCTURED_INDEX_POSTFIX = ".ind";

std::string
StructuredIndexFormat::FilePostfix() {
    std::string str = STRUCTURED_INDEX_POSTFIX;
    return str;
}

knowhere::IndexPtr
StructuredIndexFormat::CreateStructuredIndex(const engine::DataType data_type) {
    knowhere::IndexPtr index = nullptr;
    switch (data_type) {
        case engine::DataType::INT8: {
            index = std::make_shared<knowhere::StructuredIndexSort<int8_t>>();
            break;
        }
        case engine::DataType::INT16: {
            index = std::make_shared<knowhere::StructuredIndexSort<int16_t>>();
            break;
        }
        case engine::DataType::INT32: {
            index = std::make_shared<knowhere::StructuredIndexSort<int32_t>>();
            break;
        }
        case engine::DataType::INT64: {
            index = std::make_shared<knowhere::StructuredIndexSort<int64_t>>();
            break;
        }
        case engine::DataType::FLOAT: {
            index = std::make_shared<knowhere::StructuredIndexSort<float>>();
            break;
        }
        case engine::DataType::DOUBLE: {
            index = std::make_shared<knowhere::StructuredIndexSort<double>>();
            break;
        }
        default: {
            LOG_ENGINE_ERROR_ << "Invalid field type";
            return nullptr;
        }
    }
    return index;
}

Status
StructuredIndexFormat::Read(const milvus::storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                            knowhere::IndexPtr& index) {
    milvus::TimeRecorder recorder("StructuredIndexFormat::Read");
    knowhere::BinarySet load_data_list;

    std::string full_file_path = file_path + STRUCTURED_INDEX_POSTFIX;
    CHECK_MAGIC_VALID(fs_ptr, full_file_path);
    CHECK_SUM_VALID(fs_ptr, full_file_path);
    if (!fs_ptr->reader_ptr_->Open(full_file_path)) {
        return Status(SERVER_CANNOT_OPEN_FILE, "Fail to open structured index: " + full_file_path);
    }
    int64_t length = fs_ptr->reader_ptr_->Length() - SUM_SIZE;
    if (length <= 0) {
        return Status(SERVER_UNEXPECTED_ERROR, "Invalid structured index length: " + full_file_path);
    }

    size_t rp = MAGIC_SIZE + HEADER_SIZE;
    fs_ptr->reader_ptr_->Seekg(rp);

    int32_t data_type = 0;
    fs_ptr->reader_ptr_->Read(&data_type, sizeof(data_type));
    rp += sizeof(data_type) + MAGIC_SIZE + HEADER_SIZE;
    fs_ptr->reader_ptr_->Seekg(rp);

    LOG_ENGINE_DEBUG_ << "Start to read_index(" << full_file_path << ") length: " << length << " bytes";
    while (rp < length) {
        size_t meta_length;
        fs_ptr->reader_ptr_->Read(&meta_length, sizeof(meta_length));
        rp += sizeof(meta_length);
        fs_ptr->reader_ptr_->Seekg(rp);

        auto meta = new char[meta_length];
        fs_ptr->reader_ptr_->Read(meta, meta_length);
        rp += meta_length;
        fs_ptr->reader_ptr_->Seekg(rp);

        size_t bin_length;
        fs_ptr->reader_ptr_->Read(&bin_length, sizeof(bin_length));
        rp += sizeof(bin_length);
        fs_ptr->reader_ptr_->Seekg(rp);

        auto bin = new uint8_t[bin_length];
        fs_ptr->reader_ptr_->Read(bin, bin_length);
        rp += bin_length;
        fs_ptr->reader_ptr_->Seekg(rp);

        std::shared_ptr<uint8_t[]> binptr(bin);
        load_data_list.Append(std::string(meta, meta_length), binptr, bin_length);
        delete[] meta;
    }
    fs_ptr->reader_ptr_->Close();

    double span = recorder.RecordSection("End");
    double rate = length * 1000000.0 / span / 1024 / 1024;
    LOG_ENGINE_DEBUG_ << "StructuredIndexFormat::read(" << full_file_path << ") rate " << rate << "MB/s";

    auto attr_type = static_cast<engine::DataType>(data_type);
    index = CreateStructuredIndex(attr_type);
    index->Load(load_data_list);

    return Status::OK();
}

Status
StructuredIndexFormat::Write(const milvus::storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                             engine::DataType data_type, const knowhere::IndexPtr& index) {
    milvus::TimeRecorder recorder("StructuredIndexFormat::Write");

    std::string full_file_path = file_path + STRUCTURED_INDEX_POSTFIX;
    // TODO: add extra info
    std::unordered_map<std::string, std::string> maps;
    WRITE_MAGIC(fs_ptr, full_file_path);
    WRITE_HEADER(fs_ptr, full_file_path, maps);

    auto binaryset = index->Serialize(knowhere::Config());

    if (!fs_ptr->writer_ptr_->InOpen(full_file_path)) {
        return Status(SERVER_CANNOT_OPEN_FILE, "Fail to open structured index: " + full_file_path);
    }

    try {
        fs_ptr->writer_ptr_->Seekp(MAGIC_SIZE + HEADER_SIZE);
        fs_ptr->writer_ptr_->Write(&data_type, sizeof(data_type));

        for (auto& iter : binaryset.binary_map_) {
            auto meta = iter.first.c_str();
            size_t meta_length = iter.first.length();
            fs_ptr->writer_ptr_->Write(&meta_length, sizeof(meta_length));
            fs_ptr->writer_ptr_->Write(meta, meta_length);

            auto binary = iter.second;
            int64_t binary_length = binary->size;
            fs_ptr->writer_ptr_->Write(&binary_length, sizeof(binary_length));
            fs_ptr->writer_ptr_->Write(binary->data.get(), binary_length);
        }

        fs_ptr->writer_ptr_->Close();
        WRITE_SUM(fs_ptr, full_file_path);

        double span = recorder.RecordSection("End");
        double rate = fs_ptr->writer_ptr_->Length() * 1000000.0 / span / 1024 / 1024;
        LOG_ENGINE_DEBUG_ << "StructuredIndexFormat::write(" << full_file_path << ") rate " << rate << "MB/s";
    } catch (std::exception& ex) {
        std::string err_msg = "Failed to write structured index: " + std::string(ex.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }

    return Status::OK();
}

}  // namespace codec
}  // namespace milvus
