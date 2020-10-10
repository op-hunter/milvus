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

#include "codecs/BlockFormat.h"

#include <unistd.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <memory>
#include <utility>

#include "codecs/ExtraFileInfo.h"
#include "db/Utils.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace codec {

Status
BlockFormat::Read(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path, engine::BinaryDataPtr& raw) {
    milvus::TimeRecorderAuto recorder("BlockFormat::Read:" + file_path);
    if (!fs_ptr->reader_ptr_->Open(file_path)) {
        return Status(SERVER_CANNOT_OPEN_FILE, "Fail to open file: " + file_path);
    }
    CHECK_MAGIC_VALID(fs_ptr);
    std::vector<char> header;
    header.resize(HEADER_SIZE);
    fs_ptr->reader_ptr_->Read(header.data(), HEADER_SIZE);

    HeaderMap map = TransformHeaderData(header);
    size_t num_bytes = stol(map.at("size"));

    raw = std::make_shared<engine::BinaryData>();
    raw->data_.resize(num_bytes);
    fs_ptr->reader_ptr_->Seekg(MAGIC_SIZE + HEADER_SIZE);
    fs_ptr->reader_ptr_->Read(raw->data_.data(), num_bytes);

    uint32_t record;
    fs_ptr->reader_ptr_->Read(&record, SUM_SIZE);
    fs_ptr->reader_ptr_->Close();

    CHECK_SUM_VALID(header.data(), reinterpret_cast<const char*>(raw->data_.data()), num_bytes, record);

    return Status::OK();
}

Status
BlockFormat::Read(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path, int64_t offset, int64_t num_bytes,
                  engine::BinaryDataPtr& raw) {
    if (offset < 0 || num_bytes <= 0) {
        return Status(SERVER_INVALID_ARGUMENT, "Invalid input to read: " + file_path);
    }

    if (!fs_ptr->reader_ptr_->Open(file_path)) {
        return Status(SERVER_CANNOT_OPEN_FILE, "Fail to open file: " + file_path);
    }
    CHECK_MAGIC_VALID(fs_ptr);
    std::vector<char> header;
    header.resize(HEADER_SIZE);
    fs_ptr->reader_ptr_->Read(header.data(), HEADER_SIZE);

    HeaderMap map = TransformHeaderData(header);
    size_t total_num_bytes = stol(map.at("size"));

    if (offset + num_bytes > total_num_bytes) {
        return Status(SERVER_INVALID_ARGUMENT, "Invalid argument to read: " + file_path);
    }

    raw = std::make_shared<engine::BinaryData>();
    raw->data_.resize(num_bytes);

    fs_ptr->reader_ptr_->Seekg(offset + MAGIC_SIZE + HEADER_SIZE);
    fs_ptr->reader_ptr_->Read(raw->data_.data(), num_bytes);

    uint32_t record;
    fs_ptr->reader_ptr_->Read(&record, SUM_SIZE);
    fs_ptr->reader_ptr_->Close();

    CHECK_SUM_VALID(header.data(), reinterpret_cast<const char*>(raw->data_.data()), num_bytes, record);

    return Status::OK();
}

Status
BlockFormat::Read(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path, const ReadRanges& read_ranges,
                  engine::BinaryDataPtr& raw) {
    if (read_ranges.empty()) {
        return Status::OK();
    }

    if (!fs_ptr->reader_ptr_->Open(file_path)) {
        return Status(SERVER_CANNOT_OPEN_FILE, "Fail to open file: " + file_path);
    }
    CHECK_MAGIC_VALID(fs_ptr);

    std::vector<char> header;
    header.resize(HEADER_SIZE);
    fs_ptr->reader_ptr_->Read(header.data(), HEADER_SIZE);

    HeaderMap map = TransformHeaderData(header);
    size_t total_num_bytes = stol(map.at("size"));

    std::vector<char> data;
    data.resize(total_num_bytes);

    fs_ptr->reader_ptr_->Read(data.data(), total_num_bytes);
    uint32_t record;
    fs_ptr->reader_ptr_->Read(&record, SUM_SIZE);
    fs_ptr->reader_ptr_->Close();

    CHECK_SUM_VALID(header.data(), reinterpret_cast<const char*>(data.data()), total_num_bytes, record);

    int64_t total_bytes = 0;
    for (auto& range : read_ranges) {
        if (range.offset_ > total_num_bytes) {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid argument to read: " + file_path);
        }
        total_bytes += range.num_bytes_;
    }

    raw = std::make_shared<engine::BinaryData>();
    raw->data_.resize(total_bytes);
    int64_t poz = 0;
    for (auto& range : read_ranges) {
        int64_t offset = range.offset_;
        memcpy(raw->data_.data() + poz, data.data() + offset, range.num_bytes_);
        poz += range.num_bytes_;
    }

    return Status::OK();
}

Status
BlockFormat::Write(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                   const engine::BinaryDataPtr& raw) {
    if (raw == nullptr) {
        return Status::OK();
    }

    milvus::TimeRecorderAuto recorder("BlockFormat::Write:" + file_path);

    if (!fs_ptr->writer_ptr_->Open(file_path)) {
        return Status(SERVER_CANNOT_CREATE_FILE, "Fail to open file: " + file_path);
    }

    try {
        // TODO: add extra info
        WRITE_MAGIC(fs_ptr);

        size_t num_bytes = raw->data_.size();

        HeaderMap maps;
        maps.insert(std::make_pair("size", std::to_string(num_bytes)));
        std::string header = HeaderWrapper(maps);
        WRITE_HEADER(fs_ptr, header);

        fs_ptr->writer_ptr_->Write(raw->data_.data(), num_bytes);

        WRITE_SUM(fs_ptr, header, reinterpret_cast<char*>(raw->data_.data()), num_bytes);

        fs_ptr->writer_ptr_->Close();
    } catch (std::exception& ex) {
        std::string err_msg = "Failed to write block data: " + std::string(ex.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }

    return Status::OK();
}

}  // namespace codec
}  // namespace milvus
