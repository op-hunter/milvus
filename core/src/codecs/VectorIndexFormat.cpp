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

#include <boost/filesystem.hpp>
#include <memory>

#include "codecs/Codec.h"
#include "codecs/VectorIndexFormat.h"
#include "knowhere/common/BinarySet.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/VecIndexFactory.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace codec {

const char* VECTOR_INDEX_POSTFIX = ".idx";

std::string
VectorIndexFormat::FilePostfix() {
    std::string str = VECTOR_INDEX_POSTFIX;
    return str;
}

void
VectorIndexFormat::ReadRaw(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                           knowhere::BinaryPtr& data) {
    milvus::TimeRecorder recorder("VectorIndexFormat::ReadRaw");

    if (!fs_ptr->reader_ptr_->Open(file_path)) {
        THROW_ERROR(SERVER_CANNOT_OPEN_FILE, "Fail to open raw file: " + file_path);
    }

    size_t num_bytes;
    fs_ptr->reader_ptr_->Read(&num_bytes, sizeof(size_t));

    data = std::make_shared<knowhere::Binary>();
    data->size = num_bytes;
    data->data = std::shared_ptr<uint8_t[]>(new uint8_t[num_bytes]);

    // Beginning of file is num_bytes
    fs_ptr->reader_ptr_->Seekg(sizeof(size_t));
    fs_ptr->reader_ptr_->Read(data->data.get(), num_bytes);
    fs_ptr->reader_ptr_->Close();

    double span = recorder.RecordSection("End");
    double rate = num_bytes * 1000000.0 / span / 1024 / 1024;
    LOG_ENGINE_DEBUG_ << "VectorIndexFormat::ReadIndex(" << file_path << ") rate " << rate << "MB/s";
}

void
VectorIndexFormat::ReadIndex(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                             knowhere::BinarySet& data) {
    milvus::TimeRecorder recorder("VectorIndexFormat::ReadIndex");

    std::string full_file_path = file_path + VECTOR_INDEX_POSTFIX;
    if (!fs_ptr->reader_ptr_->Open(full_file_path)) {
        THROW_ERROR(SERVER_CANNOT_OPEN_FILE, "Fail to open vector index: " + full_file_path);
    }

    int64_t length = fs_ptr->reader_ptr_->Length();
    if (length <= 0) {
        THROW_ERROR(SERVER_UNEXPECTED_ERROR, "Invalid vector index length: " + full_file_path);
    }

    int64_t rp = 0;
    fs_ptr->reader_ptr_->Seekg(0);

    LOG_ENGINE_DEBUG_ << "Start to ReadIndex(" << full_file_path << ") length: " << length << " bytes";
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
        data.Append(std::string(meta, meta_length), binptr, bin_length);
        delete[] meta;
    }
    fs_ptr->reader_ptr_->Close();

    double span = recorder.RecordSection("End");
    double rate = length * 1000000.0 / span / 1024 / 1024;
    LOG_ENGINE_DEBUG_ << "VectorIndexFormat::ReadIndex(" << full_file_path << ") rate " << rate << "MB/s";
}

void
VectorIndexFormat::ReadCompress(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                                knowhere::BinaryPtr& data) {
    auto& ss_codec = codec::Codec::instance();
    ss_codec.GetVectorCompressFormat()->Read(fs_ptr, file_path, data);
}

void
VectorIndexFormat::ConvertRaw(const engine::BinaryDataPtr& raw, knowhere::BinaryPtr& data) {
    data = std::make_shared<knowhere::Binary>();
    if (raw == nullptr) {
        return;
    }

    data->size = raw->Size();
    data->data = std::shared_ptr<uint8_t[]>(new uint8_t[data->size], std::default_delete<uint8_t[]>());
    memcpy(data->data.get(), raw->data_.data(), data->size);
}

void
VectorIndexFormat::ConstructIndex(const std::string& index_name, knowhere::BinarySet& index_data,
                                  knowhere::BinaryPtr& raw_data, knowhere::BinaryPtr& compress_data,
                                  knowhere::VecIndexPtr& index) {
    knowhere::VecIndexFactory& vec_index_factory = knowhere::VecIndexFactory::GetInstance();
    index = vec_index_factory.CreateVecIndex(index_name, knowhere::IndexMode::MODE_CPU);
    if (index != nullptr) {
        int64_t length = 0;
        for (auto& pair : index_data.binary_map_) {
            length += pair.second->size;
        }

        if (raw_data != nullptr) {
            LOG_ENGINE_DEBUG_ << "load index with " << RAW_DATA << " " << raw_data->size;
            index_data.Append(RAW_DATA, raw_data);
            length += raw_data->size;
        }

        if (compress_data != nullptr) {
            LOG_ENGINE_DEBUG_ << "load index with " << SQ8_DATA << " " << compress_data->size;
            index_data.Append(SQ8_DATA, compress_data);
            length += compress_data->size;
        }

        index->Load(index_data);
        index->UpdateIndexSize();
        LOG_ENGINE_DEBUG_ << "index file size " << length << " index size " << index->IndexSize();
    } else {
        THROW_ERROR(SERVER_UNEXPECTED_ERROR, "Fail to create vector index");
    }
}

void
VectorIndexFormat::WriteIndex(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                              const knowhere::VecIndexPtr& index) {
    milvus::TimeRecorder recorder("SVectorIndexFormat::WriteIndex");

    std::string full_file_path = file_path + VECTOR_INDEX_POSTFIX;
    auto binaryset = index->Serialize(knowhere::Config());

    if (!fs_ptr->writer_ptr_->Open(full_file_path)) {
        THROW_ERROR(SERVER_CANNOT_OPEN_FILE, "Fail to open vector index: " + full_file_path);
    }

    for (auto& iter : binaryset.binary_map_) {
        auto meta = iter.first.c_str();
        size_t meta_length = iter.first.length();
        fs_ptr->writer_ptr_->Write(&meta_length, sizeof(meta_length));
        fs_ptr->writer_ptr_->Write((void*)meta, meta_length);

        auto binary = iter.second;
        int64_t binary_length = binary->size;
        fs_ptr->writer_ptr_->Write(&binary_length, sizeof(binary_length));
        fs_ptr->writer_ptr_->Write((void*)binary->data.get(), binary_length);
    }
    fs_ptr->writer_ptr_->Close();

    double span = recorder.RecordSection("End");
    double rate = fs_ptr->writer_ptr_->Length() * 1000000.0 / span / 1024 / 1024;
    LOG_ENGINE_DEBUG_ << "VectorIndexFormat::WriteIndex(" << full_file_path << ") rate " << rate << "MB/s";
}

void
VectorIndexFormat::WriteCompress(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                                 const knowhere::VecIndexPtr& index) {
    milvus::TimeRecorder recorder("VectorIndexFormat::WriteCompress");

    auto binaryset = index->Serialize(knowhere::Config());

    auto sq8_data = binaryset.Erase(SQ8_DATA);
    if (sq8_data != nullptr) {
        auto& ss_codec = codec::Codec::instance();
        ss_codec.GetVectorCompressFormat()->Write(fs_ptr, file_path, sq8_data);
    }
}

}  // namespace codec
}  // namespace milvus
