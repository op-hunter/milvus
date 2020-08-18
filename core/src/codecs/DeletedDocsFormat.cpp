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

#include "codecs/DeletedDocsFormat.h"

#include <fcntl.h>
#include <unistd.h>

#include <experimental/filesystem>

#include <memory>
#include <string>
#include <vector>

#include "utils/Exception.h"
#include "utils/Log.h"

namespace milvus {
namespace codec {

const char* DELETED_DOCS_POSTFIX = ".del";

std::string
DeletedDocsFormat::FilePostfix() {
    std::string str = DELETED_DOCS_POSTFIX;
    return str;
}

void
DeletedDocsFormat::Read(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                        segment::DeletedDocsPtr& deleted_docs) {
    const std::string full_file_path = file_path + DELETED_DOCS_POSTFIX;

    if (!fs_ptr->reader_ptr_->Open(full_file_path)) {
        THROW_ERROR(SERVER_CANNOT_OPEN_FILE, "Fail to open deleted docs file: " + full_file_path);
    }

    size_t num_bytes;
    fs_ptr->reader_ptr_->Read(&num_bytes, sizeof(size_t));

    auto deleted_docs_size = num_bytes / sizeof(engine::offset_t);
    std::vector<engine::offset_t> deleted_docs_list;
    deleted_docs_list.resize(deleted_docs_size);

    fs_ptr->reader_ptr_->Read(deleted_docs_list.data(), num_bytes);
    fs_ptr->reader_ptr_->Close();

    deleted_docs = std::make_shared<segment::DeletedDocs>(deleted_docs_list);
}

void
DeletedDocsFormat::Write(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                         const segment::DeletedDocsPtr& deleted_docs) {
    const std::string full_file_path = file_path + DELETED_DOCS_POSTFIX;

    // Create a temporary file from the existing file
    const std::string temp_path = file_path + ".temp_del";
    bool exists = std::experimental::filesystem::exists(full_file_path);
    if (exists) {
        std::experimental::filesystem::copy_file(full_file_path, temp_path,
                                                 std::experimental::filesystem::copy_options::none);
    }

    // Write to the temp file, in order to avoid possible race condition with search (concurrent read and write)
    size_t old_num_bytes;
    std::vector<engine::offset_t> delete_ids;
    if (exists) {
        if (!fs_ptr->reader_ptr_->Open(temp_path)) {
            THROW_ERROR(SERVER_CANNOT_OPEN_FILE, "Fail to open tmp deleted docs file: " + temp_path);
        }
        fs_ptr->reader_ptr_->Read(&old_num_bytes, sizeof(size_t));
        delete_ids.resize(old_num_bytes / sizeof(engine::offset_t));
        fs_ptr->reader_ptr_->Read(delete_ids.data(), old_num_bytes);
        fs_ptr->reader_ptr_->Close();
    } else {
        old_num_bytes = 0;
    }

    auto deleted_docs_list = deleted_docs->GetDeletedDocs();
    size_t new_num_bytes = old_num_bytes + sizeof(engine::offset_t) * deleted_docs->GetCount();
    if (!deleted_docs_list.empty()) {
        delete_ids.insert(delete_ids.end(), deleted_docs_list.begin(), deleted_docs_list.end());
    }

    if (!fs_ptr->writer_ptr_->Open(temp_path)) {
        THROW_ERROR(SERVER_CANNOT_CREATE_FILE, "Fail to write file: " + temp_path);
    }

    fs_ptr->writer_ptr_->Write(&new_num_bytes, sizeof(size_t));
    fs_ptr->writer_ptr_->Write(delete_ids.data(), new_num_bytes);
    fs_ptr->writer_ptr_->Close();

    // Move temp file to delete file
    std::experimental::filesystem::rename(temp_path, full_file_path);
}

void
DeletedDocsFormat::ReadSize(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path, size_t& size) {
    const std::string full_file_path = file_path + DELETED_DOCS_POSTFIX;
    if (!fs_ptr->writer_ptr_->Open(full_file_path)) {
        THROW_ERROR(SERVER_CANNOT_CREATE_FILE, "Fail to open deleted docs file: " + full_file_path);
    }

    size_t num_bytes;
    fs_ptr->reader_ptr_->Read(&num_bytes, sizeof(size_t));

    size = num_bytes / sizeof(engine::offset_t);
    fs_ptr->reader_ptr_->Close();
}

}  // namespace codec
}  // namespace milvus
