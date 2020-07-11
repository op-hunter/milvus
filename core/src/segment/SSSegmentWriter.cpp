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

#include "segment/SSSegmentWriter.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "SSSegmentReader.h"
#include "Vectors.h"
#include "codecs/snapshot/SSCodec.h"
#include "db/Utils.h"
#include "db/snapshot/ResourceHelper.h"
#include "storage/disk/DiskIOReader.h"
#include "storage/disk/DiskIOWriter.h"
#include "storage/disk/DiskOperation.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace segment {

SSSegmentWriter::SSSegmentWriter(const engine::SegmentVisitorPtr& segment_visitor) : segment_visitor_(segment_visitor) {
    auto& segment_ptr = segment_visitor_->GetSegment();
    std::string directory = engine::snapshot::GetResPath<engine::snapshot::Segment>(segment_ptr);

    storage::IOReaderPtr reader_ptr = std::make_shared<storage::DiskIOReader>();
    storage::IOWriterPtr writer_ptr = std::make_shared<storage::DiskIOWriter>();
    storage::OperationPtr operation_ptr = std::make_shared<storage::DiskOperation>(directory);
    fs_ptr_ = std::make_shared<storage::FSHandler>(reader_ptr, writer_ptr, operation_ptr);
    segment_ptr_ = std::make_shared<Segment>();
}

Status
SSSegmentWriter::AddVectors(const std::string& name, const std::vector<uint8_t>& data,
                            const std::vector<doc_id_t>& uids) {
    segment_ptr_->vectors_ptr_->AddData(data);
    segment_ptr_->vectors_ptr_->AddUids(uids);
    segment_ptr_->vectors_ptr_->SetName(name);

    return Status::OK();
}

Status
SSSegmentWriter::AddVectors(const std::string& name, const uint8_t* data, uint64_t size,
                            const std::vector<doc_id_t>& uids) {
    segment_ptr_->vectors_ptr_->AddData(data, size);
    segment_ptr_->vectors_ptr_->AddUids(uids);
    segment_ptr_->vectors_ptr_->SetName(name);

    return Status::OK();
}

Status
SSSegmentWriter::AddAttrs(const std::string& name, const std::unordered_map<std::string, uint64_t>& attr_nbytes,
                          const std::unordered_map<std::string, std::vector<uint8_t>>& attr_data,
                          const std::vector<doc_id_t>& uids) {
    auto attr_data_it = attr_data.begin();
    auto attrs = segment_ptr_->attrs_ptr_->attrs;
    for (; attr_data_it != attr_data.end(); ++attr_data_it) {
        AttrPtr attr = std::make_shared<Attr>(attr_data_it->second, attr_nbytes.at(attr_data_it->first), uids,
                                              attr_data_it->first);
        segment_ptr_->attrs_ptr_->attrs.insert(std::make_pair(attr_data_it->first, attr));

        //        if (attrs.find(attr_data_it->first) != attrs.end()) {
        //            segment_ptr_->attrs_ptr_->attrs.at(attr_data_it->first)
        //                ->AddAttr(attr_data_it->second, attr_nbytes.at(attr_data_it->first));
        //            segment_ptr_->attrs_ptr_->attrs.at(attr_data_it->first)->AddUids(uids);
        //        } else {
        //            AttrPtr attr = std::make_shared<Attr>(attr_data_it->second, attr_nbytes.at(attr_data_it->first),
        //            uids,
        //                                                  attr_data_it->first);
        //            segment_ptr_->attrs_ptr_->attrs.insert(std::make_pair(attr_data_it->first, attr));
        //        }
    }
    return Status::OK();
}

Status
SSSegmentWriter::SetAttrsIndex(const std::unordered_map<std::string, knowhere::IndexPtr>& attr_indexes,
                               const std::unordered_map<std::string, int64_t>& attr_sizes,
                               const std::unordered_map<std::string, engine::meta::hybrid::DataType>& attr_type) {
    auto attrs_index = std::make_shared<AttrsIndex>();
    auto attr_it = attr_indexes.begin();
    for (; attr_it != attr_indexes.end(); attr_it++) {
        auto attr_index = std::make_shared<AttrIndex>();
        attr_index->SetFieldName(attr_it->first);
        attr_index->SetDataType(attr_type.at(attr_it->first));
        attr_index->SetAttrIndex(attr_it->second);
        attrs_index->attr_indexes.insert(std::make_pair(attr_it->first, attr_index));
    }
    segment_ptr_->attrs_index_ptr_ = attrs_index;
    return Status::OK();
}

Status
SSSegmentWriter::SetVectorIndex(const milvus::knowhere::VecIndexPtr& index) {
    segment_ptr_->vector_index_ptr_->SetVectorIndex(index);
    return Status::OK();
}

Status
SSSegmentWriter::Serialize() {
    auto& field_visitors_map = segment_visitor_->GetFieldVisitors();
    auto uid_field_visitor = segment_visitor_->GetFieldVisitor(engine::DEFAULT_UID_NAME);

    /* write UID's raw data */
    auto uid_raw_visitor = uid_field_visitor->GetElementVisitor(engine::FieldElementType::FET_RAW);
    std::string uid_raw_path = engine::snapshot::GetResPath<engine::snapshot::SegmentFile>(uid_raw_visitor->GetFile());
    STATUS_CHECK(WriteUids(uid_raw_path, segment_ptr_->vectors_ptr_->GetUids()));

    /* write UID's deleted docs */
    auto uid_del_visitor = uid_field_visitor->GetElementVisitor(engine::FieldElementType::FET_DELETED_DOCS);
    std::string uid_del_path = engine::snapshot::GetResPath<engine::snapshot::SegmentFile>(uid_del_visitor->GetFile());
    STATUS_CHECK(WriteDeletedDocs(uid_del_path, segment_ptr_->deleted_docs_ptr_));

    /* write UID's bloom filter */
    auto uid_blf_visitor = uid_field_visitor->GetElementVisitor(engine::FieldElementType::FET_BLOOM_FILTER);
    std::string uid_blf_path = engine::snapshot::GetResPath<engine::snapshot::SegmentFile>(uid_blf_visitor->GetFile());
    STATUS_CHECK(WriteBloomFilter(uid_blf_path, segment_ptr_->id_bloom_filter_ptr_));

    /* write other data */
    for (auto& f_kv : field_visitors_map) {
        auto& field_visitor = f_kv.second;
        auto& field = field_visitor->GetField();
        for (auto& file_kv : field_visitor->GetElementVistors()) {
            auto& field_element_visitor = file_kv.second;

            auto& segment_file = field_element_visitor->GetFile();
            if (segment_file == nullptr) {
                continue;
            }
            auto file_path = engine::snapshot::GetResPath<engine::snapshot::SegmentFile>(segment_file);
            auto& field_element = field_element_visitor->GetElement();

            if ((field->GetFtype() == engine::FieldType::VECTOR_FLOAT ||
                 field->GetFtype() == engine::FieldType::VECTOR_BINARY) &&
                field_element->GetFtype() == engine::FieldElementType::FET_RAW) {
                STATUS_CHECK(WriteVectors(file_path, segment_ptr_->vectors_ptr_->GetData()));
            }

            /* SS TODO: write attr data ? */
        }
    }

    return Status::OK();
}

Status
SSSegmentWriter::WriteUids(const std::string& file_path, const std::vector<doc_id_t>& uids) {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetVectorsFormat()->write_uids(fs_ptr_, file_path, uids);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vectors: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteVectors(const std::string& file_path, const std::vector<uint8_t>& raw_vectors) {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetVectorsFormat()->write_vectors(fs_ptr_, file_path, raw_vectors);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vectors: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteAttrs() {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetAttrsFormat()->write(fs_ptr_, segment_ptr_->attrs_ptr_);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vectors: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteVectorIndex(const std::string& location) {
    if (location.empty()) {
        return Status(SERVER_WRITE_ERROR, "Invalid parameter of WriteVectorIndex");
    }

    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetVectorIndexFormat()->write(fs_ptr_, location, segment_ptr_->vector_index_ptr_);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vector index: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteAttrsIndex() {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetAttrsIndexFormat()->write(fs_ptr_, segment_ptr_->attrs_index_ptr_);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vector index: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteBloomFilter(const std::string& file_path) {
    try {
        auto& ss_codec = codec::SSCodec::instance();

        fs_ptr_->operation_ptr_->CreateDirectory();

        TimeRecorder recorder("SSSegmentWriter::WriteBloomFilter");

        ss_codec.GetIdBloomFilterFormat()->create(fs_ptr_, segment_ptr_->id_bloom_filter_ptr_);

        recorder.RecordSection("Initializing bloom filter");

        auto& uids = segment_ptr_->vectors_ptr_->GetUids();
        for (auto& uid : uids) {
            segment_ptr_->id_bloom_filter_ptr_->Add(uid);
        }

        recorder.RecordSection("Adding " + std::to_string(uids.size()) + " ids to bloom filter");

        ss_codec.GetIdBloomFilterFormat()->write(fs_ptr_, file_path, segment_ptr_->id_bloom_filter_ptr_);

        recorder.RecordSection("Writing bloom filter");
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write vectors: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteDeletedDocs(const std::string& file_path) {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        DeletedDocsPtr deleted_docs_ptr = std::make_shared<DeletedDocs>();
        ss_codec.GetDeletedDocsFormat()->write(fs_ptr_, file_path, deleted_docs_ptr);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write deleted docs: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteDeletedDocs(const std::string& file_path, const DeletedDocsPtr& deleted_docs) {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetDeletedDocsFormat()->write(fs_ptr_, file_path, deleted_docs);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write deleted docs: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::WriteBloomFilter(const std::string& file_path, const IdBloomFilterPtr& id_bloom_filter_ptr) {
    try {
        auto& ss_codec = codec::SSCodec::instance();
        fs_ptr_->operation_ptr_->CreateDirectory();
        ss_codec.GetIdBloomFilterFormat()->write(fs_ptr_, file_path, id_bloom_filter_ptr);
    } catch (std::exception& e) {
        std::string err_msg = "Failed to write bloom filter: " + std::string(e.what());
        LOG_ENGINE_ERROR_ << err_msg;

        engine::utils::SendExitSignal();
        return Status(SERVER_WRITE_ERROR, err_msg);
    }
    return Status::OK();
}

Status
SSSegmentWriter::Cache() {
    // TODO(zhiru)
    return Status::OK();
}

Status
SSSegmentWriter::GetSegment(SegmentPtr& segment_ptr) {
    segment_ptr = segment_ptr_;
    return Status::OK();
}

Status
SSSegmentWriter::Merge(const std::string& dir_to_merge, const std::string& name) {
    //    if (dir_to_merge == fs_ptr_->operation_ptr_->GetDirectory()) {
    //        return Status(DB_ERROR, "Cannot Merge Self");
    //    }
    //
    //    LOG_ENGINE_DEBUG_ << "Merging from " << dir_to_merge << " to " << fs_ptr_->operation_ptr_->GetDirectory();
    //
    //    TimeRecorder recorder("SSSegmentWriter::Merge");
    //
    //    SSSegmentReader segment_reader_to_merge(dir_to_merge);
    //    bool in_cache;
    //    auto status = segment_reader_to_merge.LoadCache(in_cache);
    //    if (!in_cache) {
    //        status = segment_reader_to_merge.Load();
    //        if (!status.ok()) {
    //            std::string msg = "Failed to load segment from " + dir_to_merge;
    //            LOG_ENGINE_ERROR_ << msg;
    //            return Status(DB_ERROR, msg);
    //        }
    //    }
    //    SegmentPtr segment_to_merge;
    //    segment_reader_to_merge.GetSegment(segment_to_merge);
    //    // auto& uids = segment_to_merge->vectors_ptr_->GetUids();
    //
    //    recorder.RecordSection("Loading segment");
    //
    //    if (segment_to_merge->deleted_docs_ptr_ != nullptr) {
    //        auto offsets_to_delete = segment_to_merge->deleted_docs_ptr_->GetDeletedDocs();
    //
    //        // Erase from raw data
    //        segment_to_merge->vectors_ptr_->Erase(offsets_to_delete);
    //    }
    //
    //    recorder.RecordSection("erase");
    //
    //    AddVectors(name, segment_to_merge->vectors_ptr_->GetData(), segment_to_merge->vectors_ptr_->GetUids());
    //
    //    auto rows = segment_to_merge->vectors_ptr_->GetCount();
    //    recorder.RecordSection("Adding " + std::to_string(rows) + " vectors and uids");
    //
    //    std::unordered_map<std::string, uint64_t> attr_nbytes;
    //    std::unordered_map<std::string, std::vector<uint8_t>> attr_data;
    //    auto attr_it = segment_to_merge->attrs_ptr_->attrs.begin();
    //    for (; attr_it != segment_to_merge->attrs_ptr_->attrs.end(); attr_it++) {
    //        attr_nbytes.insert(std::make_pair(attr_it->first, attr_it->second->GetNbytes()));
    //        attr_data.insert(std::make_pair(attr_it->first, attr_it->second->GetData()));
    //
    //        if (segment_to_merge->deleted_docs_ptr_ != nullptr) {
    //            auto offsets_to_delete = segment_to_merge->deleted_docs_ptr_->GetDeletedDocs();
    //
    //            // Erase from field data
    //            attr_it->second->Erase(offsets_to_delete);
    //        }
    //    }
    //    AddAttrs(name, attr_nbytes, attr_data, segment_to_merge->vectors_ptr_->GetUids());
    //
    //  LOG_ENGINE_DEBUG_ << "Merging completed from " << dir_to_merge << " to " <<
    //  fs_ptr_->operation_ptr_->GetDirectory();

    return Status::OK();
}

size_t
SSSegmentWriter::Size() {
    // TODO(zhiru): switch to actual directory size
    size_t vectors_size = segment_ptr_->vectors_ptr_->VectorsSize();
    size_t uids_size = segment_ptr_->vectors_ptr_->UidsSize();
    /*
    if (segment_ptr_->id_bloom_filter_ptr_) {
        ret += segment_ptr_->id_bloom_filter_ptr_->Size();
    }
     */
    return (vectors_size * sizeof(uint8_t) + uids_size * sizeof(doc_id_t));
}

size_t
SSSegmentWriter::VectorCount() {
    return segment_ptr_->vectors_ptr_->GetCount();
}

void
SSSegmentWriter::SetSegmentName(const std::string& name) {
    segment_ptr_->vectors_ptr_->SetName(name);
}

}  // namespace segment
}  // namespace milvus
