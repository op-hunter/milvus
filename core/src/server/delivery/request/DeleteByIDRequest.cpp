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

#include "server/delivery/request/DeleteByIDRequest.h"

#include <memory>
#include <string>
#include <vector>

#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

namespace milvus {
namespace server {

DeleteByIDRequest::DeleteByIDRequest(const std::shared_ptr<milvus::server::Context>& context,
                                     const std::string& collection_name, const std::vector<int64_t>& vector_ids)
    : BaseRequest(context, BaseRequest::kDeleteByID), collection_name_(collection_name), vector_ids_(vector_ids) {
}

BaseRequestPtr
DeleteByIDRequest::Create(const std::shared_ptr<milvus::server::Context>& context, const std::string& collection_name,
                          const std::vector<int64_t>& vector_ids) {
    return std::shared_ptr<BaseRequest>(new DeleteByIDRequest(context, collection_name, vector_ids));
}

Status
DeleteByIDRequest::OnExecute() {
    try {
        TimeRecorderAuto rc("DeleteByIDRequest");

        // step 1: check arguments
        auto status = ValidationUtil::ValidateCollectionName(collection_name_);
        if (!status.ok()) {
            return status;
        }

        // step 2: check collection existence
        engine::meta::CollectionSchema table_schema;
        table_schema.collection_id_ = collection_name_;
        status = DBWrapper::DB()->DescribeTable(table_schema);
        if (!status.ok()) {
            if (status.code() == DB_NOT_FOUND) {
                return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(collection_name_));
            } else {
                return status;
            }
        } else {
            if (!table_schema.owner_table_.empty()) {
                return Status(SERVER_INVALID_TABLE_NAME, TableNotExistMsg(collection_name_));
            }
        }

        // Check collection's index type supports delete
        if (table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_IDMAP &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_BIN_IDMAP &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::HNSW &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::ANNOY &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_IVFFLAT &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_BIN_IVFFLAT &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_IVFSQ8 &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_PQ &&
            table_schema.engine_type_ != (int32_t)engine::EngineType::FAISS_IVFSQ8H) {
            std::string err_msg =
                "Index type " + std::to_string(table_schema.engine_type_) + " does not support delete operation";
            SERVER_LOG_ERROR << err_msg;
            return Status(SERVER_UNSUPPORTED_ERROR, err_msg);
        }

        rc.RecordSection("check validation");

        status = DBWrapper::DB()->DeleteVectors(collection_name_, vector_ids_);
        if (!status.ok()) {
            return status;
        }
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
