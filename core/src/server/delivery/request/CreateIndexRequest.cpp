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

#include "server/delivery/request/CreateIndexRequest.h"
#include "config/Config.h"
#include "db/Utils.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <fiu-local.h>
#include <memory>
#include <string>

namespace milvus {
namespace server {

CreateIndexRequest::CreateIndexRequest(const std::shared_ptr<Context>& context, const std::string& table_name,
                                       int64_t index_type, const milvus::json& json_params)
    : BaseRequest(context, DDL_DML_REQUEST_GROUP),
      table_name_(table_name),
      index_type_(index_type),
      json_params_(json_params) {
}

BaseRequestPtr
CreateIndexRequest::Create(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t index_type,
                           const milvus::json& json_params) {
    return std::shared_ptr<BaseRequest>(new CreateIndexRequest(context, table_name, index_type, json_params));
}

Status
CreateIndexRequest::OnExecute() {
    try {
        std::string hdr = "CreateIndexRequest(table=" + table_name_ + ")";
        TimeRecorderAuto rc(hdr);

        // step 1: check arguments
        auto status = ValidationUtil::ValidateTableName(table_name_);
        if (!status.ok()) {
            return status;
        }

        // only process root table, ignore partition table
        engine::meta::TableSchema table_schema;
        table_schema.table_id_ = table_name_;
        status = DBWrapper::DB()->DescribeTable(table_schema);
        fiu_do_on("CreateIndexRequest.OnExecute.not_has_table", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        fiu_do_on("CreateIndexRequest.OnExecute.throw_std.exception", throw std::exception());
        if (!status.ok()) {
            if (status.code() == DB_NOT_FOUND) {
                return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(table_name_));
            } else {
                return status;
            }
        } else {
            if (!table_schema.owner_table_.empty()) {
                return Status(SERVER_INVALID_TABLE_NAME, TableNotExistMsg(table_name_));
            }
        }

        status = ValidationUtil::ValidateTableIndexType(index_type_);
        if (!status.ok()) {
            return status;
        }

        status = ValidationUtil::ValidateIndexParams(json_params_, table_schema, index_type_);
        if (!status.ok()) {
            return status;
        }

        // step 2: binary and float vector support different index/metric type, need to adapt here
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        status = DBWrapper::DB()->DescribeTable(table_info);

        int32_t adapter_index_type = index_type_;
        if (engine::utils::IsBinaryMetricType(table_info.metric_type_)) {  // binary vector not allow
            if (adapter_index_type == static_cast<int32_t>(engine::EngineType::FAISS_IDMAP)) {
                adapter_index_type = static_cast<int32_t>(engine::EngineType::FAISS_BIN_IDMAP);
            } else if (adapter_index_type == static_cast<int32_t>(engine::EngineType::FAISS_IVFFLAT)) {
                adapter_index_type = static_cast<int32_t>(engine::EngineType::FAISS_BIN_IVFFLAT);
            } else {
                return Status(SERVER_INVALID_INDEX_TYPE, "Invalid index type for table metric type");
            }
        }

#ifdef MILVUS_GPU_VERSION
        Status s;
        bool enable_gpu = false;
        server::Config& config = server::Config::GetInstance();
        s = config.GetGpuResourceConfigEnable(enable_gpu);
        fiu_do_on("CreateIndexRequest.OnExecute.ip_meteric",
                  table_info.metric_type_ = static_cast<int>(engine::MetricType::IP));

        if (s.ok() && adapter_index_type == (int)engine::EngineType::FAISS_PQ &&
            table_info.metric_type_ == (int)engine::MetricType::IP) {
            return Status(SERVER_UNEXPECTED_ERROR, "PQ not support IP in GPU version!");
        }
#endif

        rc.RecordSection("check validation");

        // step 3: create index
        engine::TableIndex index;
        index.engine_type_ = adapter_index_type;
        index.extra_params_ = json_params_;
        status = DBWrapper::DB()->CreateIndex(table_name_, index);
        fiu_do_on("CreateIndexRequest.OnExecute.create_index_fail",
                  status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
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
