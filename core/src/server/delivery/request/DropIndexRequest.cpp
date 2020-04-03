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

#include "server/delivery/request/DropIndexRequest.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <fiu-local.h>
#include <memory>

namespace milvus {
namespace server {

DropIndexRequest::DropIndexRequest(const std::shared_ptr<milvus::server::Context>& context,
                                   const std::string& collection_name)
    : BaseRequest(context, BaseRequest::kDropIndex), collection_name_(collection_name) {
}

BaseRequestPtr
DropIndexRequest::Create(const std::shared_ptr<milvus::server::Context>& context, const std::string& collection_name) {
    return std::shared_ptr<BaseRequest>(new DropIndexRequest(context, collection_name));
}

Status
DropIndexRequest::OnExecute() {
    try {
        fiu_do_on("DropIndexRequest.OnExecute.throw_std_exception", throw std::exception());
        std::string hdr = "DropIndexRequest(collection=" + collection_name_ + ")";
        TimeRecorderAuto rc(hdr);

        // step 1: check arguments
        auto status = ValidationUtil::ValidateCollectionName(collection_name_);
        if (!status.ok()) {
            return status;
        }

        // only process root collection, ignore partition collection
        engine::meta::CollectionSchema table_schema;
        table_schema.collection_id_ = collection_name_;
        status = DBWrapper::DB()->DescribeCollection(table_schema);
        fiu_do_on("DropIndexRequest.OnExecute.table_not_exist", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        if (!status.ok()) {
            if (status.code() == DB_NOT_FOUND) {
                return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(collection_name_));
            } else {
                return status;
            }
        } else {
            if (!table_schema.owner_collection_.empty()) {
                return Status(SERVER_INVALID_TABLE_NAME, TableNotExistMsg(collection_name_));
            }
        }

        rc.RecordSection("check validation");

        // step 2: drop index
        status = DBWrapper::DB()->DropIndex(collection_name_);
        fiu_do_on("DropIndexRequest.OnExecute.drop_index_fail", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
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
