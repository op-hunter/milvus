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

#include "server/grpc_impl/request/HasTableRequest.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <memory>

namespace milvus {
namespace server {
namespace grpc {

HasTableRequest::HasTableRequest(const std::string& table_name, bool& has_table)
    : GrpcBaseRequest(INFO_REQUEST_GROUP), table_name_(table_name), has_table_(has_table) {
}

BaseRequestPtr
HasTableRequest::Create(const std::string& table_name, bool& has_table) {
    return std::shared_ptr<GrpcBaseRequest>(new HasTableRequest(table_name, has_table));
}

Status
HasTableRequest::OnExecute() {
    try {
        TimeRecorder rc("HasTableRequest");

        // step 1: check arguments
        auto status = ValidationUtil::ValidateTableName(table_name_);
        if (!status.ok()) {
            return status;
        }

        // step 2: check table existence
        status = DBWrapper::DB()->HasTable(table_name_, has_table_);
        if (!status.ok()) {
            return status;
        }

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace grpc
}  // namespace server
}  // namespace milvus
