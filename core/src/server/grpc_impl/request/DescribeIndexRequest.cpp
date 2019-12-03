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

#include "server/grpc_impl/request/DescribeIndexRequest.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <memory>

namespace milvus {
namespace server {
namespace grpc {

DescribeIndexRequest::DescribeIndexRequest(const std::string& table_name, ::milvus::grpc::IndexParam* index_param)
    : GrpcBaseRequest(INFO_REQUEST_GROUP), table_name_(table_name), index_param_(index_param) {
}

BaseRequestPtr
DescribeIndexRequest::Create(const std::string& table_name, ::milvus::grpc::IndexParam* index_param) {
    return std::shared_ptr<GrpcBaseRequest>(new DescribeIndexRequest(table_name, index_param));
}

Status
DescribeIndexRequest::OnExecute() {
    try {
        std::string hdr = "DescribeIndexRequest(table=" + table_name_ + ")";
        TimeRecorderAuto rc(hdr);

        // step 1: check arguments
        auto status = ValidationUtil::ValidateTableName(table_name_);
        if (!status.ok()) {
            return status;
        }

        // step 2: check table existence
        engine::TableIndex index;
        status = DBWrapper::DB()->DescribeIndex(table_name_, index);
        if (!status.ok()) {
            return status;
        }

        index_param_->set_table_name(table_name_);
        index_param_->mutable_index()->set_index_type(index.engine_type_);
        index_param_->mutable_index()->set_nlist(index.nlist_);
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace grpc
}  // namespace server
}  // namespace milvus
