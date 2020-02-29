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

#pragma once

#include "include/MilvusApi.h"
#include "grpc-gen/gen-milvus/milvus.grpc.pb.h"
//#include "grpc/gen-status/status.grpc.pb.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

namespace milvus {
class GrpcClient {
 public:
    explicit GrpcClient(std::shared_ptr<::grpc::Channel>& channel);

    virtual ~GrpcClient();

    Status
    CreateTable(const grpc::TableSchema& table_schema);

    bool
    HasTable(const grpc::TableName& table_name, Status& status);

    Status
    DropTable(const grpc::TableName& table_name);

    Status
    CreateIndex(const grpc::IndexParam& index_param);

    Status
    Insert(const grpc::InsertParam& insert_param, grpc::VectorIds& vector_ids);

    Status
    GetVectorByID(const grpc::VectorIdentity& vector_identity, ::milvus::grpc::VectorData& vector_data);

    Status
    GetIDsInSegment(const grpc::GetVectorIDsParam& param, grpc::VectorIds& vector_ids);

    Status
    Search(const grpc::SearchParam& search_param, ::milvus::grpc::TopKQueryResult& topk_query_result);

    Status
    SearchByID(const grpc::SearchByIDParam& search_param, ::milvus::grpc::TopKQueryResult& topk_query_result);

    Status
    DescribeTable(const std::string& table_name, grpc::TableSchema& grpc_schema);

    int64_t
    CountTable(grpc::TableName& table_name, Status& status);

    Status
    ShowTables(milvus::grpc::TableNameList& table_name_list);

    Status
    ShowTableInfo(grpc::TableName& table_name, grpc::TableInfo& table_info);

    Status
    Cmd(const std::string& cmd, std::string& result);

    Status
    DeleteByID(grpc::DeleteByIDParam& delete_by_id_param);

    Status
    PreloadTable(grpc::TableName& table_name);

    Status
    DescribeIndex(grpc::TableName& table_name, grpc::IndexParam& index_param);

    Status
    DropIndex(grpc::TableName& table_name);

    Status
    CreatePartition(const grpc::PartitionParam& partition_param);

    Status
    ShowPartitions(const grpc::TableName& table_name, grpc::PartitionList& partition_array) const;

    Status
    DropPartition(const ::milvus::grpc::PartitionParam& partition_param);

    Status
    Flush(const std::string& table_name);

    Status
    Compact(milvus::grpc::TableName& table_name);

    Status
    Disconnect();

 private:
    std::unique_ptr<grpc::MilvusService::Stub> stub_;
};

}  // namespace milvus
