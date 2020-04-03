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

#include "grpc/GrpcClient.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <memory>
#include <string>
#include <vector>

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

namespace milvus {
GrpcClient::GrpcClient(std::shared_ptr<::grpc::Channel>& channel)
    : stub_(::milvus::grpc::MilvusService::NewStub(channel)) {
}

GrpcClient::~GrpcClient() = default;

Status
GrpcClient::CreateTable(const ::milvus::grpc::TableSchema& table_schema) {
    ClientContext context;
    grpc::Status response;
    ::grpc::Status grpc_status = stub_->CreateTable(&context, table_schema, &response);

    if (!grpc_status.ok()) {
        std::cerr << "CreateTable gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

bool
GrpcClient::HasCollection(const ::milvus::grpc::TableName& collection_name, Status& status) {
    ClientContext context;
    ::milvus::grpc::BoolReply response;
    ::grpc::Status grpc_status = stub_->HasCollection(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "HasCollection gRPC failed!" << std::endl;
        status = Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (response.status().error_code() != grpc::SUCCESS) {
        std::cerr << response.status().reason() << std::endl;
        status = Status(StatusCode::ServerFailed, response.status().reason());
    }
    status = Status::OK();
    return response.bool_reply();
}

Status
GrpcClient::DropTable(const ::milvus::grpc::TableName& collection_name) {
    ClientContext context;
    grpc::Status response;
    ::grpc::Status grpc_status = stub_->DropTable(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "DropTable gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }

    return Status::OK();
}

Status
GrpcClient::CreateIndex(const ::milvus::grpc::IndexParam& index_param) {
    ClientContext context;
    grpc::Status response;
    ::grpc::Status grpc_status = stub_->CreateIndex(&context, index_param, &response);

    if (!grpc_status.ok()) {
        std::cerr << "CreateIndex rpc failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }

    return Status::OK();
}

Status
GrpcClient::Insert(const ::milvus::grpc::InsertParam& insert_param, ::milvus::grpc::VectorIds& vector_ids) {
    ClientContext context;
    ::grpc::Status grpc_status = stub_->Insert(&context, insert_param, &vector_ids);

    if (!grpc_status.ok()) {
        std::cerr << "Insert rpc failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (vector_ids.status().error_code() != grpc::SUCCESS) {
        std::cerr << vector_ids.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, vector_ids.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::GetVectorByID(const grpc::VectorIdentity& vector_identity, ::milvus::grpc::VectorData& vector_data) {
    ClientContext context;
    ::grpc::Status grpc_status = stub_->GetVectorByID(&context, vector_identity, &vector_data);

    if (!grpc_status.ok()) {
        std::cerr << "GetVectorByID rpc failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (vector_data.status().error_code() != grpc::SUCCESS) {
        std::cerr << vector_data.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, vector_data.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::GetIDsInSegment(const grpc::GetVectorIDsParam& param, grpc::VectorIds& vector_ids) {
    ClientContext context;
    ::grpc::Status grpc_status = stub_->GetVectorIDs(&context, param, &vector_ids);

    if (!grpc_status.ok()) {
        std::cerr << "GetIDsInSegment rpc failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (vector_ids.status().error_code() != grpc::SUCCESS) {
        std::cerr << vector_ids.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, vector_ids.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::Search(
    const ::milvus::grpc::SearchParam& search_param, ::milvus::grpc::TopKQueryResult& topk_query_result) {
    ::milvus::grpc::TopKQueryResult query_result;
    ClientContext context;
    ::grpc::Status grpc_status = stub_->Search(&context, search_param, &topk_query_result);

    if (!grpc_status.ok()) {
        std::cerr << "Search rpc failed!" << std::endl;
        std::cerr << grpc_status.error_message() << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (topk_query_result.status().error_code() != grpc::SUCCESS) {
        std::cerr << topk_query_result.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, topk_query_result.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::DescribeTable(const std::string& collection_name, ::milvus::grpc::TableSchema& grpc_schema) {
    ClientContext context;
    ::milvus::grpc::TableName grpc_tablename;
    grpc_tablename.set_table_name(collection_name);
    ::grpc::Status grpc_status = stub_->DescribeTable(&context, grpc_tablename, &grpc_schema);

    if (!grpc_status.ok()) {
        std::cerr << "DescribeTable rpc failed!" << std::endl;
        std::cerr << grpc_status.error_message() << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (grpc_schema.status().error_code() != grpc::SUCCESS) {
        std::cerr << grpc_schema.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, grpc_schema.status().reason());
    }

    return Status::OK();
}

int64_t
GrpcClient::CountTable(grpc::TableName& collection_name, Status& status) {
    ClientContext context;
    ::milvus::grpc::TableRowCount response;
    ::grpc::Status grpc_status = stub_->CountTable(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "CountTable rpc failed!" << std::endl;
        status = Status(StatusCode::RPCFailed, grpc_status.error_message());
        return -1;
    }

    if (response.status().error_code() != grpc::SUCCESS) {
        std::cerr << response.status().reason() << std::endl;
        status = Status(StatusCode::ServerFailed, response.status().reason());
        return -1;
    }

    status = Status::OK();
    return response.table_row_count();
}

Status
GrpcClient::ShowTables(milvus::grpc::TableNameList& table_name_list) {
    ClientContext context;
    ::milvus::grpc::Command command;
    ::grpc::Status grpc_status = stub_->ShowTables(&context, command, &table_name_list);

    if (!grpc_status.ok()) {
        std::cerr << "ShowTables gRPC failed!" << std::endl;
        std::cerr << grpc_status.error_message() << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (table_name_list.status().error_code() != grpc::SUCCESS) {
        std::cerr << table_name_list.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, table_name_list.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::ShowTableInfo(grpc::TableName& collection_name, grpc::TableInfo& collection_info) {
    ClientContext context;
    ::milvus::grpc::Command command;
    ::grpc::Status grpc_status = stub_->ShowTableInfo(&context, collection_name, &collection_info);

    if (!grpc_status.ok()) {
        std::cerr << "ShowTableInfo gRPC failed!" << std::endl;
        std::cerr << grpc_status.error_message() << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (collection_info.status().error_code() != grpc::SUCCESS) {
        std::cerr << collection_info.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, collection_info.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::Cmd(const std::string& cmd, std::string& result) {
    ClientContext context;
    ::milvus::grpc::StringReply response;
    ::milvus::grpc::Command command;
    command.set_cmd(cmd);
    ::grpc::Status grpc_status = stub_->Cmd(&context, command, &response);

    result = response.string_reply();
    if (!grpc_status.ok()) {
        std::cerr << "Cmd gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.status().error_code() != grpc::SUCCESS) {
        std::cerr << response.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::PreloadTable(milvus::grpc::TableName& collection_name) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->PreloadTable(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "PreloadTable gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::DeleteByID(grpc::DeleteByIDParam& delete_by_id_param) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->DeleteByID(&context, delete_by_id_param, &response);

    if (!grpc_status.ok()) {
        std::cerr << "DeleteByID gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::DescribeIndex(grpc::TableName& collection_name, grpc::IndexParam& index_param) {
    ClientContext context;
    ::grpc::Status grpc_status = stub_->DescribeIndex(&context, collection_name, &index_param);

    if (!grpc_status.ok()) {
        std::cerr << "DescribeIndex rpc failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }
    if (index_param.status().error_code() != grpc::SUCCESS) {
        std::cerr << index_param.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, index_param.status().reason());
    }

    return Status::OK();
}

Status
GrpcClient::DropIndex(grpc::TableName& collection_name) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->DropIndex(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "DropIndex gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::CreatePartition(const grpc::PartitionParam& partition_param) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->CreatePartition(&context, partition_param, &response);

    if (!grpc_status.ok()) {
        std::cerr << "CreatePartition gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::ShowPartitions(const grpc::TableName& collection_name, grpc::PartitionList& partition_array) const {
    ClientContext context;
    ::grpc::Status grpc_status = stub_->ShowPartitions(&context, collection_name, &partition_array);

    if (!grpc_status.ok()) {
        std::cerr << "ShowPartitions gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (partition_array.status().error_code() != grpc::SUCCESS) {
        std::cerr << partition_array.status().reason() << std::endl;
        return Status(StatusCode::ServerFailed, partition_array.status().reason());
    }
    return Status::OK();
}

Status
GrpcClient::DropPartition(const ::milvus::grpc::PartitionParam& partition_param) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->DropPartition(&context, partition_param, &response);

    if (!grpc_status.ok()) {
        std::cerr << "DropPartition gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::Flush(const std::string& collection_name) {
    ClientContext context;

    ::milvus::grpc::FlushParam param;
    if (!collection_name.empty()) {
        param.add_table_name_array(collection_name);
    }

    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->Flush(&context, param, &response);

    if (!grpc_status.ok()) {
        std::cerr << "Flush gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::Compact(milvus::grpc::TableName& collection_name) {
    ClientContext context;
    ::milvus::grpc::Status response;
    ::grpc::Status grpc_status = stub_->Compact(&context, collection_name, &response);

    if (!grpc_status.ok()) {
        std::cerr << "Compact gRPC failed!" << std::endl;
        return Status(StatusCode::RPCFailed, grpc_status.error_message());
    }

    if (response.error_code() != grpc::SUCCESS) {
        std::cerr << response.reason() << std::endl;
        return Status(StatusCode::ServerFailed, response.reason());
    }
    return Status::OK();
}

Status
GrpcClient::Disconnect() {
    stub_.release();
    return Status::OK();
}

}  // namespace milvus
