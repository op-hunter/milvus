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

#include <memory>
#include <string>
#include <vector>

#include "../grpc/ClientProxy.h"
#include "MilvusApi.h"

namespace milvus {

class ConnectionImpl : public Connection {
 public:
    ConnectionImpl();

    // Implementations of the Connection interface
    Status
    Connect(const ConnectParam& param) override;

    Status
    Connect(const std::string& uri) override;

    Status
    Connected() const override;

    Status
    Disconnect() override;

    Status
    CreateTable(const TableSchema& param) override;

    bool
    HasTable(const std::string& table_name) override;

    Status
    DropTable(const std::string& table_name) override;

    Status
    CreateIndex(const IndexParam& index_param) override;

    Status
    Insert(const std::string& table_name, const std::string& partition_tag, const std::vector<RowRecord>& record_array,
           std::vector<int64_t>& id_array) override;

    Status
    GetVectorByID(const std::string& table_name, int64_t vector_id, RowRecord& vector_data) override;

    Status
    GetIDsInSegment(const std::string& table_name, const std::string& segment_name,
                    std::vector<int64_t>& id_array) override;

    Status
    Search(const std::string& table_name, const std::vector<std::string>& partition_tag_array,
           const std::vector<RowRecord>& query_record_array, int64_t topk,
           const std::string& extra_params, TopKQueryResult& topk_query_result) override;

    Status
    DescribeTable(const std::string& table_name, TableSchema& table_schema) override;

    Status
    CountTable(const std::string& table_name, int64_t& row_count) override;

    Status
    ShowTables(std::vector<std::string>& table_array) override;

    Status
    ShowTableInfo(const std::string& table_name, TableInfo& table_info) override;

    std::string
    ClientVersion() const override;

    std::string
    ServerVersion() const override;

    std::string
    ServerStatus() const override;

    Status
    DeleteByID(const std::string& table_name, const std::vector<int64_t>& id_array) override;

    Status
    PreloadTable(const std::string& table_name) const override;

    Status
    DescribeIndex(const std::string& table_name, IndexParam& index_param) const override;

    Status
    DropIndex(const std::string& table_name) const override;

    Status
    CreatePartition(const PartitionParam& param) override;

    Status
    ShowPartitions(const std::string& table_name, PartitionTagList& partition_tag_array) const override;

    Status
    DropPartition(const PartitionParam& param) override;

    Status
    GetConfig(const std::string& node_name, std::string& value) const override;

    Status
    SetConfig(const std::string& node_name, const std::string& value) const override;

    Status
    FlushTable(const std::string& table_name) override;

    Status
    Flush() override;

    Status
    CompactTable(const std::string& table_name) override;

 private:
    std::shared_ptr<ClientProxy> client_proxy_;
};

}  // namespace milvus
