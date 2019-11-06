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

#pragma once

#include "Meta.h"
#include "MySQLConnectionPool.h"
#include "db/Options.h"

#include <mysql++/mysql++.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace milvus {
namespace engine {
namespace meta {

class MySQLMetaImpl : public Meta {
 public:
    MySQLMetaImpl(const DBMetaOptions& options, const int& mode);
    ~MySQLMetaImpl();

    Status
    CreateTable(TableSchema& table_schema) override;

    Status
    DescribeTable(TableSchema& table_schema) override;

    Status
    HasTable(const std::string& table_id, bool& has_or_not) override;

    Status
    AllTables(std::vector<TableSchema>& table_schema_array) override;

    Status
    DeleteTable(const std::string& table_id) override;

    Status
    DeleteTableFiles(const std::string& table_id) override;

    Status
    CreateTableFile(TableFileSchema& file_schema) override;

    Status
    DropPartitionsByDates(const std::string& table_id, const DatesT& dates) override;

    Status
    GetTableFiles(const std::string& table_id, const std::vector<size_t>& ids, TableFilesSchema& table_files) override;

    Status
    FilesByType(const std::string& table_id, const std::vector<int>& file_types,
                std::vector<std::string>& file_ids) override;

    Status
    UpdateTableIndex(const std::string& table_id, const TableIndex& index) override;

    Status
    UpdateTableFlag(const std::string& table_id, int64_t flag) override;

    Status
    DescribeTableIndex(const std::string& table_id, TableIndex& index) override;

    Status
    DropTableIndex(const std::string& table_id) override;

    Status
    UpdateTableFile(TableFileSchema& file_schema) override;

    Status
    UpdateTableFilesToIndex(const std::string& table_id) override;

    Status
    UpdateTableFiles(TableFilesSchema& files) override;

    Status
    FilesToSearch(const std::string& table_id, const std::vector<size_t>& ids, const DatesT& dates,
                  DatePartionedTableFilesSchema& files) override;

    Status
    FilesToMerge(const std::string& table_id, DatePartionedTableFilesSchema& files) override;

    Status
    FilesToIndex(TableFilesSchema&) override;

    Status
    Archive() override;

    Status
    Size(uint64_t& result) override;

    Status
    CleanUp() override;

    Status
    CleanUpFilesWithTTL(uint16_t seconds) override;

    Status
    DropAll() override;

    Status
    Count(const std::string& table_id, uint64_t& result) override;

 private:
    Status
    NextFileId(std::string& file_id);
    Status
    NextTableId(std::string& table_id);
    Status
    DiscardFiles(int64_t to_discard_size);

    void
    ValidateMetaSchema();
    Status
    Initialize();

 private:
    const DBMetaOptions options_;
    const int mode_;

    std::shared_ptr<MySQLConnectionPool> mysql_connection_pool_;
    bool safe_grab_ = false;

    std::mutex genid_mutex_;
    //        std::mutex connectionMutex_;
};  // DBMetaImpl

}  // namespace meta
}  // namespace engine
}  // namespace milvus
