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

#include "db/utils.h"
#include "db/meta/SqliteMetaImpl.h"
#include "db/Utils.h"
#include "db/meta/MetaConsts.h"

#include <gtest/gtest.h>
#include <thread>
#include <stdlib.h>
#include <time.h>

namespace {

namespace ms = milvus;

} // namespace

TEST_F(MetaTest, TABLE_TEST) {
    auto table_id = "meta_test_table";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);
    ASSERT_TRUE(status.ok());

    auto gid = table.id_;
    table.id_ = -1;
    status = impl_->DescribeTable(table);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table.id_, gid);
    ASSERT_EQ(table.table_id_, table_id);

    table.table_id_ = "not_found";
    status = impl_->DescribeTable(table);
    ASSERT_TRUE(!status.ok());

    table.table_id_ = table_id;
    status = impl_->CreateTable(table);
    ASSERT_EQ(status.code(), ms::DB_ALREADY_EXIST);

    table.table_id_ = "";
    status = impl_->CreateTable(table);
    ASSERT_TRUE(status.ok());
}

TEST_F(MetaTest, TABLE_FILE_TEST) {
    auto table_id = "meta_test_table";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    table.dimension_ = 256;
    auto status = impl_->CreateTable(table);

    ms::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;
    status = impl_->CreateTableFile(table_file);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table_file.file_type_, ms::engine::meta::TableFileSchema::NEW);

    uint64_t cnt = 0;
    status = impl_->Count(table_id, cnt);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(cnt, 0UL);

    auto file_id = table_file.file_id_;

    auto new_file_type = ms::engine::meta::TableFileSchema::INDEX;
    table_file.file_type_ = new_file_type;

    status = impl_->UpdateTableFile(table_file);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table_file.file_type_, new_file_type);

    ms::engine::meta::DatesT dates;
    dates.push_back(ms::engine::utils::GetDate());
    status = impl_->DropPartitionsByDates(table_file.table_id_, dates);
    ASSERT_TRUE(status.ok());

    dates.clear();
    for (auto i = 2; i < 10; ++i) {
        dates.push_back(ms::engine::utils::GetDateWithDelta(-1 * i));
    }
    status = impl_->DropPartitionsByDates(table_file.table_id_, dates);
    ASSERT_TRUE(status.ok());

    table_file.date_ = ms::engine::utils::GetDateWithDelta(-2);
    status = impl_->UpdateTableFile(table_file);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table_file.date_, ms::engine::utils::GetDateWithDelta(-2));
    ASSERT_FALSE(table_file.file_type_ == ms::engine::meta::TableFileSchema::TO_DELETE);

    dates.clear();
    dates.push_back(table_file.date_);
    status = impl_->DropPartitionsByDates(table_file.table_id_, dates);
    ASSERT_TRUE(status.ok());

    std::vector<size_t> ids = {table_file.id_};
    ms::engine::meta::TableFilesSchema files;
    status = impl_->GetTableFiles(table_file.table_id_, ids, files);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(files.size(), 0UL);
}

TEST_F(MetaTest, ARCHIVE_TEST_DAYS) {
    srand(time(0));
    ms::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test";
    unsigned int seed = 1;
    int days_num = rand_r(&seed) % 100;
    std::stringstream ss;
    ss << "days:" << days_num;
    options.archive_conf_ = ms::engine::ArchiveConf("delete", ss.str());

    ms::engine::meta::SqliteMetaImpl impl(options);
    auto table_id = "meta_test_table";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl.CreateTable(table);

    ms::engine::meta::TableFilesSchema files;
    ms::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    auto cnt = 100;
    int64_t ts = ms::engine::utils::GetMicroSecTimeStamp();
    std::vector<int> days;
    std::vector<size_t> ids;
    for (auto i = 0; i < cnt; ++i) {
        status = impl.CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW;
        int day = rand_r(&seed) % (days_num * 2);
        table_file.created_on_ = ts - day * ms::engine::meta::D_SEC * ms::engine::meta::US_PS - 10000;
        status = impl.UpdateTableFile(table_file);
        files.push_back(table_file);
        days.push_back(day);
        ids.push_back(table_file.id_);
    }

    impl.Archive();
    int i = 0;

    ms::engine::meta::TableFilesSchema files_get;
    status = impl.GetTableFiles(table_file.table_id_, ids, files_get);
    ASSERT_TRUE(status.ok());

    for (auto &file : files_get) {
        if (days[i] < days_num) {
            ASSERT_EQ(file.file_type_, ms::engine::meta::TableFileSchema::NEW);
        }
        i++;
    }

    impl.DropAll();
}

TEST_F(MetaTest, ARCHIVE_TEST_DISK) {
    ms::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test";
    options.archive_conf_ = ms::engine::ArchiveConf("delete", "disk:11");

    ms::engine::meta::SqliteMetaImpl impl(options);
    auto table_id = "meta_test_group";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl.CreateTable(table);

    ms::engine::meta::TableFilesSchema files;
    ms::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    auto cnt = 10;
    auto each_size = 2UL;
    std::vector<size_t> ids;
    for (auto i = 0; i < cnt; ++i) {
        status = impl.CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW;
        table_file.file_size_ = each_size * ms::engine::meta::G;
        status = impl.UpdateTableFile(table_file);
        files.push_back(table_file);
        ids.push_back(table_file.id_);
    }

    impl.Archive();
    int i = 0;

    ms::engine::meta::TableFilesSchema files_get;
    status = impl.GetTableFiles(table_file.table_id_, ids, files_get);
    ASSERT_TRUE(status.ok());

    for (auto &file : files_get) {
        if (i >= 5) {
            ASSERT_EQ(file.file_type_, ms::engine::meta::TableFileSchema::NEW);
        }
        ++i;
    }

    impl.DropAll();
}

TEST_F(MetaTest, TABLE_FILES_TEST) {
    auto table_id = "meta_test_group";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);

    uint64_t new_merge_files_cnt = 1;
    uint64_t new_index_files_cnt = 2;
    uint64_t backup_files_cnt = 3;
    uint64_t new_files_cnt = 4;
    uint64_t raw_files_cnt = 5;
    uint64_t to_index_files_cnt = 6;
    uint64_t index_files_cnt = 7;

    ms::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    for (auto i = 0; i < new_merge_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW_MERGE;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < new_index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW_INDEX;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < backup_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::BACKUP;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < new_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < raw_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::RAW;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < to_index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::TO_INDEX;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = ms::engine::meta::TableFileSchema::INDEX;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    uint64_t total_row_count = 0;
    status = impl_->Count(table_id, total_row_count);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(total_row_count, raw_files_cnt + to_index_files_cnt + index_files_cnt);

    ms::engine::meta::TableFilesSchema files;
    status = impl_->FilesToIndex(files);
    ASSERT_EQ(files.size(), to_index_files_cnt);

    ms::engine::meta::DatePartionedTableFilesSchema dated_files;
    status = impl_->FilesToMerge(table.table_id_, dated_files);
    ASSERT_EQ(dated_files[table_file.date_].size(), raw_files_cnt);

    status = impl_->FilesToIndex(files);
    ASSERT_EQ(files.size(), to_index_files_cnt);

    ms::engine::meta::DatesT dates = {table_file.date_};
    std::vector<size_t> ids;
    status = impl_->FilesToSearch(table_id, ids, dates, dated_files);
    ASSERT_EQ(dated_files[table_file.date_].size(),
              to_index_files_cnt + raw_files_cnt + index_files_cnt);

    status = impl_->FilesToSearch(table_id, ids, ms::engine::meta::DatesT(), dated_files);
    ASSERT_EQ(dated_files[table_file.date_].size(),
              to_index_files_cnt + raw_files_cnt + index_files_cnt);

    status = impl_->FilesToSearch(table_id, ids, ms::engine::meta::DatesT(), dated_files);
    ASSERT_EQ(dated_files[table_file.date_].size(),
              to_index_files_cnt + raw_files_cnt + index_files_cnt);

    ids.push_back(size_t(9999999999));
    status = impl_->FilesToSearch(table_id, ids, dates, dated_files);
    ASSERT_EQ(dated_files[table_file.date_].size(), 0);

    std::vector<int> file_types;
    std::vector<std::string> file_ids;
    status = impl_->FilesByType(table.table_id_, file_types, file_ids);
    ASSERT_TRUE(file_ids.empty());
    ASSERT_FALSE(status.ok());

    file_types = {
        ms::engine::meta::TableFileSchema::NEW,
        ms::engine::meta::TableFileSchema::NEW_MERGE,
        ms::engine::meta::TableFileSchema::NEW_INDEX,
        ms::engine::meta::TableFileSchema::TO_INDEX,
        ms::engine::meta::TableFileSchema::INDEX,
        ms::engine::meta::TableFileSchema::RAW,
        ms::engine::meta::TableFileSchema::BACKUP,
    };
    status = impl_->FilesByType(table.table_id_, file_types, file_ids);
    ASSERT_TRUE(status.ok());
    uint64_t total_cnt = new_index_files_cnt + new_merge_files_cnt +
        backup_files_cnt + new_files_cnt + raw_files_cnt +
        to_index_files_cnt + index_files_cnt;
    ASSERT_EQ(file_ids.size(), total_cnt);

    status = impl_->DeleteTableFiles(table_id);
    ASSERT_TRUE(status.ok());

    status = impl_->CreateTableFile(table_file);
    table_file.file_type_ = ms::engine::meta::TableFileSchema::NEW;
    status = impl_->UpdateTableFile(table_file);
    status = impl_->CleanUp();
    ASSERT_TRUE(status.ok());

    status = impl_->DeleteTable(table_id);
    ASSERT_TRUE(status.ok());

    status = impl_->CleanUpFilesWithTTL(1UL);
    ASSERT_TRUE(status.ok());
}

TEST_F(MetaTest, INDEX_TEST) {
    auto table_id = "index_test";

    ms::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);

    ms::engine::TableIndex index;
    index.metric_type_ = 2;
    index.nlist_ = 1234;
    index.engine_type_ = 3;
    status = impl_->UpdateTableIndex(table_id, index);
    ASSERT_TRUE(status.ok());

    int64_t flag = 65536;
    status = impl_->UpdateTableFlag(table_id, flag);
    ASSERT_TRUE(status.ok());

    ms::engine::meta::TableSchema table_info;
    table_info.table_id_ = table_id;
    status = impl_->DescribeTable(table_info);
    ASSERT_EQ(table_info.flag_, flag);

    ms::engine::TableIndex index_out;
    status = impl_->DescribeTableIndex(table_id, index_out);
    ASSERT_EQ(index_out.metric_type_, index.metric_type_);
    ASSERT_EQ(index_out.nlist_, index.nlist_);
    ASSERT_EQ(index_out.engine_type_, index.engine_type_);

    status = impl_->DropTableIndex(table_id);
    ASSERT_TRUE(status.ok());
    status = impl_->DescribeTableIndex(table_id, index_out);
    ASSERT_NE(index_out.metric_type_, index.metric_type_);
    ASSERT_NE(index_out.nlist_, index.nlist_);
    ASSERT_NE(index_out.engine_type_, index.engine_type_);

    status = impl_->UpdateTableFilesToIndex(table_id);
    ASSERT_TRUE(status.ok());
}
