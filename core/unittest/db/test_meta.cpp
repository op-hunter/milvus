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

#include "db/Constants.h"
#include "db/Utils.h"
#include "db/meta/MetaConsts.h"
#include "db/meta/SqliteMetaImpl.h"
#include "db/utils.h"

#include <gtest/gtest.h>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <fiu-local.h>
#include <fiu-control.h>
#include <boost/filesystem/operations.hpp>
#include "src/db/OngoingFileChecker.h"

TEST_F(MetaTest, TABLE_TEST) {
    auto table_id = "meta_test_table";

    milvus::engine::meta::TableSchema table;
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
    ASSERT_EQ(status.code(), milvus::DB_ALREADY_EXIST);

    status = impl_->DropTable(table.table_id_);
    ASSERT_TRUE(status.ok());

    status = impl_->CreateTable(table);
    ASSERT_EQ(status.code(), milvus::DB_ERROR);

    table.table_id_ = "";
    status = impl_->CreateTable(table);
    ASSERT_TRUE(status.ok());
}

TEST_F(MetaTest, FALID_TEST) {
    fiu_init(0);
    auto options = GetOptions();
    auto table_id = "meta_test_table";
    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    milvus::Status status;

    {
        FIU_ENABLE_FIU("SqliteMetaImpl.ValidateMetaSchema.NullConnection");
        milvus::engine::meta::SqliteMetaImpl impl(options.meta_);
        fiu_disable("SqliteMetaImpl.ValidateMetaSchema.NullConnection");
    }
    {
        //failed initialize
        auto options_1 = options;
        options_1.meta_.path_ = options.meta_.path_ + "1";
        if (boost::filesystem::is_directory(options_1.meta_.path_)) {
            boost::filesystem::remove_all(options_1.meta_.path_);
        }

        FIU_ENABLE_FIU("SqliteMetaImpl.Initialize.fail_create_directory");
        ASSERT_ANY_THROW(milvus::engine::meta::SqliteMetaImpl impl(options_1.meta_));
        fiu_disable("SqliteMetaImpl.Initialize.fail_create_directory");

        boost::filesystem::remove_all(options_1.meta_.path_);
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.CreateTable.throw_exception");
        status = impl_->CreateTable(table);
        ASSERT_FALSE(status.ok());
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CreateTable.throw_exception");

        FIU_ENABLE_FIU("SqliteMetaImpl.CreateTable.insert_throw_exception");
        table.table_id_ = "";
        status = impl_->CreateTable(table);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.CreateTable.insert_throw_exception");

        //success create table
        table.table_id_ = table_id;
        status = impl_->CreateTable(table);
        ASSERT_TRUE(status.ok());
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.DescribeTable.throw_exception");
        status = impl_->DescribeTable(table);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.DescribeTable.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.HasTable.throw_exception");
        bool has = false;
        status = impl_->HasTable(table.table_id_, has);
        ASSERT_FALSE(status.ok());
        ASSERT_FALSE(has);
        fiu_disable("SqliteMetaImpl.HasTable.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.AllTables.throw_exception");
        std::vector<milvus::engine::meta::TableSchema> table_schema_array;
        status = impl_->AllTables(table_schema_array);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.AllTables.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.DropTable.throw_exception");
        status = impl_->DropTable(table.table_id_);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.DropTable.throw_exception");
    }
    {
        milvus::engine::meta::TableFileSchema schema;
        schema.table_id_ = "notexist";
        status = impl_->CreateTableFile(schema);
        ASSERT_FALSE(status.ok());

        FIU_ENABLE_FIU("SqliteMetaImpl.CreateTableFile.throw_exception");
        schema.table_id_ = table_id;
        status = impl_->CreateTableFile(schema);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.CreateTableFile.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.DeleteTableFiles.throw_exception");
        status = impl_->DeleteTableFiles(table.table_id_);
        ASSERT_FALSE(status.ok());
        fiu_disable("SqliteMetaImpl.DeleteTableFiles.throw_exception");
    }
    {
        milvus::engine::meta::TableFilesSchema schemas;
        std::vector<size_t> ids;
        status = impl_->GetTableFiles("notexist", ids, schemas);
        ASSERT_FALSE(status.ok());

        FIU_ENABLE_FIU("SqliteMetaImpl.GetTableFiles.throw_exception");
        status = impl_->GetTableFiles(table_id, ids, schemas);
        ASSERT_FALSE(status.ok());
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.GetTableFiles.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableFlag.throw_exception");
        status = impl_->UpdateTableFlag(table_id, 0);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableFlag.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableFile.throw_exception");
        milvus::engine::meta::TableFileSchema schema;
        schema.table_id_ = table_id;
        status = impl_->UpdateTableFile(schema);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableFile.throw_exception");

        schema = {};
        schema.table_id_ = "notexist";
        status = impl_->UpdateTableFile(schema);
        ASSERT_TRUE(status.ok());
    }
    {
        milvus::engine::meta::TableFilesSchema schemas;
        milvus::engine::meta::TableFileSchema schema;
        schema.table_id_ = "notexits";
        schemas.emplace_back(schema);
        status = impl_->UpdateTableFiles(schemas);
        ASSERT_TRUE(status.ok());

        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableFiles.throw_exception");
        status = impl_->UpdateTableFiles(schemas);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableFiles.throw_exception");

        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableFiles.fail_commited");
        status = impl_->UpdateTableFiles(schemas);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableFiles.fail_commited");
    }
    {
        milvus::engine::TableIndex index;
        status = impl_->UpdateTableIndex("notexist", index);
        ASSERT_EQ(status.code(), milvus::DB_NOT_FOUND);

        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableIndex.throw_exception");
        status = impl_->UpdateTableIndex("notexist", index);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableIndex.throw_exception");

        FIU_ENABLE_FIU("SqliteMetaImpl.DescribeTableIndex.throw_exception");
        status = impl_->DescribeTableIndex(table_id, index);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.DescribeTableIndex.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.UpdateTableFilesToIndex.throw_exception");
        status = impl_->UpdateTableFilesToIndex(table_id);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.UpdateTableFilesToIndex.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.DropTableIndex.throw_exception");
        status = impl_->DropTableIndex(table_id);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.DropTableIndex.throw_exception");
    }
    {
        std::string partition = "part0";
        std::string partition_tag = "tag0";
        status = impl_->CreatePartition("notexist", partition, partition_tag, 0);
        ASSERT_EQ(status.code(), milvus::DB_NOT_FOUND);

        status = impl_->CreatePartition(table_id, partition, partition_tag, 0);
        ASSERT_TRUE(status.ok());

        partition_tag = "tag1";
        status = impl_->CreatePartition(table_id, partition, partition_tag, 0);
        ASSERT_FALSE(status.ok());

        //create empty name partition
        partition = "";
        status = impl_->CreatePartition(table_id, partition, partition_tag, 0);
        ASSERT_TRUE(status.ok());

        std::vector<milvus::engine::meta::TableSchema> partions_schema;
        status = impl_->ShowPartitions(table_id, partions_schema);
        ASSERT_TRUE(status.ok());
        ASSERT_EQ(partions_schema.size(), 2);

        partions_schema.clear();
        FIU_ENABLE_FIU("SqliteMetaImpl.ShowPartitions.throw_exception");
        status = impl_->ShowPartitions(table_id, partions_schema);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.ShowPartitions.throw_exception");

        std::string partion;
        FIU_ENABLE_FIU("SqliteMetaImpl.GetPartitionName.throw_exception");
        status = impl_->GetPartitionName(table_id, "tag0", partion);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.GetPartitionName.throw_exception");
    }
    {
        milvus::engine::meta::TableFilesSchema table_files;
        status = impl_->FilesToSearch("notexist", table_files);
        ASSERT_EQ(status.code(), milvus::DB_NOT_FOUND);

        FIU_ENABLE_FIU("SqliteMetaImpl.FilesToSearch.throw_exception");
        status = impl_->FilesToSearch(table_id, table_files);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.FilesToSearch.throw_exception");
    }
    {
        milvus::engine::meta::TableFileSchema file;
        file.table_id_ = table_id;
        status = impl_->CreateTableFile(file);
        ASSERT_TRUE(status.ok());
        file.file_type_ = milvus::engine::meta::TableFileSchema::TO_INDEX;
        impl_->UpdateTableFile(file);

        milvus::engine::meta::TableFilesSchema files;
        FIU_ENABLE_FIU("SqliteMetaImpl_FilesToIndex_TableNotFound");
        status = impl_->FilesToIndex(files);
        ASSERT_EQ(status.code(), milvus::DB_NOT_FOUND);
        fiu_disable("SqliteMetaImpl_FilesToIndex_TableNotFound");

        FIU_ENABLE_FIU("SqliteMetaImpl.FilesToIndex.throw_exception");
        status = impl_->FilesToIndex(files);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.FilesToIndex.throw_exception");
    }
    {
        milvus::engine::meta::TableFilesSchema files;
        std::vector<int> file_types;
        file_types.push_back(milvus::engine::meta::TableFileSchema::INDEX);
        FIU_ENABLE_FIU("SqliteMetaImpl.FilesByType.throw_exception");
        status = impl_->FilesByType(table_id, file_types, files);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.FilesByType.throw_exception");
    }
    {
        uint64_t size = 0;
        FIU_ENABLE_FIU("SqliteMetaImpl.Size.throw_exception");
        status = impl_->Size(size);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.Size.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpShadowFiles.fail_commited");
        status = impl_->CleanUpShadowFiles();
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpShadowFiles.fail_commited");

        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpShadowFiles.throw_exception");
        status = impl_->CleanUpShadowFiles();
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpShadowFiles.throw_exception");
    }
    {
        uint64_t count;
        status = impl_->Count("notexist", count);
        ASSERT_EQ(status.code(), milvus::DB_NOT_FOUND);

        FIU_ENABLE_FIU("SqliteMetaImpl.Count.throw_exception");
        status = impl_->Count("notexist", count);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.Count.throw_exception");
    }
    {
        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveFile_ThrowException");
        status = impl_->CleanUpFilesWithTTL(1);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveFile_ThrowException");

        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveFile_FailCommited");
        status = impl_->CleanUpFilesWithTTL(1);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveFile_FailCommited");

        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTable_Failcommited");
        status = impl_->CleanUpFilesWithTTL(1);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTable_Failcommited");

        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTable_ThrowException");
        status = impl_->CleanUpFilesWithTTL(1);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTable_ThrowException");

        FIU_ENABLE_FIU("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTableFolder_ThrowException");
        status = impl_->CleanUpFilesWithTTL(1);
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.CleanUpFilesWithTTL.RemoveTableFolder_ThrowException");
    }
}

TEST_F(MetaTest, TABLE_FILE_TEST) {
    auto table_id = "meta_test_table";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    table.dimension_ = 256;
    auto status = impl_->CreateTable(table);

    milvus::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;
    status = impl_->CreateTableFile(table_file);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table_file.file_type_, milvus::engine::meta::TableFileSchema::NEW);

    uint64_t cnt = 0;
    status = impl_->Count(table_id, cnt);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(cnt, 0UL);

    auto file_id = table_file.file_id_;

    auto new_file_type = milvus::engine::meta::TableFileSchema::INDEX;
    table_file.file_type_ = new_file_type;

    status = impl_->UpdateTableFile(table_file);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(table_file.file_type_, new_file_type);
}

TEST_F(MetaTest, TABLE_FILE_ROW_COUNT_TEST) {
    auto table_id = "row_count_test_table";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    table.dimension_ = 256;
    auto status = impl_->CreateTable(table);

    milvus::engine::meta::TableFileSchema table_file;
    table_file.row_count_ = 100;
    table_file.table_id_ = table.table_id_;
    table_file.file_type_ = 1;
    status = impl_->CreateTableFile(table_file);

    uint64_t cnt = 0;
    status = impl_->Count(table_id, cnt);
    ASSERT_EQ(table_file.row_count_, cnt);

    table_file.row_count_ = 99999;
    milvus::engine::meta::TableFilesSchema table_files = {table_file};
    status = impl_->UpdateTableFilesRowCount(table_files);
    ASSERT_TRUE(status.ok());

    cnt = 0;
    status = impl_->Count(table_id, cnt);
    ASSERT_EQ(table_file.row_count_, cnt);

    std::vector<size_t> ids = {table_file.id_};
    milvus::engine::meta::TableFilesSchema schemas;
    status = impl_->GetTableFiles(table_id, ids, schemas);
    ASSERT_EQ(schemas.size(), 1UL);
    ASSERT_EQ(table_file.row_count_, schemas[0].row_count_);
    ASSERT_EQ(table_file.file_id_, schemas[0].file_id_);
    ASSERT_EQ(table_file.file_type_, schemas[0].file_type_);
    ASSERT_EQ(table_file.segment_id_, schemas[0].segment_id_);
    ASSERT_EQ(table_file.table_id_, schemas[0].table_id_);
    ASSERT_EQ(table_file.engine_type_, schemas[0].engine_type_);
    ASSERT_EQ(table_file.dimension_, schemas[0].dimension_);
    ASSERT_EQ(table_file.flush_lsn_, schemas[0].flush_lsn_);
}

TEST_F(MetaTest, ARCHIVE_TEST_DAYS) {
    srand(time(0));
    milvus::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test";
    unsigned int seed = 1;
    int days_num = rand_r(&seed) % 100;
    std::stringstream ss;
    ss << "days:" << days_num;
    options.archive_conf_ = milvus::engine::ArchiveConf("delete", ss.str());

    milvus::engine::meta::SqliteMetaImpl impl(options);
    auto table_id = "meta_test_table";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl.CreateTable(table);

    milvus::engine::meta::TableFilesSchema files;
    milvus::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    auto cnt = 100;
    int64_t ts = milvus::engine::utils::GetMicroSecTimeStamp();
    std::vector<int> days;
    std::vector<size_t> ids;
    for (auto i = 0; i < cnt; ++i) {
        status = impl.CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW;
        int day = rand_r(&seed) % (days_num * 2);
        table_file.created_on_ = ts - day * milvus::engine::meta::DAY * milvus::engine::meta::US_PS - 10000;
        status = impl.UpdateTableFile(table_file);
        files.push_back(table_file);
        days.push_back(day);
        ids.push_back(table_file.id_);
    }

    {
        fiu_init(0);
        FIU_ENABLE_FIU("SqliteMetaImpl.Archive.throw_exception");
        status = impl.Archive();
        ASSERT_EQ(status.code(), milvus::DB_META_TRANSACTION_FAILED);
        fiu_disable("SqliteMetaImpl.Archive.throw_exception");
    }

    impl.Archive();
    int i = 0;

    milvus::engine::meta::TableFilesSchema files_get;
    status = impl.GetTableFiles(table_file.table_id_, ids, files_get);
    ASSERT_TRUE(status.ok());

    for (auto& file : files_get) {
        if (days[i] < days_num) {
            ASSERT_EQ(file.file_type_, milvus::engine::meta::TableFileSchema::NEW);
        }
        i++;
    }

    impl.DropAll();
}

TEST_F(MetaTest, ARCHIVE_TEST_DISK) {
    milvus::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test";
    options.archive_conf_ = milvus::engine::ArchiveConf("delete", "disk:11");

    milvus::engine::meta::SqliteMetaImpl impl(options);
    auto table_id = "meta_test_group";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl.CreateTable(table);

    milvus::engine::meta::TableFilesSchema files;
    milvus::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    auto cnt = 10;
    auto each_size = 2UL;
    std::vector<size_t> ids;
    for (auto i = 0; i < cnt; ++i) {
        status = impl.CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW;
        table_file.file_size_ = each_size * milvus::engine::G;
        status = impl.UpdateTableFile(table_file);
        files.push_back(table_file);
        ids.push_back(table_file.id_);
    }

    {
        fiu_init(0);
        FIU_ENABLE_FIU("SqliteMetaImpl.DiscardFiles.throw_exception");
        status = impl.Archive();
        fiu_disable("SqliteMetaImpl.DiscardFiles.throw_exception");

        FIU_ENABLE_FIU("SqliteMetaImpl.DiscardFiles.fail_commited");
        status = impl.Archive();
        fiu_disable("SqliteMetaImpl.DiscardFiles.fail_commited");
    }

    impl.Archive();
    int i = 0;

    milvus::engine::meta::TableFilesSchema files_get;
    status = impl.GetTableFiles(table_file.table_id_, ids, files_get);
    ASSERT_TRUE(status.ok());

    for (auto& file : files_get) {
        if (i >= 5) {
            ASSERT_EQ(file.file_type_, milvus::engine::meta::TableFileSchema::NEW);
        }
        ++i;
    }

    impl.DropAll();
}

TEST_F(MetaTest, TABLE_FILES_TEST) {
    auto table_id = "meta_test_group";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);

    uint64_t new_merge_files_cnt = 1;
    uint64_t new_index_files_cnt = 2;
    uint64_t backup_files_cnt = 3;
    uint64_t new_files_cnt = 4;
    uint64_t raw_files_cnt = 5;
    uint64_t to_index_files_cnt = 6;
    uint64_t index_files_cnt = 7;

    milvus::engine::meta::TableFileSchema table_file;
    table_file.table_id_ = table.table_id_;

    for (auto i = 0; i < new_merge_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW_MERGE;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < new_index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW_INDEX;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < backup_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::BACKUP;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < new_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < raw_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::RAW;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < to_index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::TO_INDEX;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    for (auto i = 0; i < index_files_cnt; ++i) {
        status = impl_->CreateTableFile(table_file);
        table_file.file_type_ = milvus::engine::meta::TableFileSchema::INDEX;
        table_file.row_count_ = 1;
        status = impl_->UpdateTableFile(table_file);
    }

    uint64_t total_row_count = 0;
    status = impl_->Count(table_id, total_row_count);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(total_row_count, raw_files_cnt + to_index_files_cnt + index_files_cnt);

    milvus::engine::meta::TableFilesSchema files;
    status = impl_->FilesToIndex(files);
    ASSERT_EQ(files.size(), to_index_files_cnt);

    milvus::engine::meta::TableFilesSchema table_files;
    status = impl_->FilesToMerge(table.table_id_, table_files);
    ASSERT_EQ(table_files.size(), raw_files_cnt);

    status = impl_->FilesToIndex(files);
    ASSERT_EQ(files.size(), to_index_files_cnt);

    table_files.clear();
    status = impl_->FilesToSearch(table_id, table_files);
    ASSERT_EQ(table_files.size(), to_index_files_cnt + raw_files_cnt + index_files_cnt);

    std::vector<size_t> ids;
    for (auto& file : table_files) {
        ids.push_back(file.id_);
    }
    size_t cnt = table_files.size();
    table_files.clear();
    status = impl_->FilesByID(ids, table_files);
    ASSERT_EQ(table_files.size(), cnt);

    table_files.clear();
    ids = {9999999999UL};
    status = impl_->FilesByID(ids, table_files);
    ASSERT_EQ(table_files.size(), 0);

    table_files.clear();
    std::vector<int> file_types;
    status = impl_->FilesByType(table.table_id_, file_types, table_files);
    ASSERT_TRUE(table_files.empty());
    ASSERT_FALSE(status.ok());

    file_types = {
        milvus::engine::meta::TableFileSchema::NEW, milvus::engine::meta::TableFileSchema::NEW_MERGE,
        milvus::engine::meta::TableFileSchema::NEW_INDEX, milvus::engine::meta::TableFileSchema::TO_INDEX,
        milvus::engine::meta::TableFileSchema::INDEX, milvus::engine::meta::TableFileSchema::RAW,
        milvus::engine::meta::TableFileSchema::BACKUP,
    };
    status = impl_->FilesByType(table.table_id_, file_types, table_files);
    ASSERT_TRUE(status.ok());
    uint64_t total_cnt = new_index_files_cnt + new_merge_files_cnt + backup_files_cnt + new_files_cnt + raw_files_cnt +
                         to_index_files_cnt + index_files_cnt;
    ASSERT_EQ(table_files.size(), total_cnt);

    status = impl_->DeleteTableFiles(table_id);
    ASSERT_TRUE(status.ok());

    status = impl_->CreateTableFile(table_file);
    table_file.file_type_ = milvus::engine::meta::TableFileSchema::NEW;
    status = impl_->UpdateTableFile(table_file);
    status = impl_->CleanUpShadowFiles();
    ASSERT_TRUE(status.ok());

    table_file.table_id_ = table.table_id_;
    table_file.file_type_ = milvus::engine::meta::TableFileSchema::TO_DELETE;
    status = impl_->CreateTableFile(table_file);

    std::vector<int> files_to_delete;
    milvus::engine::meta::TableFilesSchema files_schema;
    files_to_delete.push_back(milvus::engine::meta::TableFileSchema::TO_DELETE);
    status = impl_->FilesByType(table_id, files_to_delete, files_schema);
    ASSERT_TRUE(status.ok());

    table_file.table_id_ = table_id;
    table_file.file_type_ = milvus::engine::meta::TableFileSchema::TO_DELETE;
    table_file.file_id_ = files_schema.front().file_id_;
    milvus::engine::OngoingFileChecker::GetInstance().MarkOngoingFile(table_file);
    status = impl_->CleanUpFilesWithTTL(1UL);
    ASSERT_TRUE(status.ok());

    status = impl_->DropTable(table_id);
    ASSERT_TRUE(status.ok());
}

TEST_F(MetaTest, INDEX_TEST) {
    auto table_id = "index_test";

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);

    milvus::engine::TableIndex index;
    index.metric_type_ = 2;
    index.extra_params_ = {{"nlist", 1234}};
    index.engine_type_ = 3;
    status = impl_->UpdateTableIndex(table_id, index);
    ASSERT_TRUE(status.ok());

    int64_t flag = 65536;
    status = impl_->UpdateTableFlag(table_id, flag);
    ASSERT_TRUE(status.ok());

    milvus::engine::meta::TableSchema table_info;
    table_info.table_id_ = table_id;
    status = impl_->DescribeTable(table_info);
    ASSERT_EQ(table_info.flag_, flag);

    milvus::engine::TableIndex index_out;
    status = impl_->DescribeTableIndex(table_id, index_out);
    ASSERT_EQ(index_out.metric_type_, index.metric_type_);
    ASSERT_EQ(index_out.extra_params_, index.extra_params_);
    ASSERT_EQ(index_out.engine_type_, index.engine_type_);

    status = impl_->DropTableIndex(table_id);
    ASSERT_TRUE(status.ok());
    status = impl_->DescribeTableIndex(table_id, index_out);
    ASSERT_EQ(index_out.metric_type_, index.metric_type_);
    ASSERT_NE(index_out.engine_type_, index.engine_type_);

    status = impl_->UpdateTableFilesToIndex(table_id);
    ASSERT_TRUE(status.ok());
}

TEST_F(MetaTest, LSN_TEST) {
    auto table_id = "lsn_test";
    uint64_t lsn = 42949672960;

    milvus::engine::meta::TableSchema table;
    table.table_id_ = table_id;
    auto status = impl_->CreateTable(table);

    status = impl_->UpdateTableFlushLSN(table_id, lsn);
    ASSERT_TRUE(status.ok());

    uint64_t temp_lsb = 0;
    status = impl_->GetTableFlushLSN(table_id, temp_lsb);
    ASSERT_EQ(temp_lsb, lsn);

    status = impl_->SetGlobalLastLSN(lsn);
    ASSERT_TRUE(status.ok());

    temp_lsb = 0;
    status = impl_->GetGlobalLastLSN(temp_lsb);
    ASSERT_EQ(temp_lsb, lsn);
}
