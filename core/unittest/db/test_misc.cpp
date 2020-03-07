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

#include <gtest/gtest.h>

#include <boost/filesystem.hpp>
#include <thread>
#include <vector>

#include "db/IDGenerator.h"
#include "db/IndexFailedChecker.h"
#include "db/OngoingFileChecker.h"
#include "db/Options.h"
#include "db/Utils.h"
#include "db/engine/EngineFactory.h"
#include "db/meta/SqliteMetaImpl.h"
#include "utils/Exception.h"
#include "utils/Status.h"

#include <fiu-local.h>
#include <fiu-control.h>
#include "db/utils.h"

TEST(DBMiscTest, EXCEPTION_TEST) {
    milvus::Exception ex1(100, "error");
    std::string what = ex1.what();
    ASSERT_EQ(what, "error");
    ASSERT_EQ(ex1.code(), 100);

    milvus::InvalidArgumentException ex2;
    ASSERT_EQ(ex2.code(), milvus::SERVER_INVALID_ARGUMENT);
}

TEST(DBMiscTest, OPTIONS_TEST) {
    try {
        milvus::engine::ArchiveConf archive("$$##");
    } catch (std::exception& ex) {
        ASSERT_TRUE(true);
    }

    {
        milvus::engine::ArchiveConf archive("delete", "no");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        milvus::engine::ArchiveConf archive("delete", "1:2");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        milvus::engine::ArchiveConf archive("delete", "1:2:3");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        milvus::engine::ArchiveConf archive("delete");
        milvus::engine::ArchiveConf::CriteriaT criterial = {{"disk", 1024}, {"days", 100}};
        archive.SetCriterias(criterial);

        auto crit = archive.GetCriterias();
        ASSERT_EQ(criterial["disk"], 1024);
        ASSERT_EQ(criterial["days"], 100);
    }
}

TEST(DBMiscTest, META_TEST) {
    milvus::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test";
    milvus::engine::meta::SqliteMetaImpl impl(options);

    time_t tt;
    time(&tt);
    int delta = 10;
    milvus::engine::meta::DateT dt = milvus::engine::utils::GetDate(tt, delta);
    ASSERT_GT(dt, 0);
}

TEST(DBMiscTest, UTILS_TEST) {
    milvus::engine::DBMetaOptions options;
    options.path_ = "/tmp/milvus_test/main";
    options.slave_paths_.push_back("/tmp/milvus_test/slave_1");
    options.slave_paths_.push_back("/tmp/milvus_test/slave_2");

    const std::string TABLE_NAME = "test_tbl";

    fiu_init(0);
    milvus::Status status;
    FIU_ENABLE_FIU("CommonUtil.CreateDirectory.create_parent_fail");
    status = milvus::engine::utils::CreateTablePath(options, TABLE_NAME);
    ASSERT_FALSE(status.ok());
    fiu_disable("CommonUtil.CreateDirectory.create_parent_fail");

    FIU_ENABLE_FIU("CreateTablePath.creat_slave_path");
    status = milvus::engine::utils::CreateTablePath(options, TABLE_NAME);
    ASSERT_FALSE(status.ok());
    fiu_disable("CreateTablePath.creat_slave_path");

    status = milvus::engine::utils::CreateTablePath(options, TABLE_NAME);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(boost::filesystem::exists(options.path_));
    for (auto& path : options.slave_paths_) {
        ASSERT_TRUE(boost::filesystem::exists(path));
    }

    //    options.slave_paths.push_back("/");
    //    status =  engine::utils::CreateTablePath(options, TABLE_NAME);
    //    ASSERT_FALSE(status.ok());
    //
    //    options.path = "/";
    //    status =  engine::utils::CreateTablePath(options, TABLE_NAME);
    //    ASSERT_FALSE(status.ok());

    milvus::engine::meta::TableFileSchema file;
    file.id_ = 50;
    file.table_id_ = TABLE_NAME;
    file.file_type_ = 3;
    file.date_ = 155000;
    status = milvus::engine::utils::GetTableFilePath(options, file);
    ASSERT_TRUE(status.ok());
    ASSERT_FALSE(file.location_.empty());

    status = milvus::engine::utils::DeleteTablePath(options, TABLE_NAME);
    ASSERT_TRUE(status.ok());

    status = milvus::engine::utils::DeleteTableFilePath(options, file);
    ASSERT_TRUE(status.ok());

    status = milvus::engine::utils::CreateTableFilePath(options, file);
    ASSERT_TRUE(status.ok());

    FIU_ENABLE_FIU("CreateTableFilePath.fail_create");
    status = milvus::engine::utils::CreateTableFilePath(options, file);
    ASSERT_FALSE(status.ok());
    fiu_disable("CreateTableFilePath.fail_create");

    status = milvus::engine::utils::GetTableFilePath(options, file);
    ASSERT_FALSE(file.location_.empty());

    FIU_ENABLE_FIU("CommonUtil.CreateDirectory.create_parent_fail");
    status = milvus::engine::utils::GetTableFilePath(options, file);
    ASSERT_FALSE(file.location_.empty());
    fiu_disable("CommonUtil.CreateDirectory.create_parent_fail");

    FIU_ENABLE_FIU("GetTableFilePath.enable_s3");
    status = milvus::engine::utils::GetTableFilePath(options, file);
    ASSERT_FALSE(file.location_.empty());
    fiu_disable("GetTableFilePath.enable_s3");

    status = milvus::engine::utils::DeleteTableFilePath(options, file);

    ASSERT_TRUE(status.ok());

    status = milvus::engine::utils::DeleteSegment(options, file);
}

TEST(DBMiscTest, SAFE_ID_GENERATOR_TEST) {
    milvus::engine::SafeIDGenerator& generator = milvus::engine::SafeIDGenerator::GetInstance();
    size_t n = 1000000;
    milvus::engine::IDNumbers ids;

    milvus::Status status = generator.GetNextIDNumbers(n, ids);
    ASSERT_TRUE(status.ok());

    std::set<int64_t> unique_ids;
    for (size_t i = 0; i < ids.size(); i++) {
        unique_ids.insert(ids[i]);
    }

    ASSERT_EQ(ids.size(), unique_ids.size());
}

TEST(DBMiscTest, CHECKER_TEST) {
    {
        milvus::engine::IndexFailedChecker checker;
        milvus::engine::meta::TableFileSchema schema;
        schema.table_id_ = "aaa";
        schema.file_id_ = "5000";
        checker.MarkFailedIndexFile(schema, "5000 fail");
        schema.table_id_ = "bbb";
        schema.file_id_ = "5001";
        checker.MarkFailedIndexFile(schema, "5001 fail");

        std::string err_msg;
        checker.GetErrMsgForTable("aaa", err_msg);
        ASSERT_EQ(err_msg, "5000 fail");

        schema.table_id_ = "bbb";
        schema.file_id_ = "5002";
        checker.MarkFailedIndexFile(schema, "5002 fail");
        checker.MarkFailedIndexFile(schema, "5002 fail");

        milvus::engine::meta::TableFilesSchema table_files = {schema};
        checker.IgnoreFailedIndexFiles(table_files);
        ASSERT_TRUE(table_files.empty());

        checker.GetErrMsgForTable("bbb", err_msg);
        ASSERT_EQ(err_msg, "5001 fail");

        checker.MarkSucceedIndexFile(schema);
        checker.GetErrMsgForTable("bbb", err_msg);
        ASSERT_EQ(err_msg, "5001 fail");
    }

    {
        milvus::engine::OngoingFileChecker& checker = milvus::engine::OngoingFileChecker::GetInstance();
        milvus::engine::meta::TableFileSchema schema;
        schema.table_id_ = "aaa";
        schema.file_id_ = "5000";
        checker.MarkOngoingFile(schema);

        ASSERT_TRUE(checker.IsIgnored(schema));

        schema.table_id_ = "bbb";
        schema.file_id_ = "5001";
        milvus::engine::meta::TableFilesSchema table_files = {schema};
        checker.MarkOngoingFiles(table_files);

        ASSERT_TRUE(checker.IsIgnored(schema));

        checker.UnmarkOngoingFile(schema);
        ASSERT_FALSE(checker.IsIgnored(schema));

        schema.table_id_ = "aaa";
        schema.file_id_ = "5000";
        checker.UnmarkOngoingFile(schema);
        ASSERT_FALSE(checker.IsIgnored(schema));
    }
}

TEST(DBMiscTest, IDGENERATOR_TEST) {
    milvus::engine::SimpleIDGenerator gen;
    size_t n = 1000000;
    milvus::engine::IDNumbers ids;
    gen.GetNextIDNumbers(n, ids);

    std::set<int64_t> unique_ids;
    for (size_t i = 0; i < ids.size(); i++) {
        unique_ids.insert(ids[i]);
    }

    ASSERT_EQ(ids.size(), unique_ids.size());
}
