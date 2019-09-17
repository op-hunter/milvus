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

#include <gtest/gtest.h>
#include <thread>
#include "utils/easylogging++.h"
#include <boost/filesystem.hpp>

#include "db/Exception.h"
#include "db/Status.h"
#include "db/Options.h"
#include "db/meta/SqliteMetaImpl.h"
#include "db/engine/EngineFactory.h"
#include "db/Utils.h"

#include <vector>

using namespace zilliz::milvus;

namespace {
    void CopyStatus(engine::Status& st1, engine::Status& st2) {
        st1 = st2;
    }

}

TEST(DBMiscTest, EXCEPTION_TEST) {
    engine::Exception ex1("");
    std::string what = ex1.what();
    ASSERT_FALSE(what.empty());

    engine::OutOfRangeException ex2;
    what = ex2.what();
    ASSERT_FALSE(what.empty());
}

TEST(DBMiscTest, STATUS_TEST) {
    engine::Status status = engine::Status::OK();
    std::string str = status.ToString();
    ASSERT_FALSE(str.empty());

    status = engine::Status(DB_ERROR, "mistake");
    ASSERT_EQ(status.code(), DB_ERROR);
    str = status.ToString();
    ASSERT_FALSE(str.empty());

    status = engine::Status(DB_NOT_FOUND, "mistake");
    ASSERT_EQ(status.code(), DB_NOT_FOUND);
    str = status.ToString();
    ASSERT_FALSE(str.empty());

    status = engine::Status(DB_ALREADY_EXIST, "mistake");
    ASSERT_EQ(status.code(), DB_ALREADY_EXIST);
    str = status.ToString();
    ASSERT_FALSE(str.empty());

    status = engine::Status(DB_META_TRANSACTION_FAILED, "mistake");
    ASSERT_EQ(status.code(), DB_META_TRANSACTION_FAILED);
    str = status.ToString();
    ASSERT_FALSE(str.empty());

    engine::Status status_copy = engine::Status::OK();
    CopyStatus(status_copy, status);
    ASSERT_EQ(status.code(), DB_META_TRANSACTION_FAILED);
}

TEST(DBMiscTest, OPTIONS_TEST) {
    try {
        engine::ArchiveConf archive("$$##");
    } catch (std::exception& ex) {
        ASSERT_TRUE(true);
    }

    {
        engine::ArchiveConf archive("delete", "no");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        engine::ArchiveConf archive("delete", "1:2");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        engine::ArchiveConf archive("delete", "1:2:3");
        ASSERT_TRUE(archive.GetCriterias().empty());
    }

    {
        engine::ArchiveConf archive("delete");
        engine::ArchiveConf::CriteriaT criterial = {
            {"disk", 1024},
            {"days", 100}
        };
        archive.SetCriterias(criterial);

        auto crit = archive.GetCriterias();
        ASSERT_EQ(criterial["disk"], 1024);
        ASSERT_EQ(criterial["days"], 100);
    }
}

TEST(DBMiscTest, META_TEST) {
    engine::DBMetaOptions options;
    options.path = "/tmp/milvus_test";
    engine::meta::SqliteMetaImpl impl(options);

    time_t tt;
    time( &tt );
    int delta = 10;
    engine::meta::DateT dt = engine::utils::GetDate(tt, delta);
    ASSERT_GT(dt, 0);
}

TEST(DBMiscTest, UTILS_TEST) {
    engine::DBMetaOptions options;
    options.path = "/tmp/milvus_test/main";
    options.slave_paths.push_back("/tmp/milvus_test/slave_1");
    options.slave_paths.push_back("/tmp/milvus_test/slave_2");

    const std::string TABLE_NAME = "test_tbl";
    auto status =  engine::utils::CreateTablePath(options, TABLE_NAME);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(boost::filesystem::exists(options.path));
    for(auto& path : options.slave_paths) {
       ASSERT_TRUE(boost::filesystem::exists(path));
    }

//    options.slave_paths.push_back("/");
//    status =  engine::utils::CreateTablePath(options, TABLE_NAME);
//    ASSERT_FALSE(status.ok());
//
//    options.path = "/";
//    status =  engine::utils::CreateTablePath(options, TABLE_NAME);
//    ASSERT_FALSE(status.ok());

    engine::meta::TableFileSchema file;
    file.id_ = 50;
    file.table_id_ = TABLE_NAME;
    file.file_type_ = 3;
    file.date_ = 155000;
    status = engine::utils::GetTableFilePath(options, file);
    ASSERT_FALSE(status.ok());
    ASSERT_TRUE(file.location_.empty());

    status = engine::utils::DeleteTablePath(options, TABLE_NAME);
    ASSERT_TRUE(status.ok());

    status = engine::utils::DeleteTableFilePath(options, file);
    ASSERT_TRUE(status.ok());
}