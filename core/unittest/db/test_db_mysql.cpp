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
#include <fiu-control.h>
#include <fiu-local.h>
#include <boost/filesystem.hpp>
#include <random>
#include <thread>

#include "db/Constants.h"
#include "db/DB.h"
#include "db/DBImpl.h"
#include "db/meta/MetaConsts.h"
#include "db/utils.h"

namespace {

static const char* TABLE_NAME = "test_group";
static constexpr int64_t TABLE_DIM = 256;
static constexpr int64_t VECTOR_COUNT = 25000;
static constexpr int64_t INSERT_LOOP = 1000;

milvus::engine::meta::TableSchema
BuildTableSchema() {
    milvus::engine::meta::TableSchema table_info;
    table_info.dimension_ = TABLE_DIM;
    table_info.table_id_ = TABLE_NAME;
    table_info.engine_type_ = (int)milvus::engine::EngineType::FAISS_IDMAP;
    return table_info;
}

void
BuildVectors(uint64_t n, milvus::engine::VectorsData& vectors) {
    vectors.vector_count_ = n;
    vectors.float_data_.clear();
    vectors.float_data_.resize(n * TABLE_DIM);
    float* data = vectors.float_data_.data();
    for (uint64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < TABLE_DIM; j++) data[TABLE_DIM * i + j] = drand48();
        data[TABLE_DIM * i] += i / 2000.;
    }
}

}  // namespace

TEST_F(MySqlDBTest, DB_TEST) {
    milvus::engine::meta::TableSchema table_info = BuildTableSchema();
    auto stat = db_->CreateTable(table_info);

    milvus::engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    uint64_t nb = 50;
    milvus::engine::VectorsData xb;
    BuildVectors(nb, xb);

    uint64_t qb = 5;
    milvus::engine::VectorsData qxb;
    BuildVectors(qb, qxb);

    db_->InsertVectors(TABLE_NAME, "", qxb);
    ASSERT_EQ(qxb.id_array_.size(), qb);

    std::thread search([&]() {
        milvus::engine::ResultIds result_ids;
        milvus::engine::ResultDistances result_distances;
        int k = 10;
        std::this_thread::sleep_for(std::chrono::seconds(5));

        INIT_TIMER;
        std::stringstream ss;
        uint64_t count = 0;
        uint64_t prev_count = 0;

        for (auto j = 0; j < 10; ++j) {
            ss.str("");
            db_->Size(count);
            prev_count = count;

            START_TIMER;
            std::vector<std::string> tags;
            stat = db_->Query(dummy_context_, TABLE_NAME, tags, k, 10, qxb, result_ids, result_distances);
            ss << "Search " << j << " With Size " << count / milvus::engine::M << " M";
            STOP_TIMER(ss.str());

            ASSERT_TRUE(stat.ok());
            for (auto i = 0; i < qb; ++i) {
                //                std::cout << results[k][0].first << " " << target_ids[k] << std::endl;
                //                ASSERT_EQ(results[k][0].first, target_ids[k]);
                ss.str("");
                ss << "Result [" << i << "]:";
                for (auto t = 0; t < k; t++) {
                    ss << result_ids[i * k + t] << " ";
                }
                /* LOG(DEBUG) << ss.str(); */
            }
            ASSERT_TRUE(count >= prev_count);
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }

        std::cout << "All search done!" << std::endl;
    });

    int loop = INSERT_LOOP;

    for (auto i = 0; i < loop; ++i) {
        //        if (i==10) {
        //            db_->InsertVectors(TABLE_NAME, "", qb, qxb.data(), target_ids);
        //            ASSERT_EQ(target_ids.size(), qb);
        //        } else {
        //            db_->InsertVectors(TABLE_NAME, "", nb, xb.data(), vector_ids);
        //        }
        xb.id_array_.clear();
        db_->InsertVectors(TABLE_NAME, "", xb);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    search.join();
}

TEST_F(MySqlDBTest, SEARCH_TEST) {
    milvus::engine::meta::TableSchema table_info = BuildTableSchema();
    auto stat = db_->CreateTable(table_info);

    milvus::engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    // prepare raw data
    size_t nb = VECTOR_COUNT;
    size_t nq = 10;
    size_t k = 5;
    milvus::engine::VectorsData xb, xq;
    xb.vector_count_ = nb;
    xb.float_data_.resize(nb * TABLE_DIM);
    xq.vector_count_ = nq;
    xq.float_data_.resize(nq * TABLE_DIM);
    xb.id_array_.resize(nb);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_xt(-1.0, 1.0);
    for (size_t i = 0; i < nb * TABLE_DIM; i++) {
        xb.float_data_[i] = dis_xt(gen);
        if (i < nb) {
            xb.id_array_[i] = i;
        }
    }
    for (size_t i = 0; i < nq * TABLE_DIM; i++) {
        xq.float_data_[i] = dis_xt(gen);
    }

    // result data
    // std::vector<long> nns_gt(k*nq);
    std::vector<int64_t> nns(k * nq);  // nns = nearst neg search
    // std::vector<float> dis_gt(k*nq);
    std::vector<float> dis(k * nq);

    // insert data
    stat = db_->InsertVectors(TABLE_NAME, "", xb);
    ASSERT_TRUE(stat.ok());

    sleep(2);  // wait until build index finish

    std::vector<std::string> tags;
    milvus::engine::ResultIds result_ids;
    milvus::engine::ResultDistances result_distances;
    stat = db_->Query(dummy_context_, TABLE_NAME, tags, k, 10, xq, result_ids, result_distances);
    ASSERT_TRUE(stat.ok());
}

TEST_F(MySqlDBTest, ARHIVE_DISK_CHECK) {
    milvus::engine::meta::TableSchema table_info = BuildTableSchema();
    auto stat = db_->CreateTable(table_info);

    std::vector<milvus::engine::meta::TableSchema> table_schema_array;
    stat = db_->AllTables(table_schema_array);
    ASSERT_TRUE(stat.ok());
    bool bfound = false;
    for (auto& schema : table_schema_array) {
        if (schema.table_id_ == TABLE_NAME) {
            bfound = true;
            break;
        }
    }
    ASSERT_TRUE(bfound);

    fiu_init(0);
    FIU_ENABLE_FIU("MySQLMetaImpl.AllTable.null_connection");
    stat = db_->AllTables(table_schema_array);
    ASSERT_FALSE(stat.ok());

    FIU_ENABLE_FIU("MySQLMetaImpl.AllTable.throw_exception");
    stat = db_->AllTables(table_schema_array);
    ASSERT_FALSE(stat.ok());
    fiu_disable("MySQLMetaImpl.AllTable.null_connection");
    fiu_disable("MySQLMetaImpl.AllTable.throw_exception");

    milvus::engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    milvus::engine::IDNumbers vector_ids;
    milvus::engine::IDNumbers target_ids;

    uint64_t size;
    db_->Size(size);

    int64_t nb = 10;
    milvus::engine::VectorsData xb;
    BuildVectors(nb, xb);

    int loop = INSERT_LOOP;
    for (auto i = 0; i < loop; ++i) {
        db_->InsertVectors(TABLE_NAME, "", xb);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    db_->Size(size);
    LOG(DEBUG) << "size=" << size;
    ASSERT_LE(size, 1 * milvus::engine::G);

    FIU_ENABLE_FIU("MySQLMetaImpl.Size.null_connection");
    stat = db_->Size(size);
    ASSERT_FALSE(stat.ok());
    fiu_disable("MySQLMetaImpl.Size.null_connection");
    FIU_ENABLE_FIU("MySQLMetaImpl.Size.throw_exception");
    stat = db_->Size(size);
    ASSERT_FALSE(stat.ok());
    fiu_disable("MySQLMetaImpl.Size.throw_exception");
}

TEST_F(MySqlDBTest, DELETE_TEST) {
    milvus::engine::meta::TableSchema table_info = BuildTableSchema();
    auto stat = db_->CreateTable(table_info);
    //    std::cout << stat.ToString() << std::endl;

    milvus::engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_TRUE(stat.ok());

    bool has_table = false;
    db_->HasTable(TABLE_NAME, has_table);
    ASSERT_TRUE(has_table);

    milvus::engine::IDNumbers vector_ids;

    uint64_t size;
    db_->Size(size);

    int64_t nb = INSERT_LOOP;
    milvus::engine::VectorsData xb;
    BuildVectors(nb, xb);

    int loop = 20;
    for (auto i = 0; i < loop; ++i) {
        db_->InsertVectors(TABLE_NAME, "", xb);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    //    std::vector<engine::meta::DateT> dates;
    //    stat = db_->DropTable(TABLE_NAME, dates);
    ////    std::cout << "5 sec start" << std::endl;
    //    std::this_thread::sleep_for(std::chrono::seconds(5));
    ////    std::cout << "5 sec finish" << std::endl;
    //    ASSERT_TRUE(stat.ok());
    //
    //    db_->HasTable(TABLE_NAME, has_table);
    //    ASSERT_FALSE(has_table);
}

TEST_F(MySqlDBTest, PARTITION_TEST) {
    milvus::engine::meta::TableSchema table_info = BuildTableSchema();
    auto stat = db_->CreateTable(table_info);
    ASSERT_TRUE(stat.ok());

    // create partition and insert data
    const int64_t PARTITION_COUNT = 5;
    const int64_t INSERT_BATCH = 2000;
    std::string table_name = TABLE_NAME;
    for (int64_t i = 0; i < PARTITION_COUNT; i++) {
        std::string partition_tag = std::to_string(i);
        std::string partition_name = table_name + "_" + partition_tag;
        stat = db_->CreatePartition(table_name, partition_name, partition_tag);
        ASSERT_TRUE(stat.ok());

        fiu_init(0);
        FIU_ENABLE_FIU("MySQLMetaImpl.CreatePartition.aleady_exist");
        stat = db_->CreatePartition(table_name, partition_name, partition_tag);
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.CreatePartition.aleady_exist");

        // not allow nested partition
        stat = db_->CreatePartition(partition_name, "dumy", "dummy");
        ASSERT_FALSE(stat.ok());

        // not allow duplicated partition
        stat = db_->CreatePartition(table_name, partition_name, partition_tag);
        ASSERT_FALSE(stat.ok());

        milvus::engine::VectorsData xb;
        BuildVectors(INSERT_BATCH, xb);

        milvus::engine::IDNumbers vector_ids;
        vector_ids.resize(INSERT_BATCH);
        for (int64_t k = 0; k < INSERT_BATCH; k++) {
            vector_ids[k] = i * INSERT_BATCH + k;
        }

        db_->InsertVectors(table_name, partition_tag, xb);
        ASSERT_EQ(vector_ids.size(), INSERT_BATCH);
    }

    // duplicated partition is not allowed
    stat = db_->CreatePartition(table_name, "", "0");
    ASSERT_FALSE(stat.ok());

    std::vector<milvus::engine::meta::TableSchema> partition_schema_array;
    stat = db_->ShowPartitions(table_name, partition_schema_array);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(partition_schema_array.size(), PARTITION_COUNT);
    for (int64_t i = 0; i < PARTITION_COUNT; i++) {
        ASSERT_EQ(partition_schema_array[i].table_id_, table_name + "_" + std::to_string(i));
    }

    {  // build index
        milvus::engine::TableIndex index;
        index.engine_type_ = (int)milvus::engine::EngineType::FAISS_IVFFLAT;
        index.metric_type_ = (int)milvus::engine::MetricType::L2;
        stat = db_->CreateIndex(table_info.table_id_, index);
        ASSERT_TRUE(stat.ok());

        uint64_t row_count = 0;
        stat = db_->GetTableRowCount(TABLE_NAME, row_count);
        ASSERT_TRUE(stat.ok());
        ASSERT_EQ(row_count, INSERT_BATCH * PARTITION_COUNT);
    }

    {  // search
        const int64_t nq = 5;
        const int64_t topk = 10;
        const int64_t nprobe = 10;
        milvus::engine::VectorsData xq;
        BuildVectors(nq, xq);

        // specify partition tags
        std::vector<std::string> tags = {"0", std::to_string(PARTITION_COUNT - 1)};
        milvus::engine::ResultIds result_ids;
        milvus::engine::ResultDistances result_distances;
        stat = db_->Query(dummy_context_, TABLE_NAME, tags, 10, 10, xq, result_ids, result_distances);
        ASSERT_TRUE(stat.ok());
        ASSERT_EQ(result_ids.size() / topk, nq);

        // search in whole table
        tags.clear();
        result_ids.clear();
        result_distances.clear();
        stat = db_->Query(dummy_context_, TABLE_NAME, tags, 10, 10, xq, result_ids, result_distances);
        ASSERT_TRUE(stat.ok());
        ASSERT_EQ(result_ids.size() / topk, nq);

        // search in all partitions(tag regex match)
        tags.push_back("\\d");
        result_ids.clear();
        result_distances.clear();
        stat = db_->Query(dummy_context_, TABLE_NAME, tags, 10, 10, xq, result_ids, result_distances);
        ASSERT_TRUE(stat.ok());
        ASSERT_EQ(result_ids.size() / topk, nq);
    }

    fiu_init(0);
    {
        //create partition with dummy name
        stat = db_->CreatePartition(table_name, "", "6");
        ASSERT_TRUE(stat.ok());

        // ensure DescribeTable failed
        FIU_ENABLE_FIU("MySQLMetaImpl.DescribeTable.throw_exception");
        stat = db_->CreatePartition(table_name, "", "7");
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.DescribeTable.throw_exception");

        //Drop partition will failed,since it firstly drop partition meta table.
        FIU_ENABLE_FIU("MySQLMetaImpl.DropTable.null_connection");
        stat = db_->DropPartition(table_name + "_5");
        //TODO(sjh): add assert expr, since DropPartion always return Status::OK() for now.
        //ASSERT_TRUE(stat.ok());
        fiu_disable("MySQLMetaImpl.DropTable.null_connection");

        std::vector<milvus::engine::meta::TableSchema> partition_schema_array;
        stat = db_->ShowPartitions(table_name, partition_schema_array);
        ASSERT_TRUE(stat.ok());
        ASSERT_EQ(partition_schema_array.size(), PARTITION_COUNT + 1);

        FIU_ENABLE_FIU("MySQLMetaImpl.ShowPartitions.null_connection");
        stat = db_->ShowPartitions(table_name, partition_schema_array);
        ASSERT_FALSE(stat.ok());

        FIU_ENABLE_FIU("MySQLMetaImpl.ShowPartitions.throw_exception");
        stat = db_->ShowPartitions(table_name, partition_schema_array);
        ASSERT_FALSE(stat.ok());

        FIU_ENABLE_FIU("MySQLMetaImpl.DropTable.throw_exception");
        stat = db_->DropPartition(table_name + "_4");
        fiu_disable("MySQLMetaImpl.DropTable.throw_exception");

        stat = db_->DropPartition(table_name + "_0");
        ASSERT_TRUE(stat.ok());
    }

    {
        FIU_ENABLE_FIU("MySQLMetaImpl.GetPartitionName.null_connection");
        stat = db_->DropPartitionByTag(table_name, "1");
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.GetPartitionName.null_connection");

        FIU_ENABLE_FIU("MySQLMetaImpl.GetPartitionName.throw_exception");
        stat = db_->DropPartitionByTag(table_name, "1");
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.GetPartitionName.throw_exception");

        stat = db_->DropPartitionByTag(table_name, "1");
        ASSERT_TRUE(stat.ok());

        stat = db_->CreatePartition(table_name, table_name + "_1", "1");
        FIU_ENABLE_FIU("MySQLMetaImpl.DeleteTableFiles.null_connection");
        stat = db_->DropPartition(table_name + "_1");
        fiu_disable("MySQLMetaImpl.DeleteTableFiles.null_connection");

        FIU_ENABLE_FIU("MySQLMetaImpl.DeleteTableFiles.throw_exception");
        stat = db_->DropPartition(table_name + "_1");
        fiu_disable("MySQLMetaImpl.DeleteTableFiles.throw_exception");
    }

    {
        FIU_ENABLE_FIU("MySQLMetaImpl.DropTableIndex.null_connection");
        stat = db_->DropIndex(table_name);
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.DropTableIndex.null_connection");

        FIU_ENABLE_FIU("MySQLMetaImpl.DropTableIndex.throw_exception");
        stat = db_->DropIndex(table_name);
        ASSERT_FALSE(stat.ok());
        fiu_disable("MySQLMetaImpl.DropTableIndex.throw_exception");

        stat = db_->DropIndex(table_name);
        ASSERT_TRUE(stat.ok());
    }
}


