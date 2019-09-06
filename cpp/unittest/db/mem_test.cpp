#include "gtest/gtest.h"

#include "db/insert/VectorSource.h"
#include "db/insert/MemTableFile.h"
#include "db/insert/MemTable.h"
#include "utils.h"
#include "db/Factories.h"
#include "db/Constants.h"
#include "db/engine/EngineFactory.h"
#include "metrics/Metrics.h"
#include "db/meta/MetaConsts.h"

#include <boost/filesystem.hpp>
#include <thread>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

using namespace zilliz::milvus;

namespace {

static std::string TABLE_NAME = "test_group";
static constexpr int64_t TABLE_DIM = 256;
static constexpr int64_t VECTOR_COUNT = 250000;
static constexpr int64_t INSERT_LOOP = 10000;

std::string GenTableName() {
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
    TABLE_NAME = std::to_string(micros);
    return TABLE_NAME;
}

engine::meta::TableSchema BuildTableSchema() {
    engine::meta::TableSchema table_info;
    table_info.dimension_ = TABLE_DIM;
    table_info.table_id_ = GenTableName();
    table_info.engine_type_ = (int) engine::EngineType::FAISS_IDMAP;
    return table_info;
}

void BuildVectors(int64_t n, std::vector<float> &vectors) {
    vectors.clear();
    vectors.resize(n * TABLE_DIM);
    float *data = vectors.data();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < TABLE_DIM; j++) data[TABLE_DIM * i + j] = drand48();
        data[TABLE_DIM * i] += i / 2000.;
    }
}
}

TEST_F(MemManagerTest, VECTOR_SOURCE_TEST) {

    std::shared_ptr<engine::meta::SqliteMetaImpl> impl_ = engine::DBMetaImplFactory::Build();

    engine::meta::TableSchema table_schema = BuildTableSchema();
    auto status = impl_->CreateTable(table_schema);
    ASSERT_TRUE(status.ok());

    engine::meta::TableFileSchema table_file_schema;
    table_file_schema.table_id_ = TABLE_NAME;
    status = impl_->CreateTableFile(table_file_schema);
    ASSERT_TRUE(status.ok());

    int64_t n = 100;
    std::vector<float> vectors;
    BuildVectors(n, vectors);

    engine::VectorSource source(n, vectors.data());

    size_t num_vectors_added;
    engine::ExecutionEnginePtr execution_engine_ =
            engine::EngineFactory::Build(table_file_schema.dimension_,
                    table_file_schema.location_,
                    (engine::EngineType) table_file_schema.engine_type_,
                    (engine::MetricType)table_file_schema.metric_type_,
                    table_schema.nlist_);

    engine::IDNumbers vector_ids;
    status = source.Add(execution_engine_, table_file_schema, 50, num_vectors_added, vector_ids);
    ASSERT_TRUE(status.ok());
    vector_ids = source.GetVectorIds();
    ASSERT_EQ(vector_ids.size(), 50);
    ASSERT_EQ(num_vectors_added, 50);

    vector_ids.clear();
    status = source.Add(execution_engine_, table_file_schema, 60, num_vectors_added, vector_ids);
    ASSERT_TRUE(status.ok());

    ASSERT_EQ(num_vectors_added, 50);

    vector_ids = source.GetVectorIds();
    ASSERT_EQ(vector_ids.size(), 100);


    status = impl_->DropAll();
    ASSERT_TRUE(status.ok());
}

TEST_F(MemManagerTest, MEM_TABLE_FILE_TEST) {

    std::shared_ptr<engine::meta::SqliteMetaImpl> impl_ = engine::DBMetaImplFactory::Build();
    auto options = engine::OptionsFactory::Build();

    engine::meta::TableSchema table_schema = BuildTableSchema();
    auto status = impl_->CreateTable(table_schema);
    ASSERT_TRUE(status.ok());

    engine::MemTableFile mem_table_file(TABLE_NAME, impl_, options);

    int64_t n_100 = 100;
    std::vector<float> vectors_100;
    BuildVectors(n_100, vectors_100);

    engine::VectorSource::Ptr source = std::make_shared<engine::VectorSource>(n_100, vectors_100.data());

    engine::IDNumbers vector_ids;
    status = mem_table_file.Add(source, vector_ids);
    ASSERT_TRUE(status.ok());

//    std::cout << mem_table_file.GetCurrentMem() << " " << mem_table_file.GetMemLeft() << std::endl;

    vector_ids = source->GetVectorIds();
    ASSERT_EQ(vector_ids.size(), 100);

    size_t singleVectorMem = sizeof(float) * TABLE_DIM;
    ASSERT_EQ(mem_table_file.GetCurrentMem(), n_100 * singleVectorMem);

    int64_t n_max = engine::MAX_TABLE_FILE_MEM / singleVectorMem;
    std::vector<float> vectors_128M;
    BuildVectors(n_max, vectors_128M);

    engine::VectorSource::Ptr source_128M = std::make_shared<engine::VectorSource>(n_max, vectors_128M.data());
    vector_ids.clear();
    status = mem_table_file.Add(source_128M, vector_ids);

    vector_ids = source_128M->GetVectorIds();
    ASSERT_EQ(vector_ids.size(), n_max - n_100);

    ASSERT_TRUE(mem_table_file.IsFull());

    status = impl_->DropAll();
    ASSERT_TRUE(status.ok());
}

TEST_F(MemManagerTest, MEM_TABLE_TEST) {

    std::shared_ptr<engine::meta::SqliteMetaImpl> impl_ = engine::DBMetaImplFactory::Build();
    auto options = engine::OptionsFactory::Build();

    engine::meta::TableSchema table_schema = BuildTableSchema();
    auto status = impl_->CreateTable(table_schema);
    ASSERT_TRUE(status.ok());

    int64_t n_100 = 100;
    std::vector<float> vectors_100;
    BuildVectors(n_100, vectors_100);

    engine::VectorSource::Ptr source_100 = std::make_shared<engine::VectorSource>(n_100, vectors_100.data());

    engine::MemTable mem_table(TABLE_NAME, impl_, options);

    engine::IDNumbers vector_ids;
    status = mem_table.Add(source_100, vector_ids);
    ASSERT_TRUE(status.ok());
    vector_ids = source_100->GetVectorIds();
    ASSERT_EQ(vector_ids.size(), 100);

    engine::MemTableFile::Ptr mem_table_file;
    mem_table.GetCurrentMemTableFile(mem_table_file);
    size_t singleVectorMem = sizeof(float) * TABLE_DIM;
    ASSERT_EQ(mem_table_file->GetCurrentMem(), n_100 * singleVectorMem);

    int64_t n_max = engine::MAX_TABLE_FILE_MEM / singleVectorMem;
    std::vector<float> vectors_128M;
    BuildVectors(n_max, vectors_128M);

    vector_ids.clear();
    engine::VectorSource::Ptr source_128M = std::make_shared<engine::VectorSource>(n_max, vectors_128M.data());
    status = mem_table.Add(source_128M, vector_ids);
    ASSERT_TRUE(status.ok());

    vector_ids = source_128M->GetVectorIds();
    ASSERT_EQ(vector_ids.size(), n_max);

    mem_table.GetCurrentMemTableFile(mem_table_file);
    ASSERT_EQ(mem_table_file->GetCurrentMem(), n_100 * singleVectorMem);

    ASSERT_EQ(mem_table.GetTableFileCount(), 2);

    int64_t n_1G = 1024000;
    std::vector<float> vectors_1G;
    BuildVectors(n_1G, vectors_1G);

    engine::VectorSource::Ptr source_1G = std::make_shared<engine::VectorSource>(n_1G, vectors_1G.data());

    vector_ids.clear();
    status = mem_table.Add(source_1G, vector_ids);
    ASSERT_TRUE(status.ok());

    vector_ids = source_1G->GetVectorIds();
    ASSERT_EQ(vector_ids.size(), n_1G);

    int expectedTableFileCount = 2 + std::ceil((n_1G - n_100) * singleVectorMem / engine::MAX_TABLE_FILE_MEM);
    ASSERT_EQ(mem_table.GetTableFileCount(), expectedTableFileCount);

    status = mem_table.Serialize();
    ASSERT_TRUE(status.ok());

    status = impl_->DropAll();
    ASSERT_TRUE(status.ok());
}

TEST_F(MemManagerTest2, SERIAL_INSERT_SEARCH_TEST) {
    engine::meta::TableSchema table_info = BuildTableSchema();
    engine::Status stat = db_->CreateTable(table_info);

    engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_STATS(stat);
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    std::map<int64_t, std::vector<float>> search_vectors;
    {
        engine::IDNumbers vector_ids;
        int64_t nb = 1024000;
        std::vector<float> xb;
        BuildVectors(nb, xb);
        engine::Status status = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
        ASSERT_TRUE(status.ok());

        std::this_thread::sleep_for(std::chrono::seconds(3));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(0, nb - 1);

        int64_t num_query = 20;
        for (int64_t i = 0; i < num_query; ++i) {
            int64_t index = dis(gen);
            std::vector<float> search;
            for (int64_t j = 0; j < TABLE_DIM; j++) {
                search.push_back(xb[index * TABLE_DIM + j]);
            }
            search_vectors.insert(std::make_pair(vector_ids[index], search));
        }
    }

    int k = 10;
    for (auto &pair : search_vectors) {
        auto &search = pair.second;
        engine::QueryResults results;
        stat = db_->Query(TABLE_NAME, k, 1, 10, search.data(), results);
        ASSERT_EQ(results[0][0].first, pair.first);
        ASSERT_LT(results[0][0].second, 0.00001);
    }
}

TEST_F(MemManagerTest2, INSERT_TEST) {
    engine::meta::TableSchema table_info = BuildTableSchema();
    engine::Status stat = db_->CreateTable(table_info);

    engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_STATS(stat);
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    auto start_time = METRICS_NOW_TIME;

    int insert_loop = 20;
    for (int i = 0; i < insert_loop; ++i) {
        int64_t nb = 40960;
        std::vector<float> xb;
        BuildVectors(nb, xb);
        engine::IDNumbers vector_ids;
        engine::Status status = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
        ASSERT_TRUE(status.ok());
    }
    auto end_time = METRICS_NOW_TIME;
    auto total_time = METRICS_MICROSECONDS(start_time, end_time);
    LOG(DEBUG) << "total_time spent in INSERT_TEST (ms) : " << total_time;
}

TEST_F(MemManagerTest2, CONCURRENT_INSERT_SEARCH_TEST) {
    engine::meta::TableSchema table_info = BuildTableSchema();
    engine::Status stat = db_->CreateTable(table_info);

    engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_STATS(stat);
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    engine::IDNumbers vector_ids;
    engine::IDNumbers target_ids;

    int64_t nb = 40960;
    std::vector<float> xb;
    BuildVectors(nb, xb);

    int64_t qb = 5;
    std::vector<float> qxb;
    BuildVectors(qb, qxb);

    std::thread search([&]() {
        engine::QueryResults results;
        int k = 10;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        INIT_TIMER;
        std::stringstream ss;
        uint64_t count = 0;
        uint64_t prev_count = 0;

        for (auto j = 0; j < 10; ++j) {
            ss.str("");
            db_->Size(count);
            prev_count = count;

            START_TIMER;
            stat = db_->Query(TABLE_NAME, k, qb, 10, qxb.data(), results);
            ss << "Search " << j << " With Size " << count / engine::meta::M << " M";
            STOP_TIMER(ss.str());

            ASSERT_STATS(stat);
            for (auto k = 0; k < qb; ++k) {
                ASSERT_EQ(results[k][0].first, target_ids[k]);
                ss.str("");
                ss << "Result [" << k << "]:";
                for (auto result : results[k]) {
                    ss << result.first << " ";
                }
                /* LOG(DEBUG) << ss.str(); */
            }
            ASSERT_TRUE(count >= prev_count);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    int loop = 20;

    for (auto i = 0; i < loop; ++i) {
        if (i == 0) {
            db_->InsertVectors(TABLE_NAME, qb, qxb.data(), target_ids);
            ASSERT_EQ(target_ids.size(), qb);
        } else {
            db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    search.join();
};

TEST_F(MemManagerTest2, VECTOR_IDS_TEST) {
    engine::meta::TableSchema table_info = BuildTableSchema();
    engine::Status stat = db_->CreateTable(table_info);

    engine::meta::TableSchema table_info_get;
    table_info_get.table_id_ = TABLE_NAME;
    stat = db_->DescribeTable(table_info_get);
    ASSERT_STATS(stat);
    ASSERT_EQ(table_info_get.dimension_, TABLE_DIM);

    engine::IDNumbers vector_ids;


    int64_t nb = 100000;
    std::vector<float> xb;
    BuildVectors(nb, xb);

    vector_ids.resize(nb);
    for (auto i = 0; i < nb; i++) {
        vector_ids[i] = i;
    }

    stat = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
    ASSERT_EQ(vector_ids[0], 0);
    ASSERT_STATS(stat);

    nb = 25000;
    xb.clear();
    BuildVectors(nb, xb);
    vector_ids.clear();
    vector_ids.resize(nb);
    for (auto i = 0; i < nb; i++) {
        vector_ids[i] = i + nb;
    }
    stat = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
    ASSERT_EQ(vector_ids[0], nb);
    ASSERT_STATS(stat);

    nb = 262144; //512M
    xb.clear();
    BuildVectors(nb, xb);
    vector_ids.clear();
    vector_ids.resize(nb);
    for (auto i = 0; i < nb; i++) {
        vector_ids[i] = i + nb / 2;
    }
    stat = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
    ASSERT_EQ(vector_ids[0], nb/2);
    ASSERT_STATS(stat);

    nb = 65536; //128M
    xb.clear();
    BuildVectors(nb, xb);
    vector_ids.clear();
    stat = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
    ASSERT_STATS(stat);

    nb = 100;
    xb.clear();
    BuildVectors(nb, xb);
    vector_ids.clear();
    vector_ids.resize(nb);
    for (auto i = 0; i < nb; i++) {
        vector_ids[i] = i + nb;
    }
    stat = db_->InsertVectors(TABLE_NAME, nb, xb.data(), vector_ids);
    for (auto i = 0; i < nb; i++) {
        ASSERT_EQ(vector_ids[i], i + nb);
    }
}
