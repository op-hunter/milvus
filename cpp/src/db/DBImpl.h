/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/
#pragma once

#include "DB.h"
#include "Types.h"
#include "utils/ThreadPool.h"
#include "src/db/insert/MemManager.h"

#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <thread>
#include <list>
#include <set>
#include "scheduler/context/SearchContext.h"


namespace zilliz {
namespace milvus {
namespace engine {

class Env;

namespace meta {
class Meta;
}

class DBImpl : public DB {
 public:
    explicit DBImpl(const Options &options);
    ~DBImpl();

    Status Start() override;
    Status Stop() override;
    Status DropAll() override;

    Status CreateTable(meta::TableSchema &table_schema) override;

    Status DeleteTable(const std::string &table_id, const meta::DatesT &dates) override;

    Status DescribeTable(meta::TableSchema &table_schema) override;

    Status HasTable(const std::string &table_id, bool &has_or_not) override;

    Status AllTables(std::vector<meta::TableSchema> &table_schema_array) override;

    Status PreloadTable(const std::string &table_id) override;

    Status UpdateTableFlag(const std::string &table_id, int64_t flag);

    Status GetTableRowCount(const std::string &table_id, uint64_t &row_count) override;

    Status InsertVectors(const std::string &table_id, uint64_t n, const float *vectors, IDNumbers &vector_ids) override;

    Status CreateIndex(const std::string& table_id, const TableIndex& index) override;

    Status DescribeIndex(const std::string& table_id, TableIndex& index) override;

    Status DropIndex(const std::string& table_id) override;

    Status Query(const std::string &table_id,
            uint64_t k,
            uint64_t nq,
            uint64_t nprobe,
            const float *vectors,
            QueryResults &results) override;

    Status Query(const std::string &table_id,
          uint64_t k,
          uint64_t nq,
          uint64_t nprobe,
          const float *vectors,
          const meta::DatesT &dates,
          QueryResults &results) override;

    Status Query(const std::string &table_id,
          const std::vector<std::string> &file_ids,
          uint64_t k,
          uint64_t nq,
          uint64_t nprobe,
          const float *vectors,
          const meta::DatesT &dates,
          QueryResults &results) override;

    Status Size(uint64_t &result) override;

 private:
    Status QueryAsync(const std::string &table_id,
                      const meta::TableFilesSchema &files,
                      uint64_t k,
                      uint64_t nq,
                      uint64_t nprobe,
                      const float *vectors,
                      const meta::DatesT &dates,
                      QueryResults &results);

    void BackgroundTimerTask();

    void StartMetricTask();

    void StartCompactionTask();
    Status MergeFiles(const std::string &table_id,
                      const meta::DateT &date,
                      const meta::TableFilesSchema &files);
    Status BackgroundMergeFiles(const std::string &table_id);
    void BackgroundCompaction(std::set<std::string> table_ids);

    void StartBuildIndexTask(bool force=false);
    void BackgroundBuildIndex();

    Status BuildIndex(const meta::TableFileSchema &);

    Status MemSerialize();

 private:
    const Options options_;

    std::atomic<bool> shutting_down_;

    std::thread bg_timer_thread_;

    meta::MetaPtr meta_ptr_;
    MemManagerPtr mem_mgr_;
    std::mutex mem_serialize_mutex_;

    server::ThreadPool compact_thread_pool_;
    std::list<std::future<void>> compact_thread_results_;
    std::set<std::string> compact_table_ids_;

    server::ThreadPool index_thread_pool_;
    std::list<std::future<void>> index_thread_results_;

    std::mutex build_index_mutex_;

}; // DBImpl


} // namespace engine
} // namespace milvus
} // namespace zilliz
