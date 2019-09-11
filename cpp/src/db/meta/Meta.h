/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/
#pragma once

#include "MetaTypes.h"
#include "db/Options.h"
#include "db/Status.h"
#include "db/Types.h"

#include <cstddef>
#include <memory>

namespace zilliz {
namespace milvus {
namespace engine {
namespace meta {

class Meta {
 public:
    virtual ~Meta() = default;

    virtual Status CreateTable(TableSchema &table_schema) = 0;

    virtual Status DescribeTable(TableSchema &table_schema) = 0;

    virtual Status HasTable(const std::string &table_id, bool &has_or_not) = 0;

    virtual Status AllTables(std::vector<TableSchema> &table_schema_array) = 0;

    virtual Status UpdateTableIndex(const std::string &table_id, const TableIndex& index) = 0;

    virtual Status UpdateTableFlag(const std::string &table_id, int64_t flag) = 0;

    virtual Status DeleteTable(const std::string &table_id) = 0;

    virtual Status DeleteTableFiles(const std::string &table_id) = 0;

    virtual Status CreateTableFile(TableFileSchema &file_schema) = 0;

    virtual Status DropPartitionsByDates(const std::string &table_id, const DatesT &dates) = 0;

    virtual Status GetTableFiles(const std::string &table_id,
                                 const std::vector<size_t> &ids,
                                 TableFilesSchema &table_files) = 0;

    virtual Status UpdateTableFilesToIndex(const std::string &table_id) = 0;

    virtual Status UpdateTableFile(TableFileSchema &file_schema) = 0;

    virtual Status UpdateTableFiles(TableFilesSchema &files) = 0;

    virtual Status FilesToSearch(const std::string &table_id,
                                 const std::vector<size_t> &ids,
                                 const DatesT &partition,
                                 DatePartionedTableFilesSchema &files) = 0;

    virtual Status FilesToMerge(const std::string &table_id, DatePartionedTableFilesSchema &files) = 0;

    virtual Status Size(uint64_t &result) = 0;

    virtual Status Archive() = 0;

    virtual Status FilesToIndex(TableFilesSchema &) = 0;

    virtual Status FilesByType(const std::string &table_id,
                               const std::vector<int> &file_types,
                               std::vector<std::string>& file_ids) = 0;

    virtual Status DescribeTableIndex(const std::string &table_id, TableIndex& index) = 0;

    virtual Status DropTableIndex(const std::string &table_id) = 0;

    virtual Status CleanUp() = 0;

    virtual Status CleanUpFilesWithTTL(uint16_t) = 0;

    virtual Status DropAll() = 0;

    virtual Status Count(const std::string &table_id, uint64_t &result) = 0;

}; // MetaData

using MetaPtr = std::shared_ptr<Meta>;

} // namespace meta
} // namespace engine
} // namespace milvus
} // namespace zilliz
