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

#include "Status.h"

/** \brief Milvus SDK namespace
 */
namespace milvus {

/**
 * @brief Index Type
 */
enum class IndexType {
    INVALID = 0,
    FLAT = 1,
    IVFFLAT = 2,
    IVFSQ8 = 3,
    RNSG = 4,
    IVFSQ8H = 5,
    IVFPQ = 6,
    SPTAGKDT = 7,
    SPTAGBKT = 8,
    HNSW = 11,
};

enum class MetricType {
    L2 = 1,        // Euclidean Distance
    IP = 2,        // Cosine Similarity
    HAMMING = 3,   // Hamming Distance
    JACCARD = 4,   // Jaccard Distance
    TANIMOTO = 5,  // Tanimoto Distance
};

/**
 * @brief Connect API parameter
 */
struct ConnectParam {
    std::string ip_address;  ///< Server IP address
    std::string port;        ///< Server PORT
};

/**
 * @brief Table Schema
 */
struct TableSchema {
    std::string table_name;                   ///< Table name
    int64_t dimension = 0;                    ///< Vector dimension, must be a positive value
    int64_t index_file_size = 1024;           ///< Index file size, must be a positive value, unit: MB
    MetricType metric_type = MetricType::L2;  ///< Index metric type
};


/**
 * @brief Record inserted
 */
struct RowRecord {
    std::vector<float> float_data;     ///< Vector raw float data
    std::vector<uint8_t> binary_data;  ///< Vector raw binary data
};

/**
 * @brief TopK query result
 */
struct QueryResult {
    std::vector<int64_t> ids;      ///< Query ids result
    std::vector<float> distances;  ///< Query distances result
};
using TopKQueryResult = std::vector<QueryResult>;  ///< Topk query result

/**
 * @brief Index parameters
 * Note: extra_params is extra parameters list, it must be json format
 *       For different index type, parameter list is different accordingly, for example:
 *       FLAT/IVFLAT/SQ8:  "{nlist: '16384'}"
 *       IVFPQ:  "{nlist: '16384', nbits: "16"}"
 *       NSG:  "{search_length: '100', out_degree:'40', pool_size:'66'}"
 *       HNSW  "{M: '16', ef_construct:'500'}"
 */
struct IndexParam {
    std::string table_name;             ///< Table name for create index
    IndexType index_type;               ///< Index type
    std::string extra_params;           ///< Extra parameters according to different index type, must be json format
};

/**
 * @brief partition parameters
 */
struct PartitionParam {
    std::string table_name;
    std::string partition_tag;
};

using PartitionTagList = std::vector<std::string>;

/**
 * @brief segment statistics
 */
struct SegmentStat {
    std::string segment_name;    ///< Segment name
    int64_t row_count;           ///< Segment row count
    std::string index_name;      ///< Segment index name
    int64_t data_size;           ///< Segment data size
};

/**
 * @brief partition statistics
 */
struct PartitionStat {
    std::string tag;                          ///< Partition tag
    int64_t row_count;                        ///< Partition row count
    std::vector<SegmentStat> segments_stat;   ///< Partition's segments statistics
};

/**
 * @brief table info
 */
struct TableInfo {
    int64_t total_row_count;                      ///< Table total row count
    std::vector<PartitionStat> partitions_stat;   ///< Table's partitions statistics
};

/**
 * @brief SDK main class
 */
class Connection {
 public:
    /**
     * @brief CreateConnection
     *
     * Create a connection instance and return it's shared pointer
     *
     * @return Connection instance pointer
     */

    static std::shared_ptr<Connection>
    Create();

    /**
     * @brief DestroyConnection
     *
     * Destroy the connection instance
     *
     * @param connection, the shared pointer to the instance to be destroyed
     *
     * @return if destroy is successful
     */

    static Status
    Destroy(std::shared_ptr<Connection>& connection_ptr);

    /**
     * @brief Connect
     *
     * This method is used to connect server.
     * Connect function should be called before any operations.
     *
     * @param param, use to provide server information
     *
     * @return Indicate if connect is successful
     */

    virtual Status
    Connect(const ConnectParam& param) = 0;

    /**
     * @brief Connect
     *
     * This method is used to connect server.
     * Connect function should be called before any operations.
     *
     * @param uri, use to provide server uri, example: milvus://ipaddress:port
     *
     * @return Indicate if connect is successful
     */
    virtual Status
    Connect(const std::string& uri) = 0;

    /**
     * @brief connected
     *
     * This method is used to test whether server is connected.
     *
     * @return Indicate if connection status
     */
    virtual Status
    Connected() const = 0;

    /**
     * @brief Disconnect
     *
     * This method is used to disconnect server.
     *
     * @return Indicate if disconnect is successful
     */
    virtual Status
    Disconnect() = 0;

    /**
     * @brief Create table method
     *
     * This method is used to create table.
     *
     * @param param, use to provide table information to be created.
     *
     * @return Indicate if table is created successfully
     */
    virtual Status
    CreateTable(const TableSchema& param) = 0;

    /**
     * @brief Test table existence method
     *
     * This method is used to create table.
     *
     * @param table_name, target table's name.
     *
     * @return Indicate if table is cexist
     */
    virtual bool
    HasTable(const std::string& table_name) = 0;

    /**
     * @brief Drop table method
     *
     * This method is used to drop table(and its partitions).
     *
     * @param table_name, target table's name.
     *
     * @return Indicate if table is drop successfully.
     */
    virtual Status
    DropTable(const std::string& table_name) = 0;

    /**
     * @brief Create index method
     *
     * This method is used to create index for whole table(and its partitions).
     *
     * @param IndexParam
     *  table_name, table name is going to be create index.
     *  index type,
     *  nlist,
     *  index file size
     *
     * @return Indicate if build index successfully.
     */
    virtual Status
    CreateIndex(const IndexParam& index_param) = 0;

    /**
     * @brief Insert vector to table
     *
     * This method is used to insert vector array to table.
     *
     * @param table_name, target table's name.
     * @param partition_tag, target partition's tag, keep empty if no partition specified.
     * @param record_array, vector array is inserted.
     * @param id_array,
     *  specify id for each vector,
     *  if this array is empty, milvus will generate unique id for each vector,
     *  and return all ids by this parameter.
     *
     * @return Indicate if vector array are inserted successfully
     */
    virtual Status
    Insert(const std::string& table_name, const std::string& partition_tag, const std::vector<RowRecord>& record_array,
           std::vector<int64_t>& id_array) = 0;

    /**
     * @brief Get vector data by id
     *
     * This method is used to get vector data by id from a table.
     * Return the first found vector if there are vectors with duplicated id
     *
     * @param table_name, target table's name.
     * @param vector_id, target vector id.
     * @param vector_data, returned vector data.
     *
     * @return Indicate if the operation is succeed.
     */
    virtual Status
    GetVectorByID(const std::string& table_name, int64_t vector_id, RowRecord& vector_data) = 0;

    /**
     * @brief Get vector ids from a segment
     *
     * This method is used to get vector ids from a segment
     * Return all vector(not deleted) ids
     *
     * @param table_name, target table's name.
     * @param segment_name, target segment name.
     * @param id_array, returned vector id array.
     *
     * @return Indicate if the operation is succeed.
     */
    virtual Status
    GetIDsInSegment(const std::string& table_name, const std::string& segment_name, std::vector<int64_t>& id_array) = 0;

    /**
     * @brief Search vector
     *
     * This method is used to query vector in table.
     *
     * @param table_name, target table's name.
     * @param partition_tag_array, target partitions, keep empty if no partition specified.
     * @param query_record_array, vectors to be queried.
     * @param topk, how many similarity vectors will be returned.
     * @param extra_params, extra search parameters according to different index type, must be json format.
     * Note: extra_params is extra parameters list, it must be json format, for example:
     *       For different index type, parameter list is different accordingly
     *       FLAT/IVFLAT/SQ8/IVFPQ:  "{nprobe: '32'}"
     *       NSG:  "{search_length:'100'}
     *       HNSW  "{ef: '64'}
     * @param topk_query_result, result array.
     *
     * @return Indicate if query is successful.
     */
    virtual Status
    Search(const std::string& table_name, const PartitionTagList& partition_tag_array,
           const std::vector<RowRecord>& query_record_array, int64_t topk,
           const std::string& extra_params, TopKQueryResult& topk_query_result) = 0;

    /**
     * @brief Show table description
     *
     * This method is used to show table information.
     *
     * @param table_name, target table's name.
     * @param table_schema, table_schema is given when operation is successful.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    DescribeTable(const std::string& table_name, TableSchema& table_schema) = 0;

    /**
     * @brief Get table row count
     *
     * This method is used to get table row count.
     *
     * @param table_name, target table's name.
     * @param row_count, table total row count(including partitions).
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    CountTable(const std::string& table_name, int64_t& row_count) = 0;

    /**
     * @brief Show all tables in database
     *
     * This method is used to list all tables.
     *
     * @param table_array, all tables in database.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    ShowTables(std::vector<std::string>& table_array) = 0;

    /**
     * @brief Show table information
     *
     * This method is used to get detail information of a table.
     *
     * @param table_name, target table's name.
     * @param table_info, target table's information
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    ShowTableInfo(const std::string& table_name, TableInfo& table_info) = 0;

    /**
     * @brief Give the client version
     *
     * This method is used to give the client version.
     *
     * @return Client version.
     */
    virtual std::string
    ClientVersion() const = 0;

    /**
     * @brief Give the server version
     *
     * This method is used to give the server version.
     *
     * @return Server version.
     */
    virtual std::string
    ServerVersion() const = 0;

    /**
     * @brief Give the server status
     *
     * This method is used to give the server status.
     *
     * @return Server status.
     */
    virtual std::string
    ServerStatus() const = 0;

    /**
     * @brief dump server tasks information
     *
     * This method is internal used.
     *
     * @return Task information in tasktables.
     */
    virtual std::string
    DumpTaskTables() const = 0;

    /**
     * [deprecated]
     * @brief delete tables by vector id
     *
     * This method is used to delete table data by date range.
     *
     * @param table_name, target table's name.
     * @param id_array, vector ids to deleted.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    DeleteByID(const std::string& table_name, const std::vector<int64_t>& id_array) = 0;

    /**
     * @brief preload table
     *
     * This method is used to preload table
     *
     * @param table_name
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    PreloadTable(const std::string& table_name) const = 0;

    /**
     * @brief describe index
     *
     * This method is used to describe index
     *
     * @param table_name, target table's name.
     * @param index_param, returned index information.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    DescribeIndex(const std::string& table_name, IndexParam& index_param) const = 0;

    /**
     * @brief drop index
     *
     * This method is used to drop index of table(and its partitions)
     *
     * @param table_name, target table's name.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    DropIndex(const std::string& table_name) const = 0;

    /**
     * @brief Create partition method
     *
     * This method is used to create table partition
     *
     * @param param, use to provide partition information to be created.
     *
     * @return Indicate if partition is created successfully
     */
    virtual Status
    CreatePartition(const PartitionParam& param) = 0;

    /**
     * @brief Test table existence method
     *
     * This method is used to create table
     *
     * @param table_name, table name is going to be tested.
     * @param partition_tag_array, partition tag array of the table.
     *
     * @return Indicate if this operation is successful
     */
    virtual Status
    ShowPartitions(const std::string& table_name, PartitionTagList& partition_tag_array) const = 0;

    /**
     * @brief Delete partition method
     *
     * This method is used to delete table partition.
     *
     * @param param, target partition to be deleted.
     *      NOTE: if param.table_name is empty, you must specify param.partition_name,
     *          else you can specify param.table_name and param.tag and let the param.partition_name be empty
     *
     * @return Indicate if partition is delete successfully.
     */
    virtual Status
    DropPartition(const PartitionParam& param) = 0;

    /**
     * @brief Get config method
     *
     * This method is used to set config.
     *
     * @param node_name, config node name.
     * @param value, config value.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    GetConfig(const std::string& node_name, std::string& value) const = 0;

    /**
     * @brief Set config method
     *
     * This method is used to set config.
     *
     * @param node_name, config node name.
     * @param value, config value.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    SetConfig(const std::string& node_name, const std::string& value) const = 0;

    /**
     * @brief flush table buffer into storage
     *
     * This method is used to flush table buffer into storage
     *
     * @param table_name, target table's name.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    FlushTable(const std::string& table_name) = 0;

    /**
     * @brief flush all buffer into storage
     *
     * This method is used to all table buffer into storage
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    Flush() = 0;

    /**
     * @brief compact table, remove deleted vectors
     *
     * This method is used to compact table
     *
     * @param table_name, target table's name.
     *
     * @return Indicate if this operation is successful.
     */
    virtual Status
    CompactTable(const std::string& table_name) = 0;
};

}  // namespace milvus
