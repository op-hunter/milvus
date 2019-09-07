/*******************************************************************************
* Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
******************************************************************************/
#include "GrpcRequestTask.h"
#include "../ServerConfig.h"
#include "utils/CommonUtil.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"
#include "../DBWrapper.h"
#include "version.h"
#include "GrpcMilvusServer.h"
#include "db/Utils.h"
#include "scheduler/SchedInst.h"

#include "src/server/Server.h"

#include <string.h>

namespace zilliz {
namespace milvus {
namespace server {
namespace grpc {

static const char *DQL_TASK_GROUP = "dql";
static const char *DDL_DML_TASK_GROUP = "ddl_dml";
static const char *PING_TASK_GROUP = "ping";

using DB_META = zilliz::milvus::engine::meta::Meta;
using DB_DATE = zilliz::milvus::engine::meta::DateT;

namespace {
    engine::EngineType EngineType(int type) {
        static std::map<int, engine::EngineType> map_type = {
                {0, engine::EngineType::INVALID},
                {1, engine::EngineType::FAISS_IDMAP},
                {2, engine::EngineType::FAISS_IVFFLAT},
                {3, engine::EngineType::FAISS_IVFSQ8},
        };

        if (map_type.find(type) == map_type.end()) {
            return engine::EngineType::INVALID;
        }

        return map_type[type];
    }

    int IndexType(engine::EngineType type) {
        static std::map<engine::EngineType, int> map_type = {
                {engine::EngineType::INVALID,       0},
                {engine::EngineType::FAISS_IDMAP,   1},
                {engine::EngineType::FAISS_IVFFLAT, 2},
                {engine::EngineType::FAISS_IVFSQ8,  3},
        };

        if (map_type.find(type) == map_type.end()) {
            return 0;
        }

        return map_type[type];
    }

    constexpr long DAY_SECONDS = 24 * 60 * 60;

    void
    ConvertTimeRangeToDBDates(const std::vector<::milvus::grpc::Range> &range_array,
                              std::vector<DB_DATE> &dates,
                              ErrorCode &error_code,
                              std::string &error_msg) {
        dates.clear();
        for (auto &range : range_array) {
            time_t tt_start, tt_end;
            tm tm_start, tm_end;
            if (!CommonUtil::TimeStrToTime(range.start_value(), tt_start, tm_start)) {
                error_code = SERVER_INVALID_TIME_RANGE;
                error_msg = "Invalid time range: " + range.start_value();
                return;
            }

            if (!CommonUtil::TimeStrToTime(range.end_value(), tt_end, tm_end)) {
                error_code = SERVER_INVALID_TIME_RANGE;
                error_msg = "Invalid time range: " + range.start_value();
                return;
            }

            long days = (tt_end > tt_start) ? (tt_end - tt_start) / DAY_SECONDS : (tt_start - tt_end) /
                                                                                  DAY_SECONDS;
            if (days == 0) {
                error_code = SERVER_INVALID_TIME_RANGE;
                error_msg = "Invalid time range: " + range.start_value() + " to " + range.end_value();
                return;
            }

            //range: [start_day, end_day)
            for (long i = 0; i < days; i++) {
                time_t tt_day = tt_start + DAY_SECONDS * i;
                tm tm_day;
                CommonUtil::ConvertTime(tt_day, tm_day);

                long date = tm_day.tm_year * 10000 + tm_day.tm_mon * 100 +
                            tm_day.tm_mday;//according to db logic
                dates.push_back(date);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CreateTableTask::CreateTableTask(const ::milvus::grpc::TableSchema *schema)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          schema_(schema) {

}

BaseTaskPtr
CreateTableTask::Create(const ::milvus::grpc::TableSchema *schema) {
    if(schema == nullptr) {
        SERVER_LOG_ERROR << "grpc input is null!";
        return nullptr;
    }
    return std::shared_ptr<GrpcBaseTask>(new CreateTableTask(schema));
}

ErrorCode
CreateTableTask::OnExecute() {
    TimeRecorder rc("CreateTableTask");

    try {
        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(schema_->table_name().table_name());
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + schema_->table_name().table_name());
        }

        res = ValidationUtil::ValidateTableDimension(schema_->dimension());
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table dimension: " + std::to_string(schema_->dimension()));
        }

        res = ValidationUtil::ValidateTableIndexFileSize(schema_->index_file_size());
        if(res != SERVER_SUCCESS) {
            return SetError(res, "Invalid index file size: " + std::to_string(schema_->index_file_size()));
        }

        res = ValidationUtil::ValidateTableIndexMetricType(schema_->metric_type());
        if(res != SERVER_SUCCESS) {
            return SetError(res, "Invalid index metric type: " + std::to_string(schema_->metric_type()));
        }

        //step 2: construct table schema
        engine::meta::TableSchema table_info;
        table_info.table_id_ = schema_->table_name().table_name();
        table_info.dimension_ = (uint16_t) schema_->dimension();
        table_info.index_file_size_ = schema_->index_file_size();
        table_info.metric_type_ = schema_->metric_type();

        //step 3: create table
        engine::Status stat = DBWrapper::DB()->CreateTable(table_info);
        if (!stat.ok()) {
            //table could exist
            if(stat.code() == DB_ALREADY_EXIST) {
                return SetError(SERVER_INVALID_TABLE_NAME, stat.ToString());
            }
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    rc.ElapseFromBegin("totally cost");

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DescribeTableTask::DescribeTableTask(const std::string &table_name, ::milvus::grpc::TableSchema *schema)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          table_name_(table_name),
          schema_(schema) {
}

BaseTaskPtr
DescribeTableTask::Create(const std::string &table_name, ::milvus::grpc::TableSchema *schema) {
    return std::shared_ptr<GrpcBaseTask>(new DescribeTableTask(table_name, schema));
}

ErrorCode
DescribeTableTask::OnExecute() {
    TimeRecorder rc("DescribeTableTask");

    try {
        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: get table info
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        engine::Status stat = DBWrapper::DB()->DescribeTable(table_info);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        schema_->mutable_table_name()->set_table_name(table_info.table_id_);
        schema_->set_dimension(table_info.dimension_);
        schema_->set_index_file_size(table_info.index_file_size_);
        schema_->set_metric_type(table_info.metric_type_);

    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    rc.ElapseFromBegin("totally cost");

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CreateIndexTask::CreateIndexTask(const ::milvus::grpc::IndexParam *index_param)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          index_param_(index_param) {
}

BaseTaskPtr
CreateIndexTask::Create(const ::milvus::grpc::IndexParam *index_param) {
    if(index_param == nullptr) {
        SERVER_LOG_ERROR << "grpc input is null!";
        return nullptr;
    }
    return std::shared_ptr<GrpcBaseTask>(new CreateIndexTask(index_param));
}

ErrorCode
CreateIndexTask::OnExecute() {
    try {
        TimeRecorder rc("CreateIndexTask");

        //step 1: check arguments
        std::string table_name_ = index_param_->table_name().table_name();
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        bool has_table = false;
        engine::Status stat = DBWrapper::DB()->HasTable(table_name_, has_table);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        if (!has_table) {
            return SetError(SERVER_TABLE_NOT_EXIST, "Table " + table_name_ + " not exists");
        }

        auto &grpc_index = index_param_->index();
        res = ValidationUtil::ValidateTableIndexType(grpc_index.index_type());
        if(res != SERVER_SUCCESS) {
            return SetError(res, "Invalid index type: " + std::to_string(grpc_index.index_type()));
        }

        res = ValidationUtil::ValidateTableIndexNlist(grpc_index.nlist());
        if(res != SERVER_SUCCESS) {
            return SetError(res, "Invalid index nlist: " + std::to_string(grpc_index.nlist()));
        }

        //step 2: check table existence
        engine::TableIndex index;
        index.engine_type_ = grpc_index.index_type();
        index.nlist_ = grpc_index.nlist();
        stat = DBWrapper::DB()->CreateIndex(table_name_, index);
        if (!stat.ok()) {
            return SetError(SERVER_BUILD_INDEX_ERROR, stat.ToString());
        }

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
HasTableTask::HasTableTask(const std::string &table_name, bool &has_table)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          table_name_(table_name),
          has_table_(has_table) {

}

BaseTaskPtr
HasTableTask::Create(const std::string &table_name, bool &has_table) {
    return std::shared_ptr<GrpcBaseTask>(new HasTableTask(table_name, has_table));
}

ErrorCode
HasTableTask::OnExecute() {
    try {
        TimeRecorder rc("HasTableTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        engine::Status stat = DBWrapper::DB()->HasTable(table_name_, has_table_);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DropTableTask::DropTableTask(const std::string &table_name)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          table_name_(table_name) {

}

BaseTaskPtr
DropTableTask::Create(const std::string &table_name) {
    return std::shared_ptr<GrpcBaseTask>(new DropTableTask(table_name));
}

ErrorCode
DropTableTask::OnExecute() {
    try {
        TimeRecorder rc("DropTableTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        engine::Status stat = DBWrapper::DB()->DescribeTable(table_info);
        if (!stat.ok()) {
            if (stat.code() == DB_NOT_FOUND) {
                return SetError(SERVER_TABLE_NOT_EXIST, "Table " + table_name_ + " not exists");
            } else {
                return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
            }
        }

        rc.ElapseFromBegin("check validation");

        //step 3: Drop table
        std::vector<DB_DATE> dates;
        stat = DBWrapper::DB()->DeleteTable(table_name_, dates);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        rc.ElapseFromBegin("total cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ShowTablesTask::ShowTablesTask(::grpc::ServerWriter<::milvus::grpc::TableName> *writer)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          writer_(writer) {

}

BaseTaskPtr
ShowTablesTask::Create(::grpc::ServerWriter<::milvus::grpc::TableName> *writer) {
    return std::shared_ptr<GrpcBaseTask>(new ShowTablesTask(writer));
}

ErrorCode
ShowTablesTask::OnExecute() {
    std::vector<engine::meta::TableSchema> schema_array;
    engine::Status stat = DBWrapper::DB()->AllTables(schema_array);
    if (!stat.ok()) {
        return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
    }

    for (auto &schema : schema_array) {
        ::milvus::grpc::TableName tableName;
        tableName.set_table_name(schema.table_id_);
        if (!writer_->Write(tableName)) {
            return SetError(SERVER_WRITE_ERROR, "Write table name failed!");
        }
    }
    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
InsertTask::InsertTask(const ::milvus::grpc::InsertParam *insert_param,
                       ::milvus::grpc::VectorIds *record_ids)
    : GrpcBaseTask(DDL_DML_TASK_GROUP),
      insert_param_(insert_param),
      record_ids_(record_ids) {
    record_ids_->Clear();
}

BaseTaskPtr
InsertTask::Create(const ::milvus::grpc::InsertParam *insert_param,
                         ::milvus::grpc::VectorIds *record_ids) {
    if(insert_param == nullptr) {
        SERVER_LOG_ERROR << "grpc input is null!";
        return nullptr;
    }
    return std::shared_ptr<GrpcBaseTask>(new InsertTask(insert_param, record_ids));
}

ErrorCode
InsertTask::OnExecute() {
    try {
        TimeRecorder rc("InsertVectorTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(insert_param_->table_name());
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + insert_param_->table_name());
        }
        if (insert_param_->row_record_array().empty()) {
            return SetError(SERVER_INVALID_ROWRECORD_ARRAY, "Row record array is empty");
        }

        if (!record_ids_->vector_id_array().empty()) {
            if (record_ids_->vector_id_array().size() != insert_param_->row_record_array_size()) {
                return SetError(SERVER_ILLEGAL_VECTOR_ID,
                        "Size of vector ids is not equal to row record array size");
            }
        }

        //step 2: check table existence
        engine::meta::TableSchema table_info;
        table_info.table_id_ = insert_param_->table_name();
        engine::Status stat = DBWrapper::DB()->DescribeTable(table_info);
        if (!stat.ok()) {
            if (stat.code() == DB_NOT_FOUND) {
                return SetError(SERVER_TABLE_NOT_EXIST,
                                "Table " + insert_param_->table_name() + " not exists");
            } else {
                return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
            }
        }

        //step 3: check table flag
        //all user provide id, or all internal id
        bool user_provide_ids = !insert_param_->row_id_array().empty();
        //user already provided id before, all insert action require user id
        if((table_info.flag_ & engine::meta::FLAG_MASK_HAS_USERID) && !user_provide_ids) {
            return SetError(SERVER_INVALID_ARGUMENT, "Table vector ids are user defined, please provide id for this batch");
        }

        //user didn't provided id before, no need to provide user id
        if((table_info.flag_ & engine::meta::FLAG_MASK_NO_USERID) && user_provide_ids) {
            return SetError(SERVER_INVALID_ARGUMENT, "Table vector ids are auto generated, no need to provide id for this batch");
        }

        rc.RecordSection("check validation");

#ifdef MILVUS_ENABLE_PROFILING
        std::string fname = "/tmp/insert_" + std::to_string(this->record_array_.size()) +
                            "_" + GetCurrTimeStr() + ".profiling";
        ProfilerStart(fname.c_str());
#endif

        //step 4: prepare float data
        std::vector<float> vec_f(insert_param_->row_record_array_size() * table_info.dimension_, 0);

        // TODO: change to one dimension array in protobuf or use multiple-thread to copy the data
        for (size_t i = 0; i < insert_param_->row_record_array_size(); i++) {
            if (insert_param_->row_record_array(i).vector_data().empty()) {
                return SetError(SERVER_INVALID_ROWRECORD_ARRAY, "Row record float array is empty");
            }
            uint64_t vec_dim = insert_param_->row_record_array(i).vector_data().size();
            if (vec_dim != table_info.dimension_) {
                ErrorCode error_code = SERVER_INVALID_VECTOR_DIMENSION;
                std::string error_msg = "Invalid rowrecord dimension: " + std::to_string(vec_dim)
                                        + " vs. table dimension:" +
                                        std::to_string(table_info.dimension_);
                return SetError(error_code, error_msg);
            }
            memcpy(&vec_f[i * table_info.dimension_],
                   insert_param_->row_record_array(i).vector_data().data(),
                   table_info.dimension_ * sizeof(float));
        }

        rc.ElapseFromBegin("prepare vectors data");

        //step 5: insert vectors
        auto vec_count = (uint64_t) insert_param_->row_record_array_size();
        std::vector<int64_t> vec_ids(insert_param_->row_id_array_size(), 0);
        if(!insert_param_->row_id_array().empty()) {
            const int64_t* src_data = insert_param_->row_id_array().data();
            int64_t* target_data = vec_ids.data();
            memcpy(target_data, src_data, (size_t)(sizeof(int64_t)*insert_param_->row_id_array_size()));
        }

        stat = DBWrapper::DB()->InsertVectors(insert_param_->table_name(), vec_count, vec_f.data(), vec_ids);
        rc.ElapseFromBegin("add vectors to engine");
        if (!stat.ok()) {
            return SetError(SERVER_CACHE_ERROR, "Cache error: " + stat.ToString());
        }
        for (int64_t id : vec_ids) {
            record_ids_->add_vector_id_array(id);
        }

        auto ids_size = record_ids_->vector_id_array_size();
        if (ids_size != vec_count) {
            std::string msg = "Add " + std::to_string(vec_count) + " vectors but only return "
                              + std::to_string(ids_size) + " id";
            return SetError(SERVER_ILLEGAL_VECTOR_ID, msg);
        }

        //step 6: update table flag
        user_provide_ids ? table_info.flag_ |= engine::meta::FLAG_MASK_HAS_USERID
                : table_info.flag_ |= engine::meta::FLAG_MASK_NO_USERID;
        stat = DBWrapper::DB()->UpdateTableFlag(insert_param_->table_name(), table_info.flag_);

#ifdef MILVUS_ENABLE_PROFILING
        ProfilerStop();
#endif

        rc.RecordSection("add vectors to engine");
        rc.ElapseFromBegin("total cost");

    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SearchTask::SearchTask(const ::milvus::grpc::SearchParam *search_vector_infos,
                       const std::vector<std::string> &file_id_array,
                       ::milvus::grpc::TopKQueryResultList *response)
    : GrpcBaseTask(DQL_TASK_GROUP),
      search_param_(search_vector_infos),
      file_id_array_(file_id_array),
      topk_result_list(response) {

}

BaseTaskPtr
SearchTask::Create(const ::milvus::grpc::SearchParam *search_vector_infos,
                   const std::vector<std::string> &file_id_array,
                   ::milvus::grpc::TopKQueryResultList *response) {
    if(search_vector_infos == nullptr) {
        SERVER_LOG_ERROR << "grpc input is null!";
        return nullptr;
    }
    return std::shared_ptr<GrpcBaseTask>(new SearchTask(search_vector_infos, file_id_array,
                                                        response));
}

ErrorCode
SearchTask::OnExecute() {
    try {
        TimeRecorder rc("SearchTask");

        //step 1: check table name
        std::string table_name_ = search_param_->table_name();
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        engine::Status stat = DBWrapper::DB()->DescribeTable(table_info);
        if (!stat.ok()) {
            if (stat.code() == DB_NOT_FOUND) {
                return SetError(SERVER_TABLE_NOT_EXIST, "Table " + table_name_ + " not exists");
            } else {
                return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
            }
        }

        //step 3: check search parameter
        int64_t top_k = search_param_->topk();
        res = ValidationUtil::ValidateSearchTopk(top_k, table_info);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid topk: " + std::to_string(top_k));
        }

        int64_t nprobe = search_param_->nprobe();
        res = ValidationUtil::ValidateSearchNprobe(nprobe, table_info);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid nprobe: " + std::to_string(nprobe));
        }

        if (search_param_->query_record_array().empty()) {
            return SetError(SERVER_INVALID_ROWRECORD_ARRAY, "Row record array is empty");
        }

        //step 4: check date range, and convert to db dates
        std::vector<DB_DATE> dates;
        ErrorCode error_code = SERVER_SUCCESS;
        std::string error_msg;

        std::vector<::milvus::grpc::Range> range_array;
        for (size_t i = 0; i < search_param_->query_range_array_size(); i++) {
            range_array.emplace_back(search_param_->query_range_array(i));
        }
        ConvertTimeRangeToDBDates(range_array, dates, error_code, error_msg);
        if (error_code != SERVER_SUCCESS) {
            return SetError(error_code, error_msg);
        }

        double span_check = rc.RecordSection("check validation");

#ifdef MILVUS_ENABLE_PROFILING
        std::string fname = "/tmp/search_nq_" + std::to_string(this->record_array_.size()) +
                            "_top_" + std::to_string(this->top_k_) + "_" +
                            GetCurrTimeStr() + ".profiling";
        ProfilerStart(fname.c_str());
#endif

        //step 5: prepare float data
        auto record_array_size = search_param_->query_record_array_size();
        std::vector<float> vec_f(record_array_size * table_info.dimension_, 0);
        for (size_t i = 0; i < record_array_size; i++) {
            if (search_param_->query_record_array(i).vector_data().empty()) {
                return SetError(SERVER_INVALID_ROWRECORD_ARRAY, "Query record float array is empty");
            }
            uint64_t query_vec_dim = search_param_->query_record_array(i).vector_data().size();
            if (query_vec_dim != table_info.dimension_) {
                ErrorCode error_code = SERVER_INVALID_VECTOR_DIMENSION;
                std::string error_msg = "Invalid rowrecord dimension: " + std::to_string(query_vec_dim)
                                        + " vs. table dimension:" + std::to_string(table_info.dimension_);
                return SetError(error_code, error_msg);
            }

            memcpy(&vec_f[i * table_info.dimension_],
                   search_param_->query_record_array(i).vector_data().data(),
                   table_info.dimension_ * sizeof(float));
        }
        rc.ElapseFromBegin("prepare vector data");

        //step 6: search vectors
        engine::QueryResults results;
        auto record_count = (uint64_t) search_param_->query_record_array().size();

        if (file_id_array_.empty()) {
            stat = DBWrapper::DB()->Query(table_name_, (size_t) top_k, record_count, nprobe, vec_f.data(),
                                          dates, results);
        } else {
            stat = DBWrapper::DB()->Query(table_name_, file_id_array_, (size_t) top_k,
                                          record_count, nprobe, vec_f.data(), dates, results);
        }

        rc.ElapseFromBegin("search vectors from engine");
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        if (results.empty()) {
            return SERVER_SUCCESS; //empty table
        }

        if (results.size() != record_count) {
            std::string msg = "Search " + std::to_string(record_count) + " vectors but only return "
                              + std::to_string(results.size()) + " results";
            return SetError(SERVER_ILLEGAL_SEARCH_RESULT, msg);
        }

        rc.ElapseFromBegin("do search");

        //step 7: construct result array
        for (auto &result : results) {
            ::milvus::grpc::TopKQueryResult *topk_query_result = topk_result_list->add_topk_query_result();
            for (auto &pair : result) {
                ::milvus::grpc::QueryResult *grpc_result = topk_query_result->add_query_result_arrays();
                grpc_result->set_id(pair.first);
                grpc_result->set_distance(pair.second);
            }
        }

#ifdef MILVUS_ENABLE_PROFILING
        ProfilerStop();
#endif

        //step 8: print time cost percent
        double span_result = rc.RecordSection("construct result");
        rc.ElapseFromBegin("totally cost");


    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CountTableTask::CountTableTask(const std::string &table_name, int64_t &row_count)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          table_name_(table_name),
          row_count_(row_count) {

}

BaseTaskPtr
CountTableTask::Create(const std::string &table_name, int64_t &row_count) {
    return std::shared_ptr<GrpcBaseTask>(new CountTableTask(table_name, row_count));
}

ErrorCode
CountTableTask::OnExecute() {
    try {
        TimeRecorder rc("GetTableRowCountTask");

        //step 1: check arguments
        ErrorCode res = SERVER_SUCCESS;
        res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: get row count
        uint64_t row_count = 0;
        engine::Status stat = DBWrapper::DB()->GetTableRowCount(table_name_, row_count);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        row_count_ = (int64_t) row_count;

        rc.ElapseFromBegin("total cost");

    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdTask::CmdTask(const std::string &cmd, std::string &result)
        : GrpcBaseTask(PING_TASK_GROUP),
          cmd_(cmd),
          result_(result) {

}

BaseTaskPtr
CmdTask::Create(const std::string &cmd, std::string &result) {
    return std::shared_ptr<GrpcBaseTask>(new CmdTask(cmd, result));
}

ErrorCode
CmdTask::OnExecute() {
    if (cmd_ == "version") {
        result_ = MILVUS_VERSION;
    } else if (cmd_ == "tasktable") {
        result_ = engine::ResMgrInst::GetInstance()->DumpTaskTables();
    }
    else {
        result_ = "OK";
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DeleteByRangeTask::DeleteByRangeTask(const ::milvus::grpc::DeleteByRangeParam *delete_by_range_param)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          delete_by_range_param_(delete_by_range_param){
}

BaseTaskPtr
DeleteByRangeTask::Create(const ::milvus::grpc::DeleteByRangeParam *delete_by_range_param) {
    if(delete_by_range_param == nullptr) {
        SERVER_LOG_ERROR << "grpc input is null!";
        return nullptr;
    }
    return std::shared_ptr<GrpcBaseTask>(new DeleteByRangeTask(delete_by_range_param));
}

ErrorCode
DeleteByRangeTask::OnExecute() {
    try {
        TimeRecorder rc("DeleteByRangeTask");

        //step 1: check arguments
        std::string table_name = delete_by_range_param_->table_name();
        ErrorCode res = ValidationUtil::ValidateTableName(table_name);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name);
        }

        //step 2: check table existence
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name;
        engine::Status stat = DBWrapper::DB()->DescribeTable(table_info);
        if (!stat.ok()) {
            if (stat.code(), DB_NOT_FOUND) {
                return SetError(SERVER_TABLE_NOT_EXIST, "Table " + table_name + " not exists");
            } else {
                return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
            }
        }

        rc.ElapseFromBegin("check validation");

        //step 3: check date range, and convert to db dates
        std::vector<DB_DATE> dates;
        ErrorCode error_code = SERVER_SUCCESS;
        std::string error_msg;

        std::vector<::milvus::grpc::Range> range_array;
        range_array.emplace_back(delete_by_range_param_->range());
        ConvertTimeRangeToDBDates(range_array, dates, error_code, error_msg);
        if (error_code != SERVER_SUCCESS) {
            return SetError(error_code, error_msg);
        }

#ifdef MILVUS_ENABLE_PROFILING
        std::string fname = "/tmp/search_nq_" + std::to_string(this->record_array_.size()) +
                            "_top_" + std::to_string(this->top_k_) + "_" +
                            GetCurrTimeStr() + ".profiling";
        ProfilerStart(fname.c_str());
#endif
        engine::Status status = DBWrapper::DB()->DeleteTable(table_name, dates);
        if (!status.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }
    
    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PreloadTableTask::PreloadTableTask(const std::string &table_name)
        : GrpcBaseTask(DDL_DML_TASK_GROUP),
          table_name_(table_name) {

}

BaseTaskPtr
PreloadTableTask::Create(const std::string &table_name){
    return std::shared_ptr<GrpcBaseTask>(new PreloadTableTask(table_name));
}

ErrorCode
PreloadTableTask::OnExecute() {
    try {
        TimeRecorder rc("PreloadTableTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        engine::Status stat = DBWrapper::DB()->PreloadTable(table_name_);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DescribeIndexTask::DescribeIndexTask(const std::string &table_name,
                                     ::milvus::grpc::IndexParam *index_param)
    : GrpcBaseTask(DDL_DML_TASK_GROUP),
      table_name_(table_name),
      index_param_(index_param) {

}

BaseTaskPtr
DescribeIndexTask::Create(const std::string &table_name,
                          ::milvus::grpc::IndexParam *index_param){
    return std::shared_ptr<GrpcBaseTask>(new DescribeIndexTask(table_name, index_param));
}

ErrorCode
DescribeIndexTask::OnExecute() {
    try {
        TimeRecorder rc("DescribeIndexTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        engine::TableIndex index;
        engine::Status stat = DBWrapper::DB()->DescribeIndex(table_name_, index);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        index_param_->mutable_table_name()->set_table_name(table_name_);
        index_param_->mutable_index()->set_index_type(index.engine_type_);
        index_param_->mutable_index()->set_nlist(index.nlist_);

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DropIndexTask::DropIndexTask(const std::string &table_name)
    : GrpcBaseTask(DDL_DML_TASK_GROUP),
      table_name_(table_name) {

}

BaseTaskPtr
DropIndexTask::Create(const std::string &table_name){
    return std::shared_ptr<GrpcBaseTask>(new DropIndexTask(table_name));
}

ErrorCode
DropIndexTask::OnExecute() {
    try {
        TimeRecorder rc("DropIndexTask");

        //step 1: check arguments
        ErrorCode res = ValidationUtil::ValidateTableName(table_name_);
        if (res != SERVER_SUCCESS) {
            return SetError(res, "Invalid table name: " + table_name_);
        }

        //step 2: check table existence
        auto stat = DBWrapper::DB()->DropIndex(table_name_);
        if (!stat.ok()) {
            return SetError(DB_META_TRANSACTION_FAILED, stat.ToString());
        }

        rc.ElapseFromBegin("totally cost");
    } catch (std::exception &ex) {
        return SetError(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return SERVER_SUCCESS;
}

}
}
}
}