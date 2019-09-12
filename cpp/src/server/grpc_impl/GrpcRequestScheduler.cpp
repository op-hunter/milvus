/*******************************************************************************
* Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
******************************************************************************/
#include "GrpcRequestScheduler.h"
#include "utils/Log.h"

#include "src/grpc/gen-status/status.pb.h"

namespace zilliz {
namespace milvus {
namespace server {
namespace grpc {

using namespace ::milvus;

namespace {
    ::milvus::grpc::ErrorCode ErrorMap(ErrorCode code) {
        static const std::map<ErrorCode, ::milvus::grpc::ErrorCode> code_map = {
                {SERVER_UNEXPECTED_ERROR,         ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
                {SERVER_UNSUPPORTED_ERROR,        ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
                {SERVER_NULL_POINTER,             ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
                {SERVER_INVALID_ARGUMENT,         ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
                {SERVER_FILE_NOT_FOUND,           ::milvus::grpc::ErrorCode::FILE_NOT_FOUND},
                {SERVER_NOT_IMPLEMENT,            ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
                {SERVER_BLOCKING_QUEUE_EMPTY,     ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
                {SERVER_CANNOT_CREATE_FOLDER,     ::milvus::grpc::ErrorCode::CANNOT_CREATE_FOLDER},
                {SERVER_CANNOT_CREATE_FILE,       ::milvus::grpc::ErrorCode::CANNOT_CREATE_FILE},
                {SERVER_CANNOT_DELETE_FOLDER,     ::milvus::grpc::ErrorCode::CANNOT_DELETE_FOLDER},
                {SERVER_CANNOT_DELETE_FILE,       ::milvus::grpc::ErrorCode::CANNOT_DELETE_FILE},
                {SERVER_TABLE_NOT_EXIST,          ::milvus::grpc::ErrorCode::TABLE_NOT_EXISTS},
                {SERVER_INVALID_TABLE_NAME,       ::milvus::grpc::ErrorCode::ILLEGAL_TABLE_NAME},
                {SERVER_INVALID_TABLE_DIMENSION,  ::milvus::grpc::ErrorCode::ILLEGAL_DIMENSION},
                {SERVER_INVALID_TIME_RANGE,       ::milvus::grpc::ErrorCode::ILLEGAL_RANGE},
                {SERVER_INVALID_VECTOR_DIMENSION, ::milvus::grpc::ErrorCode::ILLEGAL_DIMENSION},

                {SERVER_INVALID_INDEX_TYPE,       ::milvus::grpc::ErrorCode::ILLEGAL_INDEX_TYPE},
                {SERVER_INVALID_ROWRECORD,        ::milvus::grpc::ErrorCode::ILLEGAL_ROWRECORD},
                {SERVER_INVALID_ROWRECORD_ARRAY,  ::milvus::grpc::ErrorCode::ILLEGAL_ROWRECORD},
                {SERVER_INVALID_TOPK,             ::milvus::grpc::ErrorCode::ILLEGAL_TOPK},
                {SERVER_INVALID_NPROBE,           ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
                {SERVER_INVALID_INDEX_NLIST,      ::milvus::grpc::ErrorCode::ILLEGAL_NLIST},
                {SERVER_INVALID_INDEX_METRIC_TYPE,::milvus::grpc::ErrorCode::ILLEGAL_METRIC_TYPE},
                {SERVER_INVALID_INDEX_FILE_SIZE,  ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
                {SERVER_ILLEGAL_VECTOR_ID,        ::milvus::grpc::ErrorCode::ILLEGAL_VECTOR_ID},
                {SERVER_ILLEGAL_SEARCH_RESULT,    ::milvus::grpc::ErrorCode::ILLEGAL_SEARCH_RESULT},
                {SERVER_CACHE_ERROR,              ::milvus::grpc::ErrorCode::CACHE_FAILED},
                {DB_META_TRANSACTION_FAILED,      ::milvus::grpc::ErrorCode::META_FAILED},
                {SERVER_BUILD_INDEX_ERROR,        ::milvus::grpc::ErrorCode::BUILD_INDEX_ERROR},
                {SERVER_OUT_OF_MEMORY,            ::milvus::grpc::ErrorCode::OUT_OF_MEMORY},
        };

        if(code_map.find(code) != code_map.end()) {
            return code_map.at(code);
        } else {
            return ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GrpcBaseTask::GrpcBaseTask(const std::string &task_group, bool async)
        : task_group_(task_group),
          async_(async),
          done_(false),
          error_code_(SERVER_SUCCESS) {

}

GrpcBaseTask::~GrpcBaseTask() {
    WaitToFinish();
}

ErrorCode GrpcBaseTask::Execute() {
    error_code_ = OnExecute();
    Done();
    return error_code_;
}

void GrpcBaseTask::Done() {
    done_ = true;
    finish_cond_.notify_all();
}

ErrorCode GrpcBaseTask::SetError(ErrorCode error_code, const std::string &error_msg) {
    error_code_ = error_code;
    error_msg_ = error_msg;

    SERVER_LOG_ERROR << error_msg_;
    return error_code_;
}

ErrorCode GrpcBaseTask::WaitToFinish() {
    std::unique_lock<std::mutex> lock(finish_mtx_);
    finish_cond_.wait(lock, [this] { return done_; });

    return error_code_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GrpcRequestScheduler::GrpcRequestScheduler()
        : stopped_(false) {
    Start();
}

GrpcRequestScheduler::~GrpcRequestScheduler() {
    Stop();
}

void GrpcRequestScheduler::ExecTask(BaseTaskPtr &task_ptr, ::milvus::grpc::Status *grpc_status) {
    if (task_ptr == nullptr) {
        return;
    }

    GrpcRequestScheduler &scheduler = GrpcRequestScheduler::GetInstance();
    scheduler.ExecuteTask(task_ptr);

    if (!task_ptr->IsAsync()) {
        task_ptr->WaitToFinish();
        ErrorCode err = task_ptr->ErrorID();
        if (err != SERVER_SUCCESS) {
            grpc_status->set_reason(task_ptr->ErrorMsg());
            grpc_status->set_error_code(ErrorMap(err));
        }
    }
}

void GrpcRequestScheduler::Start() {
    if (!stopped_) {
        return;
    }

    stopped_ = false;
}

void GrpcRequestScheduler::Stop() {
    if (stopped_) {
        return;
    }

    SERVER_LOG_INFO << "Scheduler gonna stop...";
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        for (auto iter : task_groups_) {
            if (iter.second != nullptr) {
                iter.second->Put(nullptr);
            }
        }
    }

    for (auto iter : execute_threads_) {
        if (iter == nullptr)
            continue;

        iter->join();
    }
    stopped_ = true;
    SERVER_LOG_INFO << "Scheduler stopped";
}

ErrorCode GrpcRequestScheduler::ExecuteTask(const BaseTaskPtr &task_ptr) {
    if (task_ptr == nullptr) {
        return SERVER_NULL_POINTER;
    }

    ErrorCode err = PutTaskToQueue(task_ptr);
    if (err != SERVER_SUCCESS) {
        SERVER_LOG_ERROR << "Put task to queue failed with code: " << err;
        return err;
    }

    if (task_ptr->IsAsync()) {
        return SERVER_SUCCESS;//async execution, caller need to call WaitToFinish at somewhere
    }

    return task_ptr->WaitToFinish();//sync execution
}


void GrpcRequestScheduler::TakeTaskToExecute(TaskQueuePtr task_queue) {
    if (task_queue == nullptr) {
        return;
    }

    while (true) {
        BaseTaskPtr task = task_queue->Take();
        if (task == nullptr) {
            SERVER_LOG_ERROR << "Take null from task queue, stop thread";
            break;//stop the thread
        }

        try {
            ErrorCode err = task->Execute();
            if (err != SERVER_SUCCESS) {
                SERVER_LOG_ERROR << "Task failed with code: " << err;
            }
        } catch (std::exception &ex) {
            SERVER_LOG_ERROR << "Task failed to execute: " << ex.what();
        }
    }
}

ErrorCode GrpcRequestScheduler::PutTaskToQueue(const BaseTaskPtr &task_ptr) {
    std::lock_guard<std::mutex> lock(queue_mtx_);

    std::string group_name = task_ptr->TaskGroup();
    if (task_groups_.count(group_name) > 0) {
        task_groups_[group_name]->Put(task_ptr);
    } else {
        TaskQueuePtr queue = std::make_shared<TaskQueue>();
        queue->Put(task_ptr);
        task_groups_.insert(std::make_pair(group_name, queue));

        //start a thread
        ThreadPtr thread = std::make_shared<std::thread>(&GrpcRequestScheduler::TakeTaskToExecute, this, queue);
        execute_threads_.push_back(thread);
        SERVER_LOG_INFO << "Create new thread for task group: " << group_name;
    }

    return SERVER_SUCCESS;
}

}
}
}
}
