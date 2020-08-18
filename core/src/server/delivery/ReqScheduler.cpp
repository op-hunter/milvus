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

#include "server/delivery/ReqScheduler.h"
#include "utils/Log.h"

#include <fiu/fiu-local.h>
#include <unistd.h>
#include <utility>

namespace milvus {
namespace server {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReqScheduler::ReqScheduler() : stopped_(false) {
    Start();
}

ReqScheduler::~ReqScheduler() {
    Stop();
}

void
ReqScheduler::ExecReq(const BaseReqPtr& req_ptr) {
    if (req_ptr == nullptr) {
        return;
    }

    ReqScheduler& scheduler = ReqScheduler::GetInstance();
    scheduler.ExecuteReq(req_ptr);
}

void
ReqScheduler::Start() {
    if (!stopped_) {
        return;
    }

    stopped_ = false;
}

void
ReqScheduler::Stop() {
    if (stopped_ && req_groups_.empty() && execute_threads_.empty()) {
        return;
    }

    LOG_SERVER_INFO_ << "Scheduler gonna stop...";
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        for (auto& iter : req_groups_) {
            if (iter.second != nullptr) {
                iter.second->Put(nullptr);
            }
        }
    }

    for (auto& iter : execute_threads_) {
        if (iter == nullptr)
            continue;
        iter->join();
    }
    req_groups_.clear();
    execute_threads_.clear();
    stopped_ = true;
    LOG_SERVER_INFO_ << "Scheduler stopped";
}

Status
ReqScheduler::ExecuteReq(const BaseReqPtr& req_ptr) {
    if (req_ptr == nullptr) {
        return Status::OK();
    }

    auto status = req_ptr->PreExecute();
    if (!status.ok()) {
        req_ptr->Done();
        return status;
    }

    status = PutToQueue(req_ptr);
    fiu_do_on("ReqScheduler.ExecuteReq.push_queue_fail", status = Status(SERVER_INVALID_ARGUMENT, ""));

    if (!status.ok()) {
        LOG_SERVER_ERROR_ << "Put request to queue failed with code: " << status.ToString();
        req_ptr->Done();
        return status;
    }

    if (req_ptr->async()) {
        return Status::OK();  // async execution, caller need to call WaitToFinish at somewhere
    }

    status = req_ptr->WaitToFinish();  // sync execution
    if (!status.ok()) {
        return status;
    }

    return req_ptr->PostExecute();
}

void
ReqScheduler::TakeToExecute(ReqQueuePtr req_queue) {
    SetThreadName("reqsched_thread");
    if (req_queue == nullptr) {
        return;
    }

    while (true) {
        BaseReqPtr req = req_queue->TakeReq();
        if (req == nullptr) {
            LOG_SERVER_ERROR_ << "Take null from request queue, stop thread";
            break;  // stop the thread
        }

        try {
            fiu_do_on("ReqScheduler.TakeToExecute.throw_std_exception1", throw std::exception());
            auto status = req->Execute();
            fiu_do_on("ReqScheduler.TakeToExecute.throw_std_exception", throw std::exception());
            fiu_do_on("ReqScheduler.TakeToExecute.execute_fail", status = Status(SERVER_INVALID_ARGUMENT, ""));
            if (!status.ok()) {
                LOG_SERVER_ERROR_ << "Req failed with code: " << status.ToString();
            }
        } catch (std::exception& ex) {
            LOG_SERVER_ERROR_ << "Req failed to execute: " << ex.what();
        }
    }
}

Status
ReqScheduler::PutToQueue(const BaseReqPtr& req_ptr) {
    std::lock_guard<std::mutex> lock(queue_mtx_);

    std::string group_name = req_ptr->req_group();
    if (req_groups_.count(group_name) > 0) {
        req_groups_[group_name]->PutReq(req_ptr);
    } else {
        ReqQueuePtr queue = std::make_shared<ReqQueue>();
        queue->PutReq(req_ptr);
        req_groups_.insert(std::make_pair(group_name, queue));
        fiu_do_on("ReqScheduler.PutToQueue.null_queue", queue = nullptr);

        // start a thread
        ThreadPtr thread = std::make_shared<std::thread>(&ReqScheduler::TakeToExecute, this, queue);

        fiu_do_on("ReqScheduler.PutToQueue.push_null_thread", execute_threads_.push_back(nullptr));
        execute_threads_.push_back(thread);
        LOG_SERVER_INFO_ << "Create new thread for request group: " << group_name;
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
