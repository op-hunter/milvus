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
#include <mutex>
#include <thread>
#include "Operations.h"
#include "Store.h"
#include "utils/BlockingQueue.h"

namespace milvus::engine::snapshot {

using ThreadPtr = std::shared_ptr<std::thread>;
using OperationQueue = BlockingQueue<OperationsPtr>;

class OperationExecutor {
 public:
    OperationExecutor() = default;
    OperationExecutor(const OperationExecutor&) = delete;

    ~OperationExecutor() {
        Stop();
    }

    static OperationExecutor&
    GetInstance() {
        static OperationExecutor executor;
        return executor;
    }

    Status
    Submit(const OperationsPtr& operation, bool sync = true) {
        if (!operation) {
            return Status(SS_INVALID_ARGUMENT_ERROR, "Invalid Operation");
        }
        /* Store::GetInstance().Apply(*operation); */
        /* return true; */
        Enqueue(operation);
        if (sync) {
            return operation->WaitToFinish();
        }
        return Status::OK();
    }

    void
    Start() {
        if (thread_ptr_ == nullptr) {
            thread_ptr_ = std::make_shared<std::thread>(&OperationExecutor::ThreadMain, this);
        }
    }

    void
    Stop() {
        if (thread_ptr_ != nullptr) {
            Enqueue(nullptr);
            thread_ptr_->join();
            thread_ptr_ = nullptr;
            std::cout << "OperationExecutor Stopped" << std::endl;
        }
    }

 private:
    void
    ThreadMain() {
        while (true) {
            OperationsPtr operation = queue_.Take();
            if (!operation) {
                break;
            }
            /* std::cout << std::this_thread::get_id() << " Dequeue Operation " << operation->GetID() << std::endl; */
            Store::GetInstance().Apply(*operation);
        }
    }

    void
    Enqueue(const OperationsPtr& operation) {
        /* std::cout << std::this_thread::get_id() << " Enqueue Operation " << operation->GetID() << std::endl; */
        queue_.Put(operation);
    }

 private:
    ThreadPtr thread_ptr_ = nullptr;
    OperationQueue queue_;
};

}  // namespace milvus::engine::snapshot
