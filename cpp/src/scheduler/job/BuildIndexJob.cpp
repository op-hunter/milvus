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

#include "BuildIndexJob.h"
#include "utils/Log.h"


namespace milvus {
namespace scheduler {

BuildIndexJob::BuildIndexJob(JobId id, engine::meta::MetaPtr meta_ptr, engine::DBOptions options)
    : Job(id, JobType::BUILD), meta_ptr_(std::move(meta_ptr)), options_(std::move(options)) {

}

bool
BuildIndexJob::AddToIndexFiles(const engine::meta::TableFileSchemaPtr &to_index_file) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (to_index_file == nullptr || to_index_files_.find(to_index_file->id_) != to_index_files_.end()) {
        return false;
    }

    SERVER_LOG_DEBUG << "BuildIndexJob " << id() << " add to_index file: " << to_index_file->id_;

    to_index_files_[to_index_file->id_] = to_index_file;
}

Status&
BuildIndexJob::WaitBuildIndexFinish() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return to_index_files_.empty(); });
    SERVER_LOG_DEBUG << "BuildIndexJob " << id() << " all done";
}

void
BuildIndexJob::BuildIndexDone(size_t to_index_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    to_index_files_.erase(to_index_id);
    cv_.notify_all();
    SERVER_LOG_DEBUG << "BuildIndexJob " << id() << " finish index file: " << to_index_id;
}


}
}