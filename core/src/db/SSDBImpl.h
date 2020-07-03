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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "db/Options.h"
#include "db/SimpleWaitNotify.h"
#include "db/SnapshotHandlers.h"
#include "db/snapshot/Context.h"
#include "db/snapshot/ResourceTypes.h"
#include "db/snapshot/Resources.h"
#include "utils/Status.h"
#include "wal/WalManager.h"

namespace milvus {
namespace engine {

class SSDBImpl {
 public:
    explicit SSDBImpl(const DBOptions& options);

    ~SSDBImpl();

    Status
    Start();

    Status
    Stop();

    Status
    CreateCollection(const snapshot::CreateCollectionContext& context);

    Status
    DropCollection(const std::string& name);

    Status
    DescribeCollection(const std::string& collection_name, snapshot::CollectionPtr& collection,
                       std::map<snapshot::FieldPtr, std::vector<snapshot::FieldElementPtr>>& fields_schema);

    Status
    HasCollection(const std::string& collection_name, bool& has_or_not);

    Status
    AllCollections(std::vector<std::string>& names);

    Status
    GetCollectionRowCount(const std::string& collection_name, uint64_t& row_count);

    Status
    PreloadCollection(const std::shared_ptr<server::Context>& context, const std::string& collection_name,
                      bool force = false);

    Status
    CreatePartition(const std::string& collection_name, const std::string& partition_name);

    Status
    DropPartition(const std::string& collection_name, const std::string& partition_name);

    Status
    ShowPartitions(const std::string& collection_name, std::vector<std::string>& partition_names);

 private:
    void
    InternalFlush(const std::string& collection_id = "");

    void
    BackgroundFlushThread();

    void
    StartMetricTask();

    void
    BackgroundMetricThread();

    Status
    ExecWalRecord(const wal::MXLogRecord& record);

    void
    BackgroundWalThread();

 private:
    DBOptions options_;
    std::atomic<bool> initialized_;

    std::shared_ptr<wal::WalManager> wal_mgr_;
    std::shared_ptr<std::thread> bg_wal_thread_;

    std::shared_ptr<std::thread> bg_flush_thread_;

    SimpleWaitNotify swn_wal_;
    SimpleWaitNotify swn_flush_;
    SimpleWaitNotify swn_metric_;

    SimpleWaitNotify flush_req_swn_;
};  // SSDBImpl

}  // namespace engine
}  // namespace milvus
