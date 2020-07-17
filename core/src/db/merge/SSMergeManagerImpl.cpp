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

#include "db/merge/SSMergeManagerImpl.h"
#include "db/merge/SSMergeSimpleStrategy.h"
#include "db/merge/SSMergeTask.h"
#include "db/snapshot/Snapshots.h"
#include "utils/Exception.h"
#include "utils/Log.h"

#include <map>

namespace milvus {
namespace engine {

SSMergeManagerImpl::SSMergeManagerImpl(const DBOptions& options, MergeStrategyType type)
    : options_(options), strategy_type_(type) {
    UseStrategy(type);
}

Status
SSMergeManagerImpl::UseStrategy(MergeStrategyType type) {
    switch (type) {
        case MergeStrategyType::SIMPLE: {
            strategy_ = std::make_shared<SSMergeSimpleStrategy>();
            break;
        }
        case MergeStrategyType::LAYERED:
        case MergeStrategyType::ADAPTIVE:
        default: {
            std::string msg = "Unsupported merge strategy type: " + std::to_string((int32_t)type);
            LOG_ENGINE_ERROR_ << msg;
            throw Exception(DB_ERROR, msg);
        }
    }
    strategy_type_ = type;

    return Status::OK();
}

Status
SSMergeManagerImpl::MergeFiles(const std::string& collection_name) {
    if (strategy_ == nullptr) {
        std::string msg = "No merge strategy specified";
        LOG_ENGINE_ERROR_ << msg;
        return Status(DB_ERROR, msg);
    }

    int64_t row_count_per_segment = DEFAULT_ROW_COUNT_PER_SEGMENT;
    while (true) {
        snapshot::ScopedSnapshotT latest_ss;
        STATUS_CHECK(snapshot::Snapshots::GetInstance().GetSnapshot(latest_ss, collection_name));

        Partition2SegmentsMap part2seg;
        auto& segments = latest_ss->GetResources<snapshot::Segment>();
        for (auto& kv : segments) {
            auto segment_commit = latest_ss->GetSegmentCommitBySegmentId(kv.second->GetID());
            part2seg[kv.second->GetPartitionId()].push_back(kv.second->GetID());
        }

        Partition2SegmentsMap::iterator it;
        for (it = part2seg.begin(); it != part2seg.end();) {
            if (it->second.size() <= 1) {
                part2seg.erase(it++);
            } else {
                it++;
            }
        }

        if (part2seg.empty()) {
            break;
        }

        SegmentGroups segment_groups;
        auto status = strategy_->RegroupSegments(latest_ss, part2seg, segment_groups);
        if (!status.ok()) {
            LOG_ENGINE_ERROR_ << "Failed to regroup segments for: " << collection_name
                              << ", continue to merge all files into one";
            return status;
        }

        for (auto& segments : segment_groups) {
            SSMergeTask task(options_, latest_ss, segments);
            task.Execute();
        }
    }

    return Status::OK();
}

}  // namespace engine
}  // namespace milvus
