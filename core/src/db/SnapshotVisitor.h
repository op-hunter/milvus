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

#include "db/meta/FilesHolder.h"
#include "db/snapshot/Snapshot.h"

#include <memory>
#include <string>

namespace milvus {
namespace engine {

class SnapshotVisitor {
 public:
    explicit SnapshotVisitor(snapshot::ScopedSnapshotT ss);
    explicit SnapshotVisitor(const std::string& collection_name);
    explicit SnapshotVisitor(snapshot::ID_TYPE collection_id);

    Status
    SegmentsToSearch(meta::FilesHolder& files_holder);

 protected:
    snapshot::ScopedSnapshotT ss_;
    Status status_;
};

}  // namespace engine
}  // namespace milvus
