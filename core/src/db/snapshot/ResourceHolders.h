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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "db/snapshot/BaseHolders.h"
#include "db/snapshot/ResourceTypes.h"
#include "db/snapshot/Resources.h"
#include "db/snapshot/ScopedResource.h"

namespace milvus {
namespace engine {
namespace snapshot {

class CollectionsHolder : public ResourceHolder<Collection, CollectionsHolder> {
 public:
    using BaseT = ResourceHolder<Collection, CollectionsHolder>;
    using ResourcePtr = typename BaseT::ResourcePtr;
    using NameMapT = std::map<std::string, ResourcePtr>;

    ScopedT
    GetCollection(const std::string& name, bool scoped = true);

    bool
    Add(ResourcePtr resource) override;
    bool
    Release(ID_TYPE id) override;
    bool
    Release(const std::string& name);

    void
    Reset() override;

 private:
    ResourcePtr
    Load(const std::string& name) override;

    NameMapT name_map_;
};

class SchemaCommitsHolder : public ResourceHolder<SchemaCommit, SchemaCommitsHolder> {};

class FieldCommitsHolder : public ResourceHolder<FieldCommit, FieldCommitsHolder> {};

class FieldsHolder : public ResourceHolder<Field, FieldsHolder> {};

class FieldElementsHolder : public ResourceHolder<FieldElement, FieldElementsHolder> {};

class CollectionCommitsHolder : public ResourceHolder<CollectionCommit, CollectionCommitsHolder> {};

class PartitionsHolder : public ResourceHolder<Partition, PartitionsHolder> {};

class PartitionCommitsHolder : public ResourceHolder<PartitionCommit, PartitionCommitsHolder> {};

class SegmentsHolder : public ResourceHolder<Segment, SegmentsHolder> {};

class SegmentCommitsHolder : public ResourceHolder<SegmentCommit, SegmentCommitsHolder> {};

class SegmentFilesHolder : public ResourceHolder<SegmentFile, SegmentFilesHolder> {};

}  // namespace snapshot
}  // namespace engine
}  // namespace milvus
