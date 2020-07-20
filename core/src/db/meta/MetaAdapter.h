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
#include <type_traits>
#include <vector>

#include "db/meta/MetaSession.h"
#include "db/meta/backend/MockMetaEngine.h"
#include "db/meta/backend/MySqlEngine.h"
#include "db/snapshot/Resources.h"
#include "utils/Exception.h"

namespace milvus::engine::meta {

class MetaAdapter {
 public:
    explicit MetaAdapter(MetaEnginePtr engine) : engine_(engine) {
    }

    SessionPtr
    CreateSession() {
        return std::make_shared<MetaSession>(engine_);
    }

    template <typename T>
    Status
    Select(int64_t id, typename T::Ptr& resource) {
        // TODO move select logic to here
        auto session = CreateSession();
        std::vector<typename T::Ptr> resources;
        auto status = session->Select<T, snapshot::ID_TYPE>(snapshot::IdField::Name, id, {}, resources);
        if (status.ok() && !resources.empty()) {
            // TODO: may need to check num of resources
            resource = resources.at(0);
        }

        return status;
    }

    template <typename ResourceT, typename U>
    Status
    SelectBy(const std::string& field, const U& value, std::vector<typename ResourceT::Ptr>& resources) {
        auto session = CreateSession();
        return session->Select<ResourceT, U>(field, value, {}, resources);
    }

    template <typename ResourceT, typename U>
    Status
    SelectResourceIDs(std::vector<int64_t>& ids, const std::string& filter_field, const U& filter_value) {
        std::vector<typename ResourceT::Ptr> resources;
        auto session = CreateSession();
        std::vector<std::string> target_attrs = {F_ID};
        auto status = session->Select<ResourceT, U>(filter_field, filter_value, target_attrs, resources);
        if (!status.ok()) {
            return status;
        }

        for (auto& res : resources) {
            ids.push_back(res->GetID());
        }

        return Status::OK();
    }

    template <typename ResourceT>
    Status
    Apply(snapshot::ResourceContextPtr<ResourceT> resp, int64_t& result_id) {
        result_id = 0;

        auto session = CreateSession();
        session->Apply<ResourceT>(resp);

        std::vector<int64_t> result_ids;
        auto status = session->Commit(result_ids);

        if (!status.ok()) {
            throw Exception(status.code(), status.message());
        }

        if (result_ids.size() != 1) {
            throw Exception(1, "Result id is not equal one ... ");
        }

        result_id = result_ids.at(0);
        return Status::OK();
    }

    Status
    TruncateAll() {
        return engine_->TruncateAll();
    }

 private:
    MetaEnginePtr engine_;
};

using MetaAdapterPtr = std::shared_ptr<MetaAdapter>;

}  // namespace milvus::engine::meta
