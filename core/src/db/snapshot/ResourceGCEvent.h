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

#include <boost/filesystem.hpp>
#include <memory>
#include <set>
#include <string>

#include "db/snapshot/MetaEvent.h"
#include "db/snapshot/Operations.h"
#include "db/snapshot/ResourceHelper.h"
#include "db/snapshot/Store.h"
#include "utils/Status.h"

namespace milvus::engine::snapshot {

template <class ResourceT>
class ResourceGCEvent : public GCEvent {
 public:
    using Ptr = std::shared_ptr<ResourceGCEvent>;

    explicit ResourceGCEvent(class ResourceT::Ptr res) : res_(res) {
    }

    ~ResourceGCEvent() = default;

    Status
    Process(StorePtr store) override {
        /* mark resource as 'deleted' in meta */
        auto sd_op = std::make_shared<SoftDeleteOperation<ResourceT>>(res_->GetID());
        STATUS_CHECK((*sd_op)(store));

        /* TODO: physically clean resource */
        auto res_prefix = store->GetRootPath();
        std::string res_path = GetResPath<ResourceT>(res_prefix, res_);
        if (res_path.empty()) {
            /* std::cout << "[GC] No remove action for " << res_->ToString() << std::endl; */
        } else if (boost::filesystem::is_directory(res_path)) {
            auto ok = boost::filesystem::remove_all(res_path);
            std::cout << "[GC] Remove DIR " << res_->ToString() << " " << res_path << " " << ok << std::endl;
        } else if (boost::filesystem::is_regular_file(res_path)) {
            auto ok = boost::filesystem::remove(res_path);
            std::cout << "[GC] Remove FILE " << res_->ToString() << " " << res_path << " " << ok << std::endl;
        } else {
            RemoveWithSuffix<ResourceT>(res_, res_path, store->GetSuffixSet());
        }

        /* remove resource from meta */
        auto hd_op = std::make_shared<HardDeleteOperation<ResourceT>>(res_->GetID());
        STATUS_CHECK((*hd_op)(store));

        return Status::OK();
    }

 private:
    typename ResourceT::Ptr res_;
};

}  // namespace milvus::engine::snapshot
