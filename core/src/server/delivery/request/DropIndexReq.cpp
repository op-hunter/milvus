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

#include "server/delivery/request/DropIndexReq.h"
#include "server/DBWrapper.h"
#include "server/ValidationUtil.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

#include <fiu/fiu-local.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace milvus {
namespace server {

DropIndexReq::DropIndexReq(const ContextPtr& context, const std::string& collection_name, const std::string& field_name,
                           const std::string& index_name)
    : BaseReq(context, ReqType::kDropIndex),
      collection_name_(collection_name),
      field_name_(field_name),
      index_name_(index_name) {
}

BaseReqPtr
DropIndexReq::Create(const ContextPtr& context, const std::string& collection_name, const std::string& field_name,
                     const std::string& index_name) {
    return std::shared_ptr<BaseReq>(new DropIndexReq(context, collection_name, field_name, index_name));
}

Status
DropIndexReq::OnExecute() {
    try {
        fiu_do_on("DropIndexReq.OnExecute.throw_std_exception", throw std::exception());
        std::string hdr = "DropIndexReq(collection=" + collection_name_ + ")";
        TimeRecorderAuto rc(hdr);

        bool exist = false;
        STATUS_CHECK(DBWrapper::DB()->HasCollection(collection_name_, exist));
        if (!exist) {
            return Status(SERVER_COLLECTION_NOT_EXIST, "Collection not exist: " + collection_name_);
        }

        // step 2: drop index
        auto status = DBWrapper::DB()->DropIndex(collection_name_, field_name_);
        fiu_do_on("DropIndexReq.OnExecute.drop_index_fail", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
        rc.ElapseFromBegin("done");
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
