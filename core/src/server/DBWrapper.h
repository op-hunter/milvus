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

#include <string>

#include "db/DB.h"
#include "utils/Status.h"

namespace milvus {
namespace server {

class DBWrapper {
 private:
    DBWrapper() = default;
    ~DBWrapper() = default;

 public:
    static DBWrapper&
    GetInstance() {
        static DBWrapper wrapper;
        return wrapper;
    }

    static engine::DBPtr
    DB() {
        return GetInstance().EngineDB();
    }

    Status
    StartService();
    Status
    StopService();

    engine::DBPtr
    EngineDB() {
        return db_;
    }

 private:
    Status
    PreloadCollections(const std::string& preload_collections);

 private:
    engine::DBPtr db_;
};

}  // namespace server
}  // namespace milvus
