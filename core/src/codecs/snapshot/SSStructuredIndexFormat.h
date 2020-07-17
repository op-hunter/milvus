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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "db/meta/MetaTypes.h"
#include "segment/AttrsIndex.h"
#include "storage/FSHandler.h"

namespace milvus {
namespace codec {

class SSStructuredIndexFormat {
 public:
    SSStructuredIndexFormat() = default;

    void
    read(const storage::FSHandlerPtr& fs_ptr, segment::AttrsIndexPtr& attr_index);

    void
    write(const storage::FSHandlerPtr& fs_ptr, const segment::AttrsIndexPtr& attr_index);

    // No copy and move
    SSStructuredIndexFormat(const SSStructuredIndexFormat&) = delete;
    SSStructuredIndexFormat(SSStructuredIndexFormat&&) = delete;

    SSStructuredIndexFormat&
    operator=(const SSStructuredIndexFormat&) = delete;
    SSStructuredIndexFormat&
    operator=(SSStructuredIndexFormat&&) = delete;

 private:
    void
    read_internal(const milvus::storage::FSHandlerPtr& fs_ptr, const std::string& path, knowhere::IndexPtr& index,
                  engine::meta::hybrid::DataType& attr_type);

    knowhere::IndexPtr
    create_structured_index(const engine::meta::hybrid::DataType data_type);

 private:
    const std::string attr_index_extension_ = ".idx";
};

using SSStructuredIndexFormatPtr = std::shared_ptr<SSStructuredIndexFormat>;

}  // namespace codec
}  // namespace milvus
