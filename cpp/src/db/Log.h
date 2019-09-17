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

#include "utils/easylogging++.h"

namespace zilliz {
namespace milvus {
namespace engine {

#define ENGINE_DOMAIN_NAME "[ENGINE] "
#define ENGINE_ERROR_TEXT "ENGINE Error:"

#define ENGINE_LOG_TRACE LOG(TRACE) << ENGINE_DOMAIN_NAME
#define ENGINE_LOG_DEBUG LOG(DEBUG) << ENGINE_DOMAIN_NAME
#define ENGINE_LOG_INFO LOG(INFO) << ENGINE_DOMAIN_NAME
#define ENGINE_LOG_WARNING LOG(WARNING) << ENGINE_DOMAIN_NAME
#define ENGINE_LOG_ERROR LOG(ERROR) << ENGINE_DOMAIN_NAME
#define ENGINE_LOG_FATAL LOG(FATAL) << ENGINE_DOMAIN_NAME

} // namespace sql
} // namespace zilliz
} // namespace server
