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

#include "utils/SignalHandler.h"
#include "utils/Log.h"

#include <execinfo.h>
#include <signal.h>
#include <string>

namespace milvus {
namespace server {

void
SignalHandler::HandleSignal(int signum) {
    switch (signum) {
        case SIGINT:
        case SIGUSR2: {
            LOG_SERVER_INFO_ << "Server received signal: " << signum;
            if (routine_func_ != nullptr) {
                (*routine_func_)();
            }
            exit(0);
        }
        default: {
            LOG_SERVER_INFO_ << "Server received critical signal: " << signum;
            SignalHandler::PrintStacktrace();
            if (routine_func_ != nullptr) {
                (*routine_func_)();
            }
            exit(1);
        }
    }
}

void
SignalHandler::PrintStacktrace() {
    LOG_SERVER_INFO_ << "Call stack:";

    const int size = 32;
    void* array[size];
    int stack_num = backtrace(array, size);
    char** stacktrace = backtrace_symbols(array, stack_num);
    for (int i = 0; i < stack_num; ++i) {
        std::string info = stacktrace[i];
        LOG_SERVER_INFO_ << info;
    }
    free(stacktrace);
}

}  // namespace server
}  // namespace milvus
