/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/
#pragma once

#include "utils/Error.h"

#include <cstdint>
#include <string>

namespace zilliz {
namespace milvus {
namespace server {

class Server {
   public:
    static Server* Instance();

    void Init(int64_t daemonized, const std::string& pid_filename, const std::string& config_filename, const std::string &log_config_file);
    int Start();
    void Stop();

   private:
    Server();
    ~Server();

    void Daemonize();

    static void HandleSignal(int signal);
    ErrorCode LoadConfig();

    void StartService();
    void StopService();

   private:
    int64_t daemonized_ = 0;
    int64_t running_ = 1;
    int pid_fd = -1;
    std::string pid_filename_;
    std::string config_filename_;
    std::string log_config_file_;
};  // Server

}   // server
}   // sql
}   // zilliz
