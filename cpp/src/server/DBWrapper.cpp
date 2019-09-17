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


#include "DBWrapper.h"
#include "ServerConfig.h"
#include "db/DBFactory.h"
#include "utils/CommonUtil.h"
#include "utils/Log.h"
#include "utils/StringHelpFunctions.h"

#include <omp.h>

namespace zilliz {
namespace milvus {
namespace server {

DBWrapper::DBWrapper() {

}

ErrorCode DBWrapper::StartService() {
    //db config
    zilliz::milvus::engine::Options opt;
    ConfigNode& db_config = ServerConfig::GetInstance().GetConfig(CONFIG_DB);
    opt.meta.backend_uri = db_config.GetValue(CONFIG_DB_URL);
    std::string db_path = db_config.GetValue(CONFIG_DB_PATH);
    opt.meta.path = db_path + "/db";

    std::string db_slave_path = db_config.GetValue(CONFIG_DB_SLAVE_PATH);
    StringHelpFunctions::SplitStringByDelimeter(db_slave_path, ";", opt.meta.slave_paths);

    // cache config
    ConfigNode& cache_config = ServerConfig::GetInstance().GetConfig(CONFIG_CACHE);
    opt.insert_cache_immediately_ = cache_config.GetBoolValue(CONFIG_INSERT_CACHE_IMMEDIATELY, false);

    ConfigNode& serverConfig = ServerConfig::GetInstance().GetConfig(CONFIG_SERVER);
    std::string mode = serverConfig.GetValue(CONFIG_CLUSTER_MODE, "single");
    if (mode == "single") {
        opt.mode = zilliz::milvus::engine::Options::MODE::SINGLE;
    }
    else if (mode == "cluster") {
        opt.mode = zilliz::milvus::engine::Options::MODE::CLUSTER;
    }
    else if (mode == "read_only") {
        opt.mode = zilliz::milvus::engine::Options::MODE::READ_ONLY;
    }
    else {
        std::cout << "ERROR: mode specified in server_config is not one of ['single', 'cluster', 'read_only']" << std::endl;
        kill(0, SIGUSR1);
    }

    // engine config
    ConfigNode& engine_config = ServerConfig::GetInstance().GetConfig(CONFIG_ENGINE);
    int32_t omp_thread = engine_config.GetInt32Value(CONFIG_OMP_THREAD_NUM, 0);
    if(omp_thread > 0) {
        omp_set_num_threads(omp_thread);
        SERVER_LOG_DEBUG << "Specify openmp thread number: " << omp_thread;
    } else {
        uint32_t sys_thread_cnt = 8;
        if(CommonUtil::GetSystemAvailableThreads(sys_thread_cnt)) {
            omp_thread = (int32_t)ceil(sys_thread_cnt*0.5);
            omp_set_num_threads(omp_thread);
        }
    }

    //set archive config
    engine::ArchiveConf::CriteriaT criterial;
    int64_t disk = db_config.GetInt64Value(CONFIG_DB_ARCHIVE_DISK, 0);
    int64_t days = db_config.GetInt64Value(CONFIG_DB_ARCHIVE_DAYS, 0);
    if(disk > 0) {
        criterial[engine::ARCHIVE_CONF_DISK] = disk;
    }
    if(days > 0) {
        criterial[engine::ARCHIVE_CONF_DAYS] = days;
    }
    opt.meta.archive_conf.SetCriterias(criterial);

    //create db root folder
    ErrorCode err = CommonUtil::CreateDirectory(opt.meta.path);
    if(err != SERVER_SUCCESS) {
        std::cout << "ERROR! Failed to create database root path: " << opt.meta.path << std::endl;
        kill(0, SIGUSR1);
    }

    for(auto& path : opt.meta.slave_paths) {
        err = CommonUtil::CreateDirectory(path);
        if(err != SERVER_SUCCESS) {
            std::cout << "ERROR! Failed to create database slave path: " << path << std::endl;
            kill(0, SIGUSR1);
        }
    }

    //create db instance
    std::string msg = opt.meta.path;
    try {
        db_ = engine::DBFactory::Build(opt);
    } catch(std::exception& ex) {
        msg = ex.what();
    }

    if(db_ == nullptr) {
        std::cout << "ERROR! Failed to open database: " << msg << std::endl;
        kill(0, SIGUSR1);
    }

    db_->Start();

    return SERVER_SUCCESS;
}

ErrorCode DBWrapper::StopService() {
    if(db_) {
        db_->Stop();
    }

    return SERVER_SUCCESS;
}

}
}
}