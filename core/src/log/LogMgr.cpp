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

#include <fiu/fiu-local.h>
#include <libgen.h>
#include <cctype>
#include <string>

#include <boost/filesystem.hpp>

#include "config/ServerConfig.h"
#include "log/LogMgr.h"
#include "utils/Status.h"

namespace milvus {

namespace {
static int global_idx = 0;
static int debug_idx = 0;
static int warning_idx = 0;
static int trace_idx = 0;
static int error_idx = 0;
static int fatal_idx = 0;
static int64_t logs_delete_exceeds = 1;
static bool enable_log_delete = false;

/* module constant */
const int64_t CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN = 536870912;  /* 512 MB */
const int64_t CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX = 4294967296; /* 4 GB */
const int64_t CONFIG_LOGS_LOG_ROTATE_NUM_MIN = 0;
const int64_t CONFIG_LOGS_LOG_ROTATE_NUM_MAX = 1024;
}  // namespace

// TODO(yzb) : change the easylogging library to get the log level from parameter rather than filename
void
RolloutHandler(const char* filename, std::size_t size, el::Level level) {
    char* dirc = strdup(filename);
    char* basec = strdup(filename);
    char* dir = dirname(dirc);
    char* base = basename(basec);

    std::string s(base);
    std::string list[] = {"\\", " ", "\'", "\"", "*", "\?", "{", "}", ";", "<",
                          ">",  "|", "^",  "&",  "$", "#",  "!", "`", "~"};
    std::string::size_type position;
    for (auto substr : list) {
        position = 0;
        while ((position = s.find_first_of(substr, position)) != std::string::npos) {
            s.insert(position, "\\");
            position += 2;
        }
    }
    std::string m(std::string(dir) + "/" + s);
    s = m;
    try {
        switch (level) {
            case el::Level::Debug: {
                s.append("." + std::to_string(++debug_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && debug_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(debug_idx - logs_delete_exceeds);
                    // std::cout << "remote " << to_delete << std::endl;
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
            case el::Level::Warning: {
                s.append("." + std::to_string(++warning_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && warning_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(warning_idx - logs_delete_exceeds);
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
            case el::Level::Trace: {
                s.append("." + std::to_string(++trace_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && trace_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(trace_idx - logs_delete_exceeds);
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
            case el::Level::Error: {
                s.append("." + std::to_string(++error_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && error_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(error_idx - logs_delete_exceeds);
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
            case el::Level::Fatal: {
                s.append("." + std::to_string(++fatal_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && fatal_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(fatal_idx - logs_delete_exceeds);
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
            default: {
                s.append("." + std::to_string(++global_idx));
                rename(m.c_str(), s.c_str());
                if (enable_log_delete && global_idx - logs_delete_exceeds > 0) {
                    std::string to_delete = m + "." + std::to_string(global_idx - logs_delete_exceeds);
                    boost::filesystem::remove(to_delete);
                }
                break;
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << ". Exception throws from RolloutHandler." << std::endl;
    }
}

Status
LogMgr::InitLog(bool trace_enable, bool debug_enable, bool info_enable, bool warning_enable, bool error_enable,
                bool fatal_enable, const std::string& logs_path, int64_t max_log_file_size, int64_t delete_exceeds) {
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.setGlobally(el::ConfigurationType::Format, "[%datetime][%level]%msg");
    defaultConf.setGlobally(el::ConfigurationType::ToFile, "true");
    defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
    defaultConf.setGlobally(el::ConfigurationType::SubsecondPrecision, "3");
    defaultConf.setGlobally(el::ConfigurationType::PerformanceTracking, "false");

    std::string logs_reg_path = logs_path.rfind('/') == logs_path.length() - 1 ? logs_path : logs_path + "/";
    std::string global_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-global.log";
    defaultConf.set(el::Level::Global, el::ConfigurationType::Filename, global_log_path.c_str());
    defaultConf.set(el::Level::Global, el::ConfigurationType::Enabled, "true");

    std::string info_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-info.log";
    defaultConf.set(el::Level::Info, el::ConfigurationType::Filename, info_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.info_enable_to_false", info_enable = false);
    if (info_enable) {
        defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }

    std::string debug_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-debug.log";
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Filename, debug_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.debug_enable_to_false", debug_enable = false);
    if (debug_enable) {
        defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    }

    std::string warning_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-warning.log";
    defaultConf.set(el::Level::Warning, el::ConfigurationType::Filename, warning_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.warning_enable_to_false", warning_enable = false);
    if (warning_enable) {
        defaultConf.set(el::Level::Warning, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
    }

    std::string trace_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-trace.log";
    defaultConf.set(el::Level::Trace, el::ConfigurationType::Filename, trace_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.trace_enable_to_false", trace_enable = false);
    if (trace_enable) {
        defaultConf.set(el::Level::Trace, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    std::string error_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-error.log";
    defaultConf.set(el::Level::Error, el::ConfigurationType::Filename, error_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.error_enable_to_false", error_enable = false);
    if (error_enable) {
        defaultConf.set(el::Level::Error, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Error, el::ConfigurationType::Enabled, "false");
    }

    std::string fatal_log_path = logs_reg_path + "milvus-%datetime{%y-%M-%d-%H:%m}-fatal.log";
    defaultConf.set(el::Level::Fatal, el::ConfigurationType::Filename, fatal_log_path.c_str());
    fiu_do_on("LogUtil.InitLog.fatal_enable_to_false", fatal_enable = false);
    if (fatal_enable) {
        defaultConf.set(el::Level::Fatal, el::ConfigurationType::Enabled, "true");
    } else {
        defaultConf.set(el::Level::Fatal, el::ConfigurationType::Enabled, "false");
    }

    fiu_do_on("LogUtil.InitLog.set_max_log_size_small_than_min",
              max_log_file_size = CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN - 1);
    if (max_log_file_size < CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN ||
        max_log_file_size > CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX) {
        return Status(SERVER_UNEXPECTED_ERROR, "max_log_file_size must in range[" +
                                                   std::to_string(CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN) + ", " +
                                                   std::to_string(CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX) + "], now is " +
                                                   std::to_string(max_log_file_size));
    }
    defaultConf.setGlobally(el::ConfigurationType::MaxLogFileSize, std::to_string(max_log_file_size));
    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
    el::Helpers::installPreRollOutCallback(RolloutHandler);
    el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);

    // set delete_exceeds = 0 means disable throw away log file even they reach certain limit.
    if (delete_exceeds != 0) {
        fiu_do_on("LogUtil.InitLog.delete_exceeds_small_than_min", delete_exceeds = CONFIG_LOGS_LOG_ROTATE_NUM_MIN - 1);
        if (delete_exceeds < CONFIG_LOGS_LOG_ROTATE_NUM_MIN || delete_exceeds > CONFIG_LOGS_LOG_ROTATE_NUM_MAX) {
            return Status(SERVER_UNEXPECTED_ERROR, "delete_exceeds must in range[" +
                                                       std::to_string(CONFIG_LOGS_LOG_ROTATE_NUM_MIN) + ", " +
                                                       std::to_string(CONFIG_LOGS_LOG_ROTATE_NUM_MAX) + "], now is " +
                                                       std::to_string(delete_exceeds));
        }
        enable_log_delete = true;
        logs_delete_exceeds = delete_exceeds;
    }

    el::Loggers::reconfigureLogger("default", defaultConf);

    return Status::OK();
}

}  // namespace milvus
