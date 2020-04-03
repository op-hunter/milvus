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

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/ConfigNode.h"
#include "utils/Status.h"

namespace milvus {
namespace server {

using ConfigCallBackF = std::function<Status(const std::string&)>;

#define CONFIG_CHECK(func) \
    do {                   \
        Status s = func;   \
        if (!s.ok()) {     \
            return s;      \
        }                  \
    } while (false)

static const char* CONFIG_NODE_DELIMITER = ".";
static const char* CONFIG_VERSION = "version";

/* server config */
static const char* CONFIG_SERVER = "server_config";
static const char* CONFIG_SERVER_ADDRESS = "address";
static const char* CONFIG_SERVER_ADDRESS_DEFAULT = "127.0.0.1";
static const char* CONFIG_SERVER_PORT = "port";
static const char* CONFIG_SERVER_PORT_DEFAULT = "19530";
static const char* CONFIG_SERVER_DEPLOY_MODE = "deploy_mode";
static const char* CONFIG_SERVER_DEPLOY_MODE_DEFAULT = "single";
static const char* CONFIG_SERVER_TIME_ZONE = "time_zone";
static const char* CONFIG_SERVER_TIME_ZONE_DEFAULT = "UTC+8";
static const char* CONFIG_SERVER_WEB_PORT = "web_port";
static const char* CONFIG_SERVER_WEB_PORT_DEFAULT = "19121";

/* db config */
static const char* CONFIG_DB = "db_config";
static const char* CONFIG_DB_BACKEND_URL = "backend_url";
static const char* CONFIG_DB_BACKEND_URL_DEFAULT = "sqlite://:@:/";
static const char* CONFIG_DB_ARCHIVE_DISK_THRESHOLD = "archive_disk_threshold";
static const char* CONFIG_DB_ARCHIVE_DISK_THRESHOLD_DEFAULT = "0";
static const char* CONFIG_DB_ARCHIVE_DAYS_THRESHOLD = "archive_days_threshold";
static const char* CONFIG_DB_ARCHIVE_DAYS_THRESHOLD_DEFAULT = "0";
static const char* CONFIG_DB_PRELOAD_TABLE = "preload_table";
static const char* CONFIG_DB_PRELOAD_TABLE_DEFAULT = "";
static const char* CONFIG_DB_AUTO_FLUSH_INTERVAL = "auto_flush_interval";
static const char* CONFIG_DB_AUTO_FLUSH_INTERVAL_DEFAULT = "1";

/* storage config */
static const char* CONFIG_STORAGE = "storage_config";
static const char* CONFIG_STORAGE_PRIMARY_PATH = "primary_path";
static const char* CONFIG_STORAGE_PRIMARY_PATH_DEFAULT = "/tmp/milvus";
static const char* CONFIG_STORAGE_SECONDARY_PATH = "secondary_path";
static const char* CONFIG_STORAGE_SECONDARY_PATH_DEFAULT = "";
static const char* CONFIG_STORAGE_S3_ENABLE = "s3_enable";
static const char* CONFIG_STORAGE_S3_ENABLE_DEFAULT = "false";
static const char* CONFIG_STORAGE_S3_ADDRESS = "s3_address";
static const char* CONFIG_STORAGE_S3_ADDRESS_DEFAULT = "127.0.0.1";
static const char* CONFIG_STORAGE_S3_PORT = "s3_port";
static const char* CONFIG_STORAGE_S3_PORT_DEFAULT = "9000";
static const char* CONFIG_STORAGE_S3_ACCESS_KEY = "s3_access_key";
static const char* CONFIG_STORAGE_S3_ACCESS_KEY_DEFAULT = "minioadmin";
static const char* CONFIG_STORAGE_S3_SECRET_KEY = "s3_secret_key";
static const char* CONFIG_STORAGE_S3_SECRET_KEY_DEFAULT = "minioadmin";
static const char* CONFIG_STORAGE_S3_BUCKET = "s3_bucket";
static const char* CONFIG_STORAGE_S3_BUCKET_DEFAULT = "milvus-bucket";

/* cache config */
static const char* CONFIG_CACHE = "cache_config";
static const char* CONFIG_CACHE_CPU_CACHE_CAPACITY = "cpu_cache_capacity";
static const char* CONFIG_CACHE_CPU_CACHE_CAPACITY_DEFAULT = "4";
static const char* CONFIG_CACHE_CPU_CACHE_THRESHOLD = "cpu_cache_threshold";
static const char* CONFIG_CACHE_CPU_CACHE_THRESHOLD_DEFAULT = "0.7";
static const char* CONFIG_CACHE_INSERT_BUFFER_SIZE = "insert_buffer_size";
static const char* CONFIG_CACHE_INSERT_BUFFER_SIZE_DEFAULT = "1";
static const char* CONFIG_CACHE_CACHE_INSERT_DATA = "cache_insert_data";
static const char* CONFIG_CACHE_CACHE_INSERT_DATA_DEFAULT = "false";

/* metric config */
static const char* CONFIG_METRIC = "metric_config";
static const char* CONFIG_METRIC_ENABLE_MONITOR = "enable_monitor";
static const char* CONFIG_METRIC_ENABLE_MONITOR_DEFAULT = "false";
static const char* CONFIG_METRIC_ADDRESS = "address";
static const char* CONFIG_METRIC_ADDRESS_DEFAULT = "127.0.0.1";
static const char* CONFIG_METRIC_PORT = "port";
static const char* CONFIG_METRIC_PORT_DEFAULT = "9091";

/* engine config */
static const char* CONFIG_ENGINE = "engine_config";
static const char* CONFIG_ENGINE_USE_BLAS_THRESHOLD = "use_blas_threshold";
static const char* CONFIG_ENGINE_USE_BLAS_THRESHOLD_DEFAULT = "1100";
static const char* CONFIG_ENGINE_OMP_THREAD_NUM = "omp_thread_num";
static const char* CONFIG_ENGINE_OMP_THREAD_NUM_DEFAULT = "0";
static const char* CONFIG_ENGINE_USE_AVX512 = "use_avx512";
static const char* CONFIG_ENGINE_USE_AVX512_DEFAULT = "true";
static const char* CONFIG_ENGINE_GPU_SEARCH_THRESHOLD = "gpu_search_threshold";
static const char* CONFIG_ENGINE_GPU_SEARCH_THRESHOLD_DEFAULT = "1000";

/* gpu resource config */
static const char* CONFIG_GPU_RESOURCE = "gpu_resource_config";
static const char* CONFIG_GPU_RESOURCE_ENABLE = "enable";
#ifdef MILVUS_GPU_VERSION
static const char* CONFIG_GPU_RESOURCE_ENABLE_DEFAULT = "true";
#else
static const char* CONFIG_GPU_RESOURCE_ENABLE_DEFAULT = "false";
#endif
static const char* CONFIG_GPU_RESOURCE_CACHE_CAPACITY = "cache_capacity";
static const char* CONFIG_GPU_RESOURCE_CACHE_CAPACITY_DEFAULT = "1";
static const char* CONFIG_GPU_RESOURCE_CACHE_THRESHOLD = "cache_threshold";
static const char* CONFIG_GPU_RESOURCE_CACHE_THRESHOLD_DEFAULT = "0.7";
static const char* CONFIG_GPU_RESOURCE_DELIMITER = ",";
static const char* CONFIG_GPU_RESOURCE_SEARCH_RESOURCES = "search_resources";
static const char* CONFIG_GPU_RESOURCE_SEARCH_RESOURCES_DEFAULT = "gpu0";
static const char* CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES = "build_index_resources";
static const char* CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES_DEFAULT = "gpu0";

/* tracing config */
static const char* CONFIG_TRACING = "tracing_config";
static const char* CONFIG_TRACING_JSON_CONFIG_PATH = "json_config_path";

/* wal config */
static const char* CONFIG_WAL = "wal_config";
static const char* CONFIG_WAL_ENABLE = "enable";
static const char* CONFIG_WAL_ENABLE_DEFAULT = "true";
static const char* CONFIG_WAL_RECOVERY_ERROR_IGNORE = "recovery_error_ignore";
static const char* CONFIG_WAL_RECOVERY_ERROR_IGNORE_DEFAULT = "true";
static const char* CONFIG_WAL_BUFFER_SIZE = "buffer_size";
static const char* CONFIG_WAL_BUFFER_SIZE_DEFAULT = "256";
static const int64_t CONFIG_WAL_BUFFER_SIZE_MAX = 4096;
static const int64_t CONFIG_WAL_BUFFER_SIZE_MIN = 64;
static const char* CONFIG_WAL_WAL_PATH = "wal_path";
static const char* CONFIG_WAL_WAL_PATH_DEFAULT = "/tmp/milvus/wal";

class Config {
 private:
    Config();

 public:
    static Config&
    GetInstance();
    Status
    LoadConfigFile(const std::string& filename);
    Status
    ValidateConfig();
    Status
    ResetDefaultConfig();
    void
    GetConfigJsonStr(std::string& result);
    Status
    ProcessConfigCli(std::string& result, const std::string& cmd);

    Status
    GenUniqueIdentityID(const std::string& identity, std::string& uid);

    Status
    RegisterCallBack(const std::string& node, const std::string& sub_node, const std::string& key,
                     ConfigCallBackF& callback);

    Status
    CancelCallBack(const std::string& node, const std::string& sub_node, const std::string& key);

 private:
    ConfigNode&
    GetConfigRoot();
    ConfigNode&
    GetConfigNode(const std::string& name);
    bool
    ConfigNodeValid(const std::string& parent_key, const std::string& child_key);
    Status
    GetConfigValueInMem(const std::string& parent_key, const std::string& child_key, std::string& value);
    Status
    SetConfigValueInMem(const std::string& parent_key, const std::string& child_key, const std::string& value);
    Status
    GetConfigCli(std::string& value, const std::string& parent_key, const std::string& child_key);
    Status
    SetConfigCli(const std::string& parent_key, const std::string& child_key, const std::string& value);

    Status
    UpdateFileConfigFromMem(const std::string& parent_key, const std::string& child_key);

    ///////////////////////////////////////////////////////////////////////////
    Status
    CheckConfigVersion(const std::string& value);

    /* server config */
    Status
    CheckServerConfigAddress(const std::string& value);
    Status
    CheckServerConfigPort(const std::string& value);
    Status
    CheckServerConfigDeployMode(const std::string& value);
    Status
    CheckServerConfigTimeZone(const std::string& value);
    Status
    CheckServerConfigWebPort(const std::string& value);

    /* db config */
    Status
    CheckDBConfigBackendUrl(const std::string& value);
    Status
    CheckDBConfigPreloadCollection(const std::string& value);
    Status
    CheckDBConfigArchiveDiskThreshold(const std::string& value);
    Status
    CheckDBConfigArchiveDaysThreshold(const std::string& value);
    Status
    CheckDBConfigAutoFlushInterval(const std::string& value);

    /* storage config */
    Status
    CheckStorageConfigPrimaryPath(const std::string& value);
    Status
    CheckStorageConfigSecondaryPath(const std::string& value);
    Status
    CheckStorageConfigS3Enable(const std::string& value);
    Status
    CheckStorageConfigS3Address(const std::string& value);
    Status
    CheckStorageConfigS3Port(const std::string& value);
    Status
    CheckStorageConfigS3AccessKey(const std::string& value);
    Status
    CheckStorageConfigS3SecretKey(const std::string& value);
    Status
    CheckStorageConfigS3Bucket(const std::string& value);

    /* metric config */
    Status
    CheckMetricConfigEnableMonitor(const std::string& value);
    Status
    CheckMetricConfigAddress(const std::string& value);
    Status
    CheckMetricConfigPort(const std::string& value);

    /* cache config */
    Status
    CheckCacheConfigCpuCacheCapacity(const std::string& value);
    Status
    CheckCacheConfigCpuCacheThreshold(const std::string& value);
    Status
    CheckCacheConfigInsertBufferSize(const std::string& value);
    Status
    CheckCacheConfigCacheInsertData(const std::string& value);

    /* engine config */
    Status
    CheckEngineConfigUseBlasThreshold(const std::string& value);
    Status
    CheckEngineConfigOmpThreadNum(const std::string& value);
    Status
    CheckEngineConfigUseAVX512(const std::string& value);

#ifdef MILVUS_GPU_VERSION
    Status
    CheckEngineConfigGpuSearchThreshold(const std::string& value);

    /* gpu resource config */
    Status
    CheckGpuResourceConfigEnable(const std::string& value);
    Status
    CheckGpuResourceConfigCacheCapacity(const std::string& value);
    Status
    CheckGpuResourceConfigCacheThreshold(const std::string& value);
    Status
    CheckGpuResourceConfigSearchResources(const std::vector<std::string>& value);
    Status
    CheckGpuResourceConfigBuildIndexResources(const std::vector<std::string>& value);
#endif

    /* tracing config */
    Status
    CheckTracingConfigJsonConfigPath(const std::string& value);

    /* wal config */
    Status
    CheckWalConfigEnable(const std::string& value);
    Status
    CheckWalConfigRecoveryErrorIgnore(const std::string& value);
    Status
    CheckWalConfigBufferSize(const std::string& value);
    Status
    CheckWalConfigWalPath(const std::string& value);

    std::string
    GetConfigStr(const std::string& parent_key, const std::string& child_key, const std::string& default_value = "");
    std::string
    GetConfigSequenceStr(const std::string& parent_key, const std::string& child_key, const std::string& delim = ",",
                         const std::string& default_value = "");
    Status
    GetConfigVersion(std::string& value);

    Status
    ExecCallBacks(const std::string& node, const std::string& sub_node, const std::string& value);

 public:
    /* server config */
    Status
    GetServerConfigAddress(std::string& value);
    Status
    GetServerConfigPort(std::string& value);
    Status
    GetServerConfigDeployMode(std::string& value);
    Status
    GetServerConfigTimeZone(std::string& value);
    Status
    GetServerConfigWebPort(std::string& value);

    /* db config */
    Status
    GetDBConfigBackendUrl(std::string& value);
    Status
    GetDBConfigArchiveDiskThreshold(int64_t& value);
    Status
    GetDBConfigArchiveDaysThreshold(int64_t& value);
    Status
    GetDBConfigPreloadCollection(std::string& value);
    Status
    GetDBConfigAutoFlushInterval(int64_t& value);

    /* storage config */
    Status
    GetStorageConfigPrimaryPath(std::string& value);
    Status
    GetStorageConfigSecondaryPath(std::string& value);
    Status
    GetStorageConfigS3Enable(bool& value);
    Status
    GetStorageConfigS3Address(std::string& value);
    Status
    GetStorageConfigS3Port(std::string& value);
    Status
    GetStorageConfigS3AccessKey(std::string& value);
    Status
    GetStorageConfigS3SecretKey(std::string& value);
    Status
    GetStorageConfigS3Bucket(std::string& value);

    /* metric config */
    Status
    GetMetricConfigEnableMonitor(bool& value);
    Status
    GetMetricConfigAddress(std::string& value);
    Status
    GetMetricConfigPort(std::string& value);

    /* cache config */
    Status
    GetCacheConfigCpuCacheCapacity(int64_t& value);
    Status
    GetCacheConfigCpuCacheThreshold(float& value);
    Status
    GetCacheConfigInsertBufferSize(int64_t& value);
    Status
    GetCacheConfigCacheInsertData(bool& value);

    /* engine config */
    Status
    GetEngineConfigUseBlasThreshold(int64_t& value);
    Status
    GetEngineConfigOmpThreadNum(int64_t& value);
    Status
    GetEngineConfigUseAVX512(bool& value);

#ifdef MILVUS_GPU_VERSION
    Status
    GetEngineConfigGpuSearchThreshold(int64_t& value);

    /* gpu resource config */
    Status
    GetGpuResourceConfigEnable(bool& value);
    Status
    GetGpuResourceConfigCacheCapacity(int64_t& value);
    Status
    GetGpuResourceConfigCacheThreshold(float& value);
    Status
    GetGpuResourceConfigSearchResources(std::vector<int64_t>& value);
    Status
    GetGpuResourceConfigBuildIndexResources(std::vector<int64_t>& value);
#endif

    /* tracing config */
    Status
    GetTracingConfigJsonConfigPath(std::string& value);

    /* wal config */
    Status
    GetWalConfigEnable(bool& value);
    Status
    GetWalConfigRecoveryErrorIgnore(bool& value);
    Status
    GetWalConfigBufferSize(int64_t& value);
    Status
    GetWalConfigWalPath(std::string& value);

    Status
    GetServerRestartRequired(bool& required);

 public:
    /* server config */
    Status
    SetServerConfigAddress(const std::string& value);
    Status
    SetServerConfigPort(const std::string& value);
    Status
    SetServerConfigDeployMode(const std::string& value);
    Status
    SetServerConfigTimeZone(const std::string& value);
    Status
    SetServerConfigWebPort(const std::string& value);

    /* db config */
    Status
    SetDBConfigBackendUrl(const std::string& value);
    Status
    SetDBConfigPreloadCollection(const std::string& value);
    Status
    SetDBConfigArchiveDiskThreshold(const std::string& value);
    Status
    SetDBConfigArchiveDaysThreshold(const std::string& value);
    Status
    SetDBConfigAutoFlushInterval(const std::string& value);

    /* storage config */
    Status
    SetStorageConfigPrimaryPath(const std::string& value);
    Status
    SetStorageConfigSecondaryPath(const std::string& value);
    Status
    SetStorageConfigS3Enable(const std::string& value);
    Status
    SetStorageConfigS3Address(const std::string& value);
    Status
    SetStorageConfigS3Port(const std::string& value);
    Status
    SetStorageConfigS3AccessKey(const std::string& value);
    Status
    SetStorageConfigS3SecretKey(const std::string& value);
    Status
    SetStorageConfigS3Bucket(const std::string& value);

    /* metric config */
    Status
    SetMetricConfigEnableMonitor(const std::string& value);
    Status
    SetMetricConfigAddress(const std::string& value);
    Status
    SetMetricConfigPort(const std::string& value);

    /* cache config */
    Status
    SetCacheConfigCpuCacheCapacity(const std::string& value);
    Status
    SetCacheConfigCpuCacheThreshold(const std::string& value);
    Status
    SetCacheConfigInsertBufferSize(const std::string& value);
    Status
    SetCacheConfigCacheInsertData(const std::string& value);

    /* engine config */
    Status
    SetEngineConfigUseBlasThreshold(const std::string& value);
    Status
    SetEngineConfigOmpThreadNum(const std::string& value);
    Status
    SetEngineConfigUseAVX512(const std::string& value);

    /* tracing config */
    Status
    SetTracingConfigJsonConfigPath(const std::string& value);

    /* wal config */
    Status
    SetWalConfigEnable(const std::string& value);
    Status
    SetWalConfigRecoveryErrorIgnore(const std::string& value);
    Status
    SetWalConfigBufferSize(const std::string& value);
    Status
    SetWalConfigWalPath(const std::string& value);

#ifdef MILVUS_GPU_VERSION
    Status
    SetEngineConfigGpuSearchThreshold(const std::string& value);

    /* gpu resource config */
    Status
    SetGpuResourceConfigEnable(const std::string& value);
    Status
    SetGpuResourceConfigCacheCapacity(const std::string& value);
    Status
    SetGpuResourceConfigCacheThreshold(const std::string& value);
    Status
    SetGpuResourceConfigSearchResources(const std::string& value);
    Status
    SetGpuResourceConfigBuildIndexResources(const std::string& value);
#endif

 private:
    bool restart_required_ = false;
    std::string config_file_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_map_;
    std::unordered_map<std::string, std::unordered_map<std::string, ConfigCallBackF>> config_callback_;
    std::mutex mutex_;
};

}  // namespace server
}  // namespace milvus
