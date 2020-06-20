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

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fiu-local.h>

#include "config/Config.h"
#include "config/Utils.h"
#include "config/YamlConfigMgr.h"
#include "server/DBWrapper.h"
#include "thirdparty/nlohmann/json.hpp"
#include "utils/Log.h"
#include "utils/StringHelpFunctions.h"

namespace milvus {
namespace server {

const char* CONFIG_NODE_DELIMITER = ".";
const char* CONFIG_VERSION = "version";

/* cluster config */
const char* CONFIG_CLUSTER = "cluster";
const char* CONFIG_CLUSTER_ENABLE = "enable";
const char* CONFIG_CLUSTER_ENABLE_DEFAULT = "true";
const char* CONFIG_CLUSTER_ROLE = "role";
const char* CONFIG_CLUSTER_ROLE_DEFAULT = "rw";

/* general config */
const char* CONFIG_GENERAL = "general";
const char* CONFIG_GENERAL_TIMEZONE = "timezone";
const char* CONFIG_GENERAL_TIMEZONE_DEFAULT = "UTC+8";
const char* CONFIG_GENERAL_METAURI = "meta_uri";
const char* CONFIG_GENERAL_METAURI_DEFAULT = "sqlite://:@:/";

/* network config */
const char* CONFIG_NETWORK = "network";
const char* CONFIG_NETWORK_BIND_ADDRESS = "bind.address";
const char* CONFIG_NETWORK_BIND_ADDRESS_DEFAULT = "127.0.0.1";
const char* CONFIG_NETWORK_BIND_PORT = "bind.port";
const char* CONFIG_NETWORK_BIND_PORT_DEFAULT = "19530";
const char* CONFIG_NETWORK_HTTP_ENABLE = "http.enable";
const char* CONFIG_NETWORK_HTTP_ENABLE_DEFAULT = "true";
const char* CONFIG_NETWORK_HTTP_PORT = "http.port";
const char* CONFIG_NETWORK_HTTP_PORT_DEFAULT = "19121";

/* db config */
const char* CONFIG_DB = "db_config";
const char* CONFIG_DB_ARCHIVE_DISK_THRESHOLD = "archive_disk_threshold";
const char* CONFIG_DB_ARCHIVE_DISK_THRESHOLD_DEFAULT = "0";
const char* CONFIG_DB_ARCHIVE_DAYS_THRESHOLD = "archive_days_threshold";
const char* CONFIG_DB_ARCHIVE_DAYS_THRESHOLD_DEFAULT = "0";

/* storage config */
const char* CONFIG_STORAGE = "storage";
const char* CONFIG_STORAGE_PATH = "path";
const char* CONFIG_STORAGE_PATH_DEFAULT = "/tmp/milvus";
const char* CONFIG_STORAGE_AUTO_FLUSH_INTERVAL = "auto_flush_interval";
const char* CONFIG_STORAGE_AUTO_FLUSH_INTERVAL_DEFAULT = "1";
const char* CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT = "file_cleanup_timeout";
const char* CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_DEFAULT = "10";
const int64_t CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MIN = 0;
const int64_t CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MAX = 3600;

/* cache config */
const char* CONFIG_CACHE = "cache";
const char* CONFIG_CACHE_CPU_CACHE_CAPACITY = "cache_size";
const char* CONFIG_CACHE_CPU_CACHE_CAPACITY_DEFAULT = "4294967296"; /* 4 GB */
const char* CONFIG_CACHE_CPU_CACHE_THRESHOLD = "cpu_cache_threshold";
const char* CONFIG_CACHE_CPU_CACHE_THRESHOLD_DEFAULT = "0.7";
const char* CONFIG_CACHE_INSERT_BUFFER_SIZE = "insert_buffer_size";
const char* CONFIG_CACHE_INSERT_BUFFER_SIZE_DEFAULT = "1073741824"; /* 1 GB */
const char* CONFIG_CACHE_CACHE_INSERT_DATA = "cache_insert_data";
const char* CONFIG_CACHE_CACHE_INSERT_DATA_DEFAULT = "false";
const char* CONFIG_CACHE_PRELOAD_COLLECTION = "preload_collection";
const char* CONFIG_CACHE_PRELOAD_COLLECTION_DEFAULT = "";

/* metric config */
const char* CONFIG_METRIC = "metric";
const char* CONFIG_METRIC_ENABLE_MONITOR = "enable";
const char* CONFIG_METRIC_ENABLE_MONITOR_DEFAULT = "false";
const char* CONFIG_METRIC_ADDRESS = "address";
const char* CONFIG_METRIC_ADDRESS_DEFAULT = "127.0.0.1";
const char* CONFIG_METRIC_PORT = "port";
const char* CONFIG_METRIC_PORT_DEFAULT = "9091";

/* engine config */
const char* CONFIG_ENGINE = "engine_config";
const char* CONFIG_ENGINE_USE_BLAS_THRESHOLD = "use_blas_threshold";
const char* CONFIG_ENGINE_USE_BLAS_THRESHOLD_DEFAULT = "1100";
const char* CONFIG_ENGINE_OMP_THREAD_NUM = "omp_thread_num";
const char* CONFIG_ENGINE_OMP_THREAD_NUM_DEFAULT = "0";
const char* CONFIG_ENGINE_SIMD_TYPE = "simd_type";
const char* CONFIG_ENGINE_SIMD_TYPE_DEFAULT = "auto";

/* gpu resource config */
const char* CONFIG_GPU_RESOURCE = "gpu";
const char* CONFIG_GPU_RESOURCE_ENABLE = "enable";
#ifdef MILVUS_GPU_VERSION
const char* CONFIG_GPU_RESOURCE_ENABLE_DEFAULT = "true";
#else
const char* CONFIG_GPU_RESOURCE_ENABLE_DEFAULT = "false";
#endif
const char* CONFIG_GPU_RESOURCE_CACHE_CAPACITY = "cache_size";
const char* CONFIG_GPU_RESOURCE_CACHE_CAPACITY_DEFAULT = "1073741824"; /* 1 GB */
const char* CONFIG_GPU_RESOURCE_CACHE_THRESHOLD = "cache_threshold";
const char* CONFIG_GPU_RESOURCE_CACHE_THRESHOLD_DEFAULT = "0.7";
const char* CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD = "gpu_search_threshold";
const char* CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD_DEFAULT = "1000";
const char* CONFIG_GPU_RESOURCE_DELIMITER = ",";
const char* CONFIG_GPU_RESOURCE_SEARCH_RESOURCES = "search_devices";
const char* CONFIG_GPU_RESOURCE_SEARCH_RESOURCES_DEFAULT = "gpu0";
const char* CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES = "build_index_devices";
const char* CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES_DEFAULT = "gpu0";

/* tracing config */
const char* CONFIG_TRACING = "tracing_config";
const char* CONFIG_TRACING_JSON_CONFIG_PATH = "json_config_path";

/* wal config */
const char* CONFIG_WAL = "wal";
const char* CONFIG_WAL_ENABLE = "enable";
const char* CONFIG_WAL_ENABLE_DEFAULT = "true";
const char* CONFIG_WAL_RECOVERY_ERROR_IGNORE = "recovery_error_ignore";
const char* CONFIG_WAL_RECOVERY_ERROR_IGNORE_DEFAULT = "true";
const char* CONFIG_WAL_BUFFER_SIZE = "buffer_size";
const char* CONFIG_WAL_BUFFER_SIZE_DEFAULT = "268435456"; /* 256 MB */
const int64_t CONFIG_WAL_BUFFER_SIZE_MIN = 67108864;      /* 64 MB */
const int64_t CONFIG_WAL_BUFFER_SIZE_MAX = 4294967296;    /* 4 GB */
const char* CONFIG_WAL_WAL_PATH = "path";
const char* CONFIG_WAL_WAL_PATH_DEFAULT = "/tmp/milvus/wal";

/* logs config */
const char* CONFIG_LOGS = "logs";
const char* CONFIG_LOGS_LEVEL = "level";
const char* CONFIG_LOGS_LEVEL_DEFAULT = "debug";
const char* CONFIG_LOGS_TRACE_ENABLE = "trace.enable";
const char* CONFIG_LOGS_TRACE_ENABLE_DEFAULT = "true";
const char* CONFIG_LOGS_PATH = "path";
const char* CONFIG_LOGS_PATH_DEFAULT = "/tmp/milvus/logs";
const char* CONFIG_LOGS_MAX_LOG_FILE_SIZE = "max_log_file_size";
const char* CONFIG_LOGS_MAX_LOG_FILE_SIZE_DEFAULT = "1073741824"; /* 1 GB */
const int64_t CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN = 536870912;      /* 512 MB */
const int64_t CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX = 4294967296;     /* 4 GB */
const char* CONFIG_LOGS_LOG_ROTATE_NUM = "log_rotate_num";
const char* CONFIG_LOGS_LOG_ROTATE_NUM_DEFAULT = "0";
const int64_t CONFIG_LOGS_LOG_ROTATE_NUM_MIN = 0;
const int64_t CONFIG_LOGS_LOG_ROTATE_NUM_MAX = 1024;

constexpr int64_t GB = 1UL << 30;
constexpr int32_t PORT_NUMBER_MIN = 1024;
constexpr int32_t PORT_NUMBER_MAX = 65535;

static const std::unordered_map<std::string, std::string> milvus_config_version_map({{"0.6.0", "0.1"},
                                                                                     {"0.7.0", "0.2"},
                                                                                     {"0.7.1", "0.2"},
                                                                                     {"0.8.0", "0.3"},
                                                                                     {"0.9.0", "0.4"},
                                                                                     {"0.9.1", "0.4"},
                                                                                     {"0.10.0", "0.5"}});

/////////////////////////////////////////////////////////////
Config::Config() {
    auto empty_map = std::unordered_map<std::string, ConfigCallBackF>();

    // cache config
    std::string node_cpu_cache_capacity = std::string(CONFIG_CACHE) + "." + CONFIG_CACHE_CPU_CACHE_CAPACITY;
    config_callback_[node_cpu_cache_capacity] = empty_map;

    std::string node_insert_buffer_size = std::string(CONFIG_CACHE) + "." + CONFIG_CACHE_INSERT_BUFFER_SIZE;
    config_callback_[node_insert_buffer_size] = empty_map;

    std::string node_cache_insert_data = std::string(CONFIG_CACHE) + "." + CONFIG_CACHE_CACHE_INSERT_DATA;
    config_callback_[node_cache_insert_data] = empty_map;

    // engine config
    std::string node_blas_threshold = std::string(CONFIG_ENGINE) + "." + CONFIG_ENGINE_USE_BLAS_THRESHOLD;
    config_callback_[node_blas_threshold] = empty_map;

    // gpu resources config
    std::string node_gpu_enable = std::string(CONFIG_GPU_RESOURCE) + "." + CONFIG_GPU_RESOURCE_ENABLE;
    config_callback_[node_gpu_enable] = empty_map;

    std::string node_gpu_cache_capacity = std::string(CONFIG_GPU_RESOURCE) + "." + CONFIG_GPU_RESOURCE_CACHE_CAPACITY;
    config_callback_[node_gpu_cache_capacity] = empty_map;

    std::string node_gpu_search_threshold =
        std::string(CONFIG_GPU_RESOURCE) + "." + CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD;
    config_callback_[node_gpu_search_threshold] = empty_map;

    std::string node_gpu_search_res = std::string(CONFIG_GPU_RESOURCE) + "." + CONFIG_GPU_RESOURCE_SEARCH_RESOURCES;
    config_callback_[node_gpu_search_res] = empty_map;

    std::string node_gpu_build_res = std::string(CONFIG_GPU_RESOURCE) + "." + CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES;
    config_callback_[node_gpu_build_res] = empty_map;
}

Config&
Config::GetInstance() {
    static Config config_inst;
    return config_inst;
}

Status
Config::LoadConfigFile(const std::string& filename) {
    if (filename.empty()) {
        return Status(SERVER_UNEXPECTED_ERROR, "No specified config file");
    }

    struct stat file_stat;
    if (stat(filename.c_str(), &file_stat) != 0) {
        std::string str = "Config file not exist: " + filename;
        return Status(SERVER_FILE_NOT_FOUND, str);
    }

    ConfigMgr* mgr = YamlConfigMgr::GetInstance();
    STATUS_CHECK(mgr->LoadConfigFile(filename));

    // store config file path
    config_file_ = filename;

    return Status::OK();
}

Status
Config::ValidateConfig() {
    std::string config_version;
    STATUS_CHECK(GetConfigVersion(config_version));

    /* cluster config */
    bool cluster_enable;
    STATUS_CHECK(GetClusterConfigEnable(cluster_enable));

    std::string cluster_role;
    STATUS_CHECK(GetClusterConfigRole(cluster_role));

    /* general config */
    std::string general_timezone;
    STATUS_CHECK(GetGeneralConfigTimezone(general_timezone));

    std::string general_metauri;
    STATUS_CHECK(GetGeneralConfigMetaURI(general_metauri));

    /* network config */
    std::string bind_address;
    STATUS_CHECK(GetNetworkConfigBindAddress(bind_address));

    std::string bind_port;
    STATUS_CHECK(GetNetworkConfigBindPort(bind_port));

    bool http_enable = false;
    STATUS_CHECK(GetNetworkConfigHTTPEnable(http_enable));

    std::string http_port;
    STATUS_CHECK(GetNetworkConfigHTTPPort(http_port));

    /* db config */
    int64_t db_archive_disk_threshold;
    STATUS_CHECK(GetDBConfigArchiveDiskThreshold(db_archive_disk_threshold));

    int64_t db_archive_days_threshold;
    STATUS_CHECK(GetDBConfigArchiveDaysThreshold(db_archive_days_threshold));

    /* storage config */
    std::string storage_path;
    STATUS_CHECK(GetStorageConfigPath(storage_path));

    int64_t auto_flush_interval;
    STATUS_CHECK(GetStorageConfigAutoFlushInterval(auto_flush_interval));

    // bool storage_s3_enable;
    // STATUS_CHECK(GetStorageConfigS3Enable(storage_s3_enable));
    // // std::cout << "S3 " << (storage_s3_enable ? "ENABLED !" : "DISABLED !") << std::endl;
    //
    // std::string storage_s3_address;
    // STATUS_CHECK(GetStorageConfigS3Address(storage_s3_address));
    //
    // std::string storage_s3_port;
    // STATUS_CHECK(GetStorageConfigS3Port(storage_s3_port));
    //
    // std::string storage_s3_access_key;
    // STATUS_CHECK(GetStorageConfigS3AccessKey(storage_s3_access_key));
    //
    // std::string storage_s3_secret_key;
    // STATUS_CHECK(GetStorageConfigS3SecretKey(storage_s3_secret_key));
    //
    // std::string storage_s3_bucket;
    // STATUS_CHECK(GetStorageConfigS3Bucket(storage_s3_bucket));

    /* metric config */
    bool metric_enable_monitor;
    STATUS_CHECK(GetMetricConfigEnableMonitor(metric_enable_monitor));

    std::string metric_address;
    STATUS_CHECK(GetMetricConfigAddress(metric_address));

    std::string metric_port;
    STATUS_CHECK(GetMetricConfigPort(metric_port));

    /* cache config */
    int64_t cache_cpu_cache_capacity;
    STATUS_CHECK(GetCacheConfigCpuCacheCapacity(cache_cpu_cache_capacity));

    float cache_cpu_cache_threshold;
    STATUS_CHECK(GetCacheConfigCpuCacheThreshold(cache_cpu_cache_threshold));

    int64_t cache_insert_buffer_size;
    STATUS_CHECK(GetCacheConfigInsertBufferSize(cache_insert_buffer_size));

    bool cache_insert_data;
    STATUS_CHECK(GetCacheConfigCacheInsertData(cache_insert_data));

    std::string cache_preload_collection;
    STATUS_CHECK(GetCacheConfigPreloadCollection(cache_preload_collection));

    /* engine config */
    int64_t engine_use_blas_threshold;
    STATUS_CHECK(GetEngineConfigUseBlasThreshold(engine_use_blas_threshold));

    int64_t engine_omp_thread_num;
    STATUS_CHECK(GetEngineConfigOmpThreadNum(engine_omp_thread_num));

    std::string engine_simd_type;
    STATUS_CHECK(GetEngineConfigSimdType(engine_simd_type));

    /* gpu resource config */
#ifdef MILVUS_GPU_VERSION
    bool gpu_resource_enable;
    STATUS_CHECK(GetGpuResourceConfigEnable(gpu_resource_enable));
    std::cout << "GPU resources " << (gpu_resource_enable ? "ENABLED !" : "DISABLED !") << std::endl;

    if (gpu_resource_enable) {
        int64_t resource_cache_capacity;
        STATUS_CHECK(GetGpuResourceConfigCacheCapacity(resource_cache_capacity));

        float resource_cache_threshold;
        STATUS_CHECK(GetGpuResourceConfigCacheThreshold(resource_cache_threshold));

        int64_t engine_gpu_search_threshold;
        STATUS_CHECK(GetGpuResourceConfigGpuSearchThreshold(engine_gpu_search_threshold));

        std::vector<int64_t> search_resources;
        STATUS_CHECK(GetGpuResourceConfigSearchResources(search_resources));

        std::vector<int64_t> index_build_resources;
        STATUS_CHECK(GetGpuResourceConfigBuildIndexResources(index_build_resources));
    }
#endif

    /* tracing config */
    std::string tracing_config_path;
    STATUS_CHECK(GetTracingConfigJsonConfigPath(tracing_config_path));

    /* wal config */
    bool enable;
    STATUS_CHECK(GetWalConfigEnable(enable));

    bool recovery_error_ignore;
    STATUS_CHECK(GetWalConfigRecoveryErrorIgnore(recovery_error_ignore));

    int64_t buffer_size;
    STATUS_CHECK(GetWalConfigBufferSize(buffer_size));

    std::string wal_path;
    STATUS_CHECK(GetWalConfigWalPath(wal_path));

    /* logs config */
    std::string logs_level;
    STATUS_CHECK(GetLogsLevel(logs_level));

    bool trace_enable;
    STATUS_CHECK(GetLogsTraceEnable(trace_enable));

    std::string logs_path;
    STATUS_CHECK(GetLogsPath(logs_path));

    int64_t logs_max_log_file_size;
    STATUS_CHECK(GetLogsMaxLogFileSize(logs_max_log_file_size));

    int64_t logs_log_rotate_num;
    STATUS_CHECK(GetLogsLogRotateNum(logs_log_rotate_num));

    return Status::OK();
}

Status
Config::ResetDefaultConfig() {
    /* cluster config */
    STATUS_CHECK(SetClusterConfigEnable(CONFIG_CLUSTER_ENABLE_DEFAULT));
    STATUS_CHECK(SetClusterConfigRole(CONFIG_CLUSTER_ROLE_DEFAULT));

    /* general config */
    STATUS_CHECK(SetGeneralConfigTimezone(CONFIG_GENERAL_TIMEZONE_DEFAULT));
    STATUS_CHECK(SetGeneralConfigMetaURI(CONFIG_GENERAL_METAURI_DEFAULT));

    /* network config */
    STATUS_CHECK(SetNetworkConfigBindAddress(CONFIG_NETWORK_BIND_ADDRESS_DEFAULT));
    STATUS_CHECK(SetNetworkConfigBindPort(CONFIG_NETWORK_BIND_PORT_DEFAULT));
    STATUS_CHECK(SetNetworkConfigHTTPEnable(CONFIG_NETWORK_HTTP_ENABLE_DEFAULT));
    STATUS_CHECK(SetNetworkConfigHTTPPort(CONFIG_NETWORK_HTTP_PORT_DEFAULT));

    /* db config */
    STATUS_CHECK(SetDBConfigArchiveDiskThreshold(CONFIG_DB_ARCHIVE_DISK_THRESHOLD_DEFAULT));
    STATUS_CHECK(SetDBConfigArchiveDaysThreshold(CONFIG_DB_ARCHIVE_DAYS_THRESHOLD_DEFAULT));

    /* storage config */
    STATUS_CHECK(SetStorageConfigPath(CONFIG_STORAGE_PATH_DEFAULT));
    STATUS_CHECK(SetStorageConfigAutoFlushInterval(CONFIG_STORAGE_AUTO_FLUSH_INTERVAL_DEFAULT));
    STATUS_CHECK(SetStorageConfigFileCleanupTimeout(CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3Enable(CONFIG_STORAGE_S3_ENABLE_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3Address(CONFIG_STORAGE_S3_ADDRESS_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3Port(CONFIG_STORAGE_S3_PORT_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3AccessKey(CONFIG_STORAGE_S3_ACCESS_KEY_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3SecretKey(CONFIG_STORAGE_S3_SECRET_KEY_DEFAULT));
    // STATUS_CHECK(SetStorageConfigS3Bucket(CONFIG_STORAGE_S3_BUCKET_DEFAULT));

    /* metric config */
    STATUS_CHECK(SetMetricConfigEnableMonitor(CONFIG_METRIC_ENABLE_MONITOR_DEFAULT));
    STATUS_CHECK(SetMetricConfigAddress(CONFIG_METRIC_ADDRESS_DEFAULT));
    STATUS_CHECK(SetMetricConfigPort(CONFIG_METRIC_PORT_DEFAULT));

    /* cache config */
    STATUS_CHECK(SetCacheConfigCpuCacheCapacity(CONFIG_CACHE_CPU_CACHE_CAPACITY_DEFAULT));
    STATUS_CHECK(SetCacheConfigCpuCacheThreshold(CONFIG_CACHE_CPU_CACHE_THRESHOLD_DEFAULT));
    STATUS_CHECK(SetCacheConfigInsertBufferSize(CONFIG_CACHE_INSERT_BUFFER_SIZE_DEFAULT));
    STATUS_CHECK(SetCacheConfigCacheInsertData(CONFIG_CACHE_CACHE_INSERT_DATA_DEFAULT));
    STATUS_CHECK(SetCacheConfigPreloadCollection(CONFIG_CACHE_PRELOAD_COLLECTION_DEFAULT));

    /* engine config */
    STATUS_CHECK(SetEngineConfigUseBlasThreshold(CONFIG_ENGINE_USE_BLAS_THRESHOLD_DEFAULT));
    STATUS_CHECK(SetEngineConfigOmpThreadNum(CONFIG_ENGINE_OMP_THREAD_NUM_DEFAULT));
    STATUS_CHECK(SetEngineConfigSimdType(CONFIG_ENGINE_SIMD_TYPE_DEFAULT));

    /* gpu resource config */
#ifdef MILVUS_GPU_VERSION
    STATUS_CHECK(SetGpuResourceConfigEnable(CONFIG_GPU_RESOURCE_ENABLE_DEFAULT));
    STATUS_CHECK(SetGpuResourceConfigCacheCapacity(CONFIG_GPU_RESOURCE_CACHE_CAPACITY_DEFAULT));
    STATUS_CHECK(SetGpuResourceConfigCacheThreshold(CONFIG_GPU_RESOURCE_CACHE_THRESHOLD_DEFAULT));
    STATUS_CHECK(SetGpuResourceConfigGpuSearchThreshold(CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD_DEFAULT));
    STATUS_CHECK(SetGpuResourceConfigSearchResources(CONFIG_GPU_RESOURCE_SEARCH_RESOURCES_DEFAULT));
    STATUS_CHECK(SetGpuResourceConfigBuildIndexResources(CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES_DEFAULT));
#endif

    /* wal config */
    STATUS_CHECK(SetWalConfigEnable(CONFIG_WAL_ENABLE_DEFAULT));
    STATUS_CHECK(SetWalConfigRecoveryErrorIgnore(CONFIG_WAL_RECOVERY_ERROR_IGNORE_DEFAULT));
    STATUS_CHECK(SetWalConfigBufferSize(CONFIG_WAL_BUFFER_SIZE_DEFAULT));
    STATUS_CHECK(SetWalConfigWalPath(CONFIG_WAL_WAL_PATH_DEFAULT));

    /* logs config */
    STATUS_CHECK(SetLogsLevel(CONFIG_LOGS_LEVEL_DEFAULT));
    STATUS_CHECK(SetLogsTraceEnable(CONFIG_LOGS_TRACE_ENABLE_DEFAULT));
    STATUS_CHECK(SetLogsPath(CONFIG_LOGS_PATH_DEFAULT));
    STATUS_CHECK(SetLogsMaxLogFileSize(CONFIG_LOGS_MAX_LOG_FILE_SIZE_DEFAULT));
    STATUS_CHECK(SetLogsLogRotateNum(CONFIG_LOGS_LOG_ROTATE_NUM_DEFAULT));

    return Status::OK();
}

void
Config::GetConfigJsonStr(std::string& result, int64_t indent) {
    nlohmann::json config_json(config_map_);
    result = config_json.dump(indent);
}

Status
Config::GetConfigCli(std::string& value, const std::string& parent_key, const std::string& child_key) {
    if (!ConfigNodeValid(parent_key, child_key)) {
        std::string str = "Config node invalid: " + parent_key + CONFIG_NODE_DELIMITER + child_key;
        return Status(SERVER_UNEXPECTED_ERROR, str);
    }
    return GetConfigValueInMem(parent_key, child_key, value);
}

Status
Config::SetConfigCli(const std::string& parent_key, const std::string& child_key, const std::string& value) {
    std::string invalid_node_str = "Config node invalid: " + parent_key + CONFIG_NODE_DELIMITER + child_key;

    if (!ConfigNodeValid(parent_key, child_key)) {
        return Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
    }
    auto status = Status::OK();
    if (parent_key == CONFIG_CLUSTER) {
        if (child_key == CONFIG_CLUSTER_ENABLE) {
            status = SetClusterConfigEnable(value);
        } else if (child_key == CONFIG_CLUSTER_ROLE) {
            status = SetClusterConfigRole(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_GENERAL) {
        if (child_key == CONFIG_GENERAL_TIMEZONE) {
            status = SetGeneralConfigTimezone(value);
        } else if (child_key == CONFIG_GENERAL_METAURI) {
            status = SetGeneralConfigMetaURI(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_NETWORK) {
        if (child_key == CONFIG_NETWORK_BIND_ADDRESS) {
            status = SetNetworkConfigBindAddress(value);
        } else if (child_key == CONFIG_NETWORK_BIND_PORT) {
            status = SetNetworkConfigBindPort(value);
        } else if (child_key == CONFIG_NETWORK_HTTP_PORT) {
            status = SetNetworkConfigHTTPPort(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_STORAGE) {
        if (child_key == CONFIG_STORAGE_PATH) {
            status = SetStorageConfigPath(value);
        } else if (child_key == CONFIG_STORAGE_AUTO_FLUSH_INTERVAL) {
            status = SetStorageConfigAutoFlushInterval(value);
            // } else if (child_key == CONFIG_STORAGE_S3_ENABLE) {
            //     status = SetStorageConfigS3Enable(value);
            // } else if (child_key == CONFIG_STORAGE_S3_ADDRESS) {
            //     status = SetStorageConfigS3Address(value);
            // } else if (child_key == CONFIG_STORAGE_S3_PORT) {
            //     status = SetStorageConfigS3Port(value);
            // } else if (child_key == CONFIG_STORAGE_S3_ACCESS_KEY) {
            //     status = SetStorageConfigS3AccessKey(value);
            // } else if (child_key == CONFIG_STORAGE_S3_SECRET_KEY) {
            //     status = SetStorageConfigS3SecretKey(value);
            // } else if (child_key == CONFIG_STORAGE_S3_BUCKET) {
            //     status = SetStorageConfigS3Bucket(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_METRIC) {
        if (child_key == CONFIG_METRIC_ENABLE_MONITOR) {
            status = SetMetricConfigEnableMonitor(value);
        } else if (child_key == CONFIG_METRIC_ADDRESS) {
            status = SetMetricConfigAddress(value);
        } else if (child_key == CONFIG_METRIC_PORT) {
            status = SetMetricConfigPort(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_CACHE) {
        if (child_key == CONFIG_CACHE_CPU_CACHE_CAPACITY) {
            status = SetCacheConfigCpuCacheCapacity(value);
        } else if (child_key == CONFIG_CACHE_CPU_CACHE_THRESHOLD) {
            status = SetCacheConfigCpuCacheThreshold(value);
        } else if (child_key == CONFIG_CACHE_CACHE_INSERT_DATA) {
            status = SetCacheConfigCacheInsertData(value);
        } else if (child_key == CONFIG_CACHE_INSERT_BUFFER_SIZE) {
            status = SetCacheConfigInsertBufferSize(value);
        } else if (child_key == CONFIG_CACHE_PRELOAD_COLLECTION) {
            status = SetCacheConfigPreloadCollection(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_ENGINE) {
        if (child_key == CONFIG_ENGINE_USE_BLAS_THRESHOLD) {
            status = SetEngineConfigUseBlasThreshold(value);
        } else if (child_key == CONFIG_ENGINE_OMP_THREAD_NUM) {
            status = SetEngineConfigOmpThreadNum(value);
        } else if (child_key == CONFIG_ENGINE_SIMD_TYPE) {
            status = SetEngineConfigSimdType(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
#ifdef MILVUS_GPU_VERSION
    } else if (parent_key == CONFIG_GPU_RESOURCE) {
        if (child_key == CONFIG_GPU_RESOURCE_ENABLE) {
            status = SetGpuResourceConfigEnable(value);
        } else if (child_key == CONFIG_GPU_RESOURCE_CACHE_CAPACITY) {
            status = SetGpuResourceConfigCacheCapacity(value);
        } else if (child_key == CONFIG_GPU_RESOURCE_CACHE_THRESHOLD) {
            status = SetGpuResourceConfigCacheThreshold(value);
        } else if (child_key == CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD) {
            status = SetGpuResourceConfigGpuSearchThreshold(value);
        } else if (child_key == CONFIG_GPU_RESOURCE_SEARCH_RESOURCES) {
            status = SetGpuResourceConfigSearchResources(value);
        } else if (child_key == CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES) {
            status = SetGpuResourceConfigBuildIndexResources(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
#endif
    } else if (parent_key == CONFIG_TRACING) {
        if (child_key == CONFIG_TRACING_JSON_CONFIG_PATH) {
            status = SetTracingConfigJsonConfigPath(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_WAL) {
        if (child_key == CONFIG_WAL_ENABLE) {
            status = SetWalConfigEnable(value);
        } else if (child_key == CONFIG_WAL_RECOVERY_ERROR_IGNORE) {
            status = SetWalConfigRecoveryErrorIgnore(value);
        } else if (child_key == CONFIG_WAL_BUFFER_SIZE) {
            status = SetWalConfigBufferSize(value);
        } else if (child_key == CONFIG_WAL_WAL_PATH) {
            status = SetWalConfigWalPath(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    } else if (parent_key == CONFIG_LOGS) {
        if (child_key == CONFIG_LOGS_LEVEL) {
            status = SetLogsLevel(value);
        } else if (child_key == CONFIG_LOGS_TRACE_ENABLE) {
            status = SetLogsTraceEnable(value);
        } else if (child_key == CONFIG_LOGS_PATH) {
            status = SetLogsPath(value);
        } else if (child_key == CONFIG_LOGS_MAX_LOG_FILE_SIZE) {
            status = SetLogsMaxLogFileSize(value);
        } else if (child_key == CONFIG_LOGS_LOG_ROTATE_NUM) {
            status = SetLogsLogRotateNum(value);
        } else {
            status = Status(SERVER_UNEXPECTED_ERROR, invalid_node_str);
        }
    }

    if (status.ok()) {
        status = UpdateFileConfigFromMem(parent_key, child_key);
        if (status.ok() &&
            !(parent_key == CONFIG_CACHE || parent_key == CONFIG_ENGINE || parent_key == CONFIG_GPU_RESOURCE)) {
            restart_required_ = true;
        }
    }

    return status;
}

//////////////////////////////////////////////////////////////
Status
Config::ProcessConfigCli(std::string& result, const std::string& cmd) {
    std::vector<std::string> tokens;
    std::vector<std::string> nodes;
    StringHelpFunctions::SplitStringByDelimeter(cmd, " ", tokens);
    if (tokens[0] == "get_config") {
        if (tokens.size() != 2) {
            return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
        }
        if (tokens[1] == "*") {
            GetConfigJsonStr(result);
            return Status::OK();
        } else {
            StringHelpFunctions::SplitStringByDelimeter(tokens[1], CONFIG_NODE_DELIMITER, nodes);
            if (nodes.size() < 2) {
                return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
            } else if (nodes.size() > 2) {
                // to support case likes network.bind.address
                std::string result;
                std::vector<std::string> nodes_s(nodes.begin() + 1, nodes.end());
                StringHelpFunctions::MergeStringWithDelimeter(nodes_s, CONFIG_NODE_DELIMITER, result);
                nodes[1] = result;
            }
            // if (nodes.size() != 2) {
            //     return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
            // }
            return GetConfigCli(result, nodes[0], nodes[1]);
        }
    } else if (tokens[0] == "set_config") {
        if (tokens.size() != 3) {
            return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
        }
        StringHelpFunctions::SplitStringByDelimeter(tokens[1], CONFIG_NODE_DELIMITER, nodes);
        if (nodes.size() < 2) {
            return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
        } else if (nodes.size() > 2) {
            // to support case likes network.bind.address
            std::string result;
            std::vector<std::string> nodes_s(nodes.begin() + 1, nodes.end());
            StringHelpFunctions::MergeStringWithDelimeter(nodes_s, CONFIG_NODE_DELIMITER, result);
            nodes[1] = result;
        }
        return SetConfigCli(nodes[0], nodes[1], tokens[2]);
    } else {
        return Status(SERVER_UNEXPECTED_ERROR, "Invalid command: " + cmd);
    }
}

Status
Config::GenUniqueIdentityID(const std::string& identity, std::string& uid) {
    std::vector<std::string> elements;
    elements.push_back(identity);

    // get current process id
    int64_t pid = getpid();
    elements.push_back(std::to_string(pid));

    // get current thread id
    std::stringstream ss;
    ss << std::this_thread::get_id();
    elements.push_back(ss.str());

    // get current timestamp
    auto time_now = std::chrono::system_clock::now();
    auto duration_in_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(time_now.time_since_epoch());
    elements.push_back(std::to_string(duration_in_ms.count()));

    StringHelpFunctions::MergeStringWithDelimeter(elements, "-", uid);

    return Status::OK();
}

Status
Config::UpdateFileConfigFromMem(const std::string& parent_key, const std::string& child_key) {
    if (access(config_file_.c_str(), F_OK | R_OK) != 0) {
        return Status(SERVER_UNEXPECTED_ERROR, "Cannot find configure file: " + config_file_);
    }

    // Store original configure file
    std::string ori_file = config_file_ + ".ori";
    if (access(ori_file.c_str(), F_OK) != 0) {
        std::fstream fin(config_file_, std::ios::in);
        std::ofstream fout(ori_file);

        if (!fin.is_open() || !fout.is_open()) {
            return Status(SERVER_UNEXPECTED_ERROR, "Cannot open conf file. Store original conf file failed");
        }
        fout << fin.rdbuf();
        fout.flush();
        fout.close();
        fin.close();
    }

    std::string value;
    auto status = GetConfigValueInMem(parent_key, child_key, value);
    if (!status.ok()) {
        return status;
    }

    // convert value string to standard string stored in yaml file
    std::string value_str;
    if (child_key == CONFIG_CACHE_CACHE_INSERT_DATA ||
        // child_key == CONFIG_STORAGE_S3_ENABLE ||
        child_key == CONFIG_METRIC_ENABLE_MONITOR || child_key == CONFIG_GPU_RESOURCE_ENABLE ||
        child_key == CONFIG_WAL_ENABLE || child_key == CONFIG_WAL_RECOVERY_ERROR_IGNORE) {
        bool ok = false;
        STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(value, ok));
        value_str = ok ? "true" : "false";
    } else if (child_key == CONFIG_GPU_RESOURCE_SEARCH_RESOURCES ||
               child_key == CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES) {
        std::vector<std::string> vec;
        StringHelpFunctions::SplitStringByDelimeter(value, ",", vec);
        for (auto& s : vec) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            value_str += "\n    - " + s;
        }
    } else {
        value_str = value;
    }

    std::fstream conf_fin(config_file_, std::ios::in);
    if (!conf_fin.is_open()) {
        return Status(SERVER_UNEXPECTED_ERROR, "Cannot open conf file: " + config_file_);
    }

    bool parent_key_read = false;
    std::string conf_str, line;
    while (getline(conf_fin, line)) {
        if (!parent_key_read) {
            conf_str += line + "\n";
            // TODO: danger
            if (not(line.empty() || line.find_first_of('#') == 0))
                if (line.find(parent_key) == 0)
                    parent_key_read = true;
            continue;
        }

        if (line.find_first_of('#') == 0) {
            status = Status(SERVER_UNEXPECTED_ERROR, "Cannot find child key: " + child_key + ", line is " + line);
            break;
        }

        if (line.find(child_key) != std::string::npos) {
            // may loss comments here, need to extract comments from line
            conf_str += "  " + child_key + ": " + value_str + "\n";
            break;
        }

        conf_str += line + "\n";
    }

    // values of gpu resources are sequences, need to remove old here
    if (child_key == CONFIG_GPU_RESOURCE_SEARCH_RESOURCES || child_key == CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES) {
        while (getline(conf_fin, line)) {
            if (line.find("- gpu") != std::string::npos)
                continue;

            conf_str += line + "\n";
            if (!line.empty() && line.size() > 2 && isalnum(line.at(2))) {
                break;
            }
        }
    }

    if (status.ok()) {
        while (getline(conf_fin, line)) {
            conf_str += line + "\n";
        }
        conf_fin.close();

        std::fstream fout(config_file_, std::ios::out | std::ios::trunc);
        fout << conf_str;
        fout.flush();
        fout.close();
    }

    return status;
}

Status
Config::RegisterCallBack(const std::string& node, const std::string& sub_node, const std::string& key,
                         ConfigCallBackF& cb) {
    std::string cb_node = node + "." + sub_node;
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (config_callback_.find(cb_node) == config_callback_.end()) {
        return Status(SERVER_UNEXPECTED_ERROR, cb_node + " is not supported changed in mem");
    }

    auto& callback_map = config_callback_.at(cb_node);
    callback_map[key] = cb;

    return Status::OK();
}

Status
Config::CancelCallBack(const std::string& node, const std::string& sub_node, const std::string& key) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (config_callback_.empty() || key.empty()) {
        return Status::OK();
    }

    std::string cb_node = node + "." + sub_node;
    if (config_callback_.find(cb_node) == config_callback_.end()) {
        return Status(SERVER_UNEXPECTED_ERROR, cb_node + " cannot found in callback map");
    }

    auto& cb_map = config_callback_.at(cb_node);
    cb_map.erase(key);

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
Status
Config::CheckConfigVersion(const std::string& value) {
    if (milvus_config_version_map.find(MILVUS_VERSION) != milvus_config_version_map.end()) {
        bool exist_error = milvus_config_version_map.at(MILVUS_VERSION) != value;
        fiu_do_on("check_config_version_fail", exist_error = true);
        if (exist_error) {
            std::string msg = "Invalid config version: " + value +
                              ". Expected config version: " + milvus_config_version_map.at(MILVUS_VERSION);
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

/* cluster config */
Status
Config::CheckClusterConfigEnable(const std::string& value) {
    return ValidateStringIsBool(value);
}

Status
Config::CheckClusterConfigRole(const std::string& value) {
    fiu_return_on("check_config_cluster_role_fail",
                  Status(SERVER_INVALID_ARGUMENT, "cluster.role is not one of rw and ro."));

    if (value != "rw" && value != "ro") {
        return Status(SERVER_INVALID_ARGUMENT, "cluster.role is not one of rw and ro.");
    }
    return Status::OK();
}

/* general config */
Status
Config::CheckGeneralConfigTimezone(const std::string& value) {
    fiu_return_on("check_config_timezone_fail", Status(SERVER_INVALID_ARGUMENT, "Invalid general.timezone: " + value));

    if (value.length() <= 3) {
        return Status(SERVER_INVALID_ARGUMENT, "Invalid general.timezone: " + value);
    } else {
        if (value.substr(0, 3) != "UTC") {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid general.timezone: " + value);
        } else {
            if (!IsNumber(value.substr(4))) {
                return Status(SERVER_INVALID_ARGUMENT, "Invalid general.timezone: " + value);
            }
        }
    }
    return Status::OK();
}

Status
Config::CheckGeneralConfigMetaURI(const std::string& value) {
    auto exist_error = !ValidateDbURI(value).ok();
    fiu_do_on("check_config_meta_uri_fail", exist_error = true);

    if (exist_error) {
        std::string msg =
            "Invalid meta uri: " + value + ". Possible reason: general.meta_uri is invalid. " +
            "The correct format should be like sqlite://:@:/ or mysql://root:123456@127.0.0.1:3306/milvus.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

/* network config */
Status
Config::CheckNetworkConfigBindAddress(const std::string& value) {
    auto exist_error = !ValidateIpAddress(value).ok();
    fiu_do_on("check_config_bind_address_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid server IP address: " + value + ". Possible reason: network.bind.address is invalid.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckNetworkConfigBindPort(const std::string& value) {
    auto exist_error = !ValidateStringIsNumber(value).ok();
    fiu_do_on("check_config_bind_port_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid server port: " + value + ". Possible reason: network.bind.port is not a number.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        try {
            int32_t port = std::stoi(value);
            if (!(port > PORT_NUMBER_MIN && port < PORT_NUMBER_MAX)) {
                std::string msg = "Invalid server port: " + value +
                                  ". Possible reason: network.bind.port is not in range (1024, 65535).";
                return Status(SERVER_INVALID_ARGUMENT, msg);
            }
        } catch (...) {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid network.bind.port: " + value);
        }
    }
    return Status::OK();
}

Status
Config::CheckNetworkConfigHTTPEnable(const std::string& value) {
    return ValidateStringIsBool(value);
}

Status
Config::CheckNetworkConfigHTTPPort(const std::string& value) {
    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid web server port: " + value + ". Possible reason: network.http.port is not a number.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        try {
            int32_t port = std::stoi(value);
            if (!(port > PORT_NUMBER_MIN && port < PORT_NUMBER_MAX)) {
                std::string msg = "Invalid web server port: " + value +
                                  ". Possible reason: network.http.port is not in range (1024, 65535).";
                return Status(SERVER_INVALID_ARGUMENT, msg);
            }
        } catch (...) {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid network.http.port: " + value);
        }
    }
    return Status::OK();
}

/* DB config */
Status
Config::CheckDBConfigArchiveDiskThreshold(const std::string& value) {
    auto exist_error = !ValidateStringIsNumber(value).ok();
    fiu_do_on("check_config_archive_disk_threshold_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid archive disk threshold: " + value +
                          ". Possible reason: db_config.archive_disk_threshold is invalid.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckDBConfigArchiveDaysThreshold(const std::string& value) {
    auto exist_error = !ValidateStringIsNumber(value).ok();
    fiu_do_on("check_config_archive_days_threshold_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid archive days threshold: " + value +
                          ". Possible reason: db_config.archive_days_threshold is invalid.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

/* storage config */
Status
Config::CheckStorageConfigPath(const std::string& value) {
    fiu_return_on("check_config_path_fail", Status(SERVER_INVALID_ARGUMENT, ""));
    if (value.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, "storage.path is empty.");
    }

    return ValidateStoragePath(value);
}

Status
Config::CheckStorageConfigAutoFlushInterval(const std::string& value) {
    auto exist_error = !ValidateStringIsNumber(value).ok();
    fiu_do_on("check_config_auto_flush_interval_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid storage configuration auto_flush_interval: " + value +
                          ". Possible reason: storage.auto_flush_interval is not a natural number.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}

Status
Config::CheckStorageConfigFileCleanupTimeout(const std::string& value) {
    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid file_cleanup_timeout: " + value +
                          ". Possible reason: storage.file_cleanup_timeout is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        int64_t file_cleanup_timeout = std::stoll(value);
        if (file_cleanup_timeout < CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MIN ||
            file_cleanup_timeout > CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MAX) {
            std::string msg = "Invalid file_cleanup_timeout: " + value +
                              ". Possible reason: storage.file_cleanup_timeout is not in range [" +
                              std::to_string(CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MIN) + ", " +
                              std::to_string(CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_MIN) + "].";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }

    return Status::OK();
}

// Status
// Config::CheckStorageConfigS3Enable(const std::string& value) {
//    if (!ValidateStringIsBool(value).ok()) {
//        std::string msg =
//            "Invalid storage config: " + value + ". Possible reason: storage_config.s3_enable is not a boolean.";
//        return Status(SERVER_INVALID_ARGUMENT, msg);
//    }
//    return Status::OK();
// }
//
// Status
// Config::CheckStorageConfigS3Address(const std::string& value) {
//    if (!ValidateIpAddress(value).ok()) {
//        std::string msg = "Invalid s3 address: " + value + ". Possible reason: storage_config.s3_address is invalid.";
//        return Status(SERVER_INVALID_ARGUMENT, msg);
//    }
//    return Status::OK();
// }
//
// Status
// Config::CheckStorageConfigS3Port(const std::string& value) {
//    if (!ValidateStringIsNumber(value).ok()) {
//        std::string msg = "Invalid s3 port: " + value + ". Possible reason: storage_config.s3_port is not a number.";
//        return Status(SERVER_INVALID_ARGUMENT, msg);
//    } else {
//        try {
//            int32_t port = std::stoi(value);
//            if (!(port > PORT_NUMBER_MIN && port < PORT_NUMBER_MAX)) {
//                std::string msg = "Invalid s3 port: " + value +
//                                  ". Possible reason: storage_config.s3_port is not in range (1024, 65535).";
//                return Status(SERVER_INVALID_ARGUMENT, msg);
//            }
//        } catch (...) {
//            return Status(SERVER_INVALID_ARGUMENT, "Invalid storage_config.s3_port: " + value);
//        }
//    }
//    return Status::OK();
// }
//
// Status
// Config::CheckStorageConfigS3AccessKey(const std::string& value) {
//    if (value.empty()) {
//        return Status(SERVER_INVALID_ARGUMENT, "storage_config.s3_access_key is empty.");
//    }
//    return Status::OK();
// }
//
// Status
// Config::CheckStorageConfigS3SecretKey(const std::string& value) {
//    if (value.empty()) {
//        return Status(SERVER_INVALID_ARGUMENT, "storage_config.s3_secret_key is empty.");
//    }
//    return Status::OK();
// }
//
// Status
// Config::CheckStorageConfigS3Bucket(const std::string& value) {
//    if (value.empty()) {
//        return Status(SERVER_INVALID_ARGUMENT, "storage_config.s3_bucket is empty.");
//    }
//    return Status::OK();
// }

/* metric config */
Status
Config::CheckMetricConfigEnableMonitor(const std::string& value) {
    auto exist_error = !ValidateStringIsBool(value).ok();
    fiu_do_on("check_config_enable_monitor_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid metric config: " + value + ". Possible reason: metric.enable is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckMetricConfigAddress(const std::string& value) {
    if (!ValidateIpAddress(value).ok()) {
        std::string msg = "Invalid metric ip: " + value + ". Possible reason: metric.ip is invalid.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckMetricConfigPort(const std::string& value) {
    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid metric port: " + value + ". Possible reason: metric.port is not a number.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        try {
            int32_t port = std::stoi(value);
            if (!(port > PORT_NUMBER_MIN && port < PORT_NUMBER_MAX)) {
                std::string msg =
                    "Invalid metric port: " + value + ". Possible reason: metric.port is not in range (1024, 65535).";
                return Status(SERVER_INVALID_ARGUMENT, msg);
            }
        } catch (...) {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid metric.port: " + value);
        }
    }
    return Status::OK();
}

/* cache config */
Status
Config::CheckCacheConfigCpuCacheCapacity(const std::string& value) {
    fiu_return_on("check_config_cache_size_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    std::string err;
    int64_t cache_size = parse_bytes(value, err);
    if (not err.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, err);
    } else {
        if (cache_size <= 0) {
            std::string msg = "Invalid cpu cache capacity: " + value +
                              ". Possible reason: cache.cache_size is not a positive integer.";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }

        int64_t total_mem = 0, free_mem = 0;
        GetSystemMemInfo(total_mem, free_mem);
        if (cache_size >= total_mem) {
            std::string msg =
                "Invalid cpu cache size: " + value + ". Possible reason: cache.cache_size exceeds system memory.";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        } else if (static_cast<double>(cache_size) > static_cast<double>(total_mem * 0.9)) {
            std::cerr << "WARNING: cpu cache size value is too big" << std::endl;
        }

        std::string str = GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_INSERT_BUFFER_SIZE, "0");

        int64_t insert_buffer_size = parse_bytes(str, err);
        fiu_do_on("Config.CheckCacheConfigCpuCacheCapacity.large_insert_buffer", insert_buffer_size = total_mem + 1);
        if (insert_buffer_size + cache_size >= total_mem) {
            std::string msg = "Invalid cpu cache size: " + value +
                              ". Possible reason: sum of cache.cache_size and "
                              "cache.insert_buffer_size exceeds system memory.";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

Status
Config::CheckCacheConfigCpuCacheThreshold(const std::string& value) {
    fiu_return_on("check_config_cpu_cache_threshold_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsFloat(value).ok()) {
        std::string msg = "Invalid cpu cache threshold: " + value +
                          ". Possible reason: cache_config.cpu_cache_threshold is not in range (0.0, 1.0].";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        float cpu_cache_threshold = std::stof(value);
        if (cpu_cache_threshold <= 0.0 || cpu_cache_threshold >= 1.0) {
            std::string msg = "Invalid cpu cache threshold: " + value +
                              ". Possible reason: cache_config.cpu_cache_threshold is not in range (0.0, 1.0].";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

Status
Config::CheckCacheConfigInsertBufferSize(const std::string& value) {
    fiu_return_on("check_config_insert_buffer_size_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    std::string err;
    int64_t buffer_size = parse_bytes(value, err);
    if (not err.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, err);
    } else {
        if (buffer_size <= 0) {
            std::string msg = "Invalid insert buffer size: " + value +
                              ". Possible reason: cache.insert_buffer_size is not a positive integer.";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }

        std::string str = GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_CAPACITY, "0");
        std::string err;
        int64_t cache_size = parse_bytes(str, err);

        int64_t total_mem = 0, free_mem = 0;
        GetSystemMemInfo(total_mem, free_mem);
        if (buffer_size + cache_size >= total_mem) {
            std::string msg = "Invalid insert buffer size: " + value +
                              ". Possible reason: sum of cache.cache_size and "
                              "cache.insert_buffer_size exceeds system memory.";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

Status
Config::CheckCacheConfigCacheInsertData(const std::string& value) {
    fiu_return_on("check_config_cache_insert_data_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsBool(value).ok()) {
        std::string msg = "Invalid cache insert data option: " + value +
                          ". Possible reason: cache_config.cache_insert_data is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckCacheConfigPreloadCollection(const std::string& value) {
    fiu_return_on("check_config_preload_collection_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (value.empty() || value == "*") {
        return Status::OK();
    }

    std::vector<std::string> tables;
    StringHelpFunctions::SplitStringByDelimeter(value, ",", tables);

    std::unordered_set<std::string> table_set;

    for (auto& collection : tables) {
        bool exist = false;
        auto status = DBWrapper::DB()->HasNativeCollection(collection, exist);
        if (!(status.ok() && exist)) {
            return Status(SERVER_COLLECTION_NOT_EXIST, "Collection " + collection + " not exist");
        }
        table_set.insert(collection);
    }

    if (table_set.size() != tables.size()) {
        std::string msg =
            "Invalid preload tables. "
            "Possible reason: cache.preload_collection contains duplicate collection.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}

/* engine config */
Status
Config::CheckEngineConfigUseBlasThreshold(const std::string& value) {
    fiu_return_on("check_config_use_blas_threshold_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid use blas threshold: " + value +
                          ". Possible reason: engine_config.use_blas_threshold is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckEngineConfigOmpThreadNum(const std::string& value) {
    fiu_return_on("check_config_omp_thread_num_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid omp thread num: " + value +
                          ". Possible reason: engine_config.omp_thread_num is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    int64_t omp_thread = std::stoll(value);
    int64_t sys_thread_cnt = 8;
    GetSystemAvailableThreads(sys_thread_cnt);
    if (omp_thread > sys_thread_cnt) {
        std::string msg = "Invalid omp thread num: " + value +
                          ". Possible reason: engine_config.omp_thread_num exceeds system cpu cores.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckEngineConfigSimdType(const std::string& value) {
    fiu_return_on("check_config_simd_type_fail",
                  Status(SERVER_INVALID_ARGUMENT, "engine_config.simd_type is not one of avx512, avx2, sse and auto."));

    if (value != "avx512" && value != "avx2" && value != "sse" && value != "auto") {
        return Status(SERVER_INVALID_ARGUMENT, "engine_config.simd_type is not one of avx512, avx2, sse and auto.");
    }
    return Status::OK();
}

/* gpu resource config */
#ifdef MILVUS_GPU_VERSION
Status
Config::CheckGpuResourceConfigEnable(const std::string& value) {
    fiu_return_on("check_config_gpu_resource_enable_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsBool(value).ok()) {
        std::string msg = "Invalid gpu resource config: " + value + ". Possible reason: gpu.enable is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckGpuResourceConfigCacheCapacity(const std::string& value) {
    fiu_return_on("check_gpu_cache_size_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    std::string err;
    int64_t gpu_cache_size = parse_bytes(value, err);
    if (not err.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, err);
    } else {
        if (gpu_cache_size < 0) {
            std::string msg = "gpu.cache_size must greater than 0, now is " + std::to_string(gpu_cache_size) + ".";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
        std::vector<int64_t> gpu_ids;
        STATUS_CHECK(GetGpuResourceConfigBuildIndexResources(gpu_ids));

        for (int64_t gpu_id : gpu_ids) {
            int64_t gpu_memory;
            if (!GetGpuMemory(gpu_id, gpu_memory).ok()) {
                std::string msg = "Fail to get GPU memory for GPU device: " + std::to_string(gpu_id);
                return Status(SERVER_UNEXPECTED_ERROR, msg);
            } else if (gpu_cache_size >= gpu_memory) {
                std::string msg =
                    "Invalid gpu cache capacity: " + value + ". Possible reason: gpu.cache_size exceeds GPU memory.";
                return Status(SERVER_INVALID_ARGUMENT, msg);
            } else if (gpu_cache_size > (double)gpu_memory * 0.9) {
                std::cerr << "Warning: gpu cache size value is too big" << std::endl;
            }
        }
    }
    return Status::OK();
}

Status
Config::CheckGpuResourceConfigCacheThreshold(const std::string& value) {
    fiu_return_on("check_config_gpu_resource_cache_threshold_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsFloat(value).ok()) {
        std::string msg = "Invalid gpu cache threshold: " + value +
                          ". Possible reason: gpu.cache_threshold is not in range (0.0, 1.0].";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        float gpu_cache_threshold = std::stof(value);
        if (gpu_cache_threshold <= 0.0 || gpu_cache_threshold >= 1.0) {
            std::string msg = "Invalid gpu cache threshold: " + value +
                              ". Possible reason: gpu.cache_threshold is not in range (0.0, 1.0].";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

Status
Config::CheckGpuResourceConfigGpuSearchThreshold(const std::string& value) {
    fiu_return_on("check_config_gpu_search_threshold_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (!ValidateStringIsNumber(value).ok()) {
        std::string msg = "Invalid gpu search threshold: " + value +
                          ". Possible reason: gpu.gpu_search_threshold is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
CheckGpuResource(const std::string& value) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    const std::regex pat("gpu(\\d+)");
    std::smatch m;
    if (!std::regex_match(s, m, pat)) {
        std::string msg =
            "Invalid gpu resource: " + value + ". Possible reason: gpu is not in the format of cpux or gpux";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    if (s.compare(0, 3, "gpu") == 0) {
        try {
            int32_t gpu_index = std::stoi(s.substr(3));
            if (!ValidateGpuIndex(gpu_index).ok()) {
                std::string msg =
                    "Invalid gpu resource: " + value + ". Possible reason: gpu does not match with the hardware.";
                return Status(SERVER_INVALID_ARGUMENT, msg);
            }
        } catch (...) {
            return Status(SERVER_INVALID_ARGUMENT, "Invalid gpu_resource_config: " + value);
        }
    }

    return Status::OK();
}

Status
Config::CheckGpuResourceConfigSearchResources(const std::vector<std::string>& value) {
    fiu_return_on("check_gpu_search_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (value.empty()) {
        std::string msg =
            "Invalid gpu search resource. "
            "Possible reason: gpu.search_resources is empty.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    std::unordered_set<std::string> value_set;
    for (auto& resource : value) {
        STATUS_CHECK(CheckGpuResource(resource));
        value_set.insert(resource);
    }

    if (value_set.size() != value.size()) {
        std::string msg =
            "Invalid gpu build search resource. "
            "Possible reason: gpu.gpu_search_resources contains duplicate resources.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}

Status
Config::CheckGpuResourceConfigBuildIndexResources(const std::vector<std::string>& value) {
    fiu_return_on("check_gpu_build_index_fail", Status(SERVER_INVALID_ARGUMENT, ""));

    if (value.empty()) {
        std::string msg =
            "Invalid gpu build index resource. "
            "Possible reason: gpu.build_index_resources is empty.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    std::unordered_set<std::string> value_set;
    for (auto& resource : value) {
        STATUS_CHECK(CheckGpuResource(resource));
        value_set.insert(resource);
    }

    if (value_set.size() != value.size()) {
        std::string msg =
            "Invalid gpu build index resource. "
            "Possible reason: gpu.build_index_resources contains duplicate resources.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}
#endif

/* tracing config */
Status
Config::CheckTracingConfigJsonConfigPath(const std::string& value) {
    std::string msg = "Invalid wal config: " + value +
                      ". Possible reason: tracing_config.json_config_path is not supported to configure.";
    return Status(SERVER_INVALID_ARGUMENT, msg);
}

/* wal config */
Status
Config::CheckWalConfigEnable(const std::string& value) {
    auto exist_error = !ValidateStringIsBool(value).ok();
    fiu_do_on("check_config_wal_enable_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid wal config: " + value + ". Possible reason: wal.enable is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckWalConfigRecoveryErrorIgnore(const std::string& value) {
    auto exist_error = !ValidateStringIsBool(value).ok();
    fiu_do_on("check_config_wal_recovery_error_ignore_fail", exist_error = true);

    if (exist_error) {
        std::string msg =
            "Invalid wal config: " + value + ". Possible reason: wal.recovery_error_ignore is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckWalConfigBufferSize(const std::string& value) {
    std::string err;
    auto buffer_size = parse_bytes(value, err);
    auto exist_error = not err.empty();
    fiu_do_on("check_config_wal_buffer_size_fail", exist_error = true);

    if (exist_error || buffer_size < 0) {
        std::string msg =
            "Invalid wal buffer size: " + value + ". Possible reason: wal.buffer_size is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckWalConfigWalPath(const std::string& value) {
    fiu_return_on("check_wal_path_fail", Status(SERVER_INVALID_ARGUMENT, ""));
    if (value.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, "wal.path is empty!");
    }

    return ValidateStoragePath(value);
}

/* logs config */
Status
Config::CheckLogsLevel(const std::string& value) {
    fiu_return_on("check_logs_level_fail", Status(SERVER_INVALID_ARGUMENT, ""));
    if (value.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, "logs.level is empty!");
    }
    return ValidateLogLevel(value);
}

Status
Config::CheckLogsTraceEnable(const std::string& value) {
    auto exist_error = !ValidateStringIsBool(value).ok();
    fiu_do_on("check_logs_trace_enable_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid logs config: " + value + ". Possible reason: logs.trace.enable is not a boolean.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }
    return Status::OK();
}

Status
Config::CheckLogsPath(const std::string& value) {
    fiu_return_on("check_logs_path_fail", Status(SERVER_INVALID_ARGUMENT, ""));
    if (value.empty()) {
        return Status(SERVER_INVALID_ARGUMENT, "logs.path is empty!");
    }

    return ValidateStoragePath(value);
}

Status
Config::CheckLogsMaxLogFileSize(const std::string& value) {
    std::string err;
    int64_t max_log_file_size = parse_bytes(value, err);
    auto exist_error = not err.empty();
    fiu_do_on("check_logs_max_log_file_size_fail", exist_error = true);

    if (exist_error) {
        std::string msg = "Invalid max_log_file_size: " + value +
                          ". Possible reason: logs.max_log_file_size is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        if (max_log_file_size < CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN ||
            max_log_file_size > CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX) {
            std::string msg = "Invalid max_log_file_size: " + value +
                              ". Possible reason: logs.max_log_file_size is not in range [" +
                              std::to_string(CONFIG_LOGS_MAX_LOG_FILE_SIZE_MIN) + ", " +
                              std::to_string(CONFIG_LOGS_MAX_LOG_FILE_SIZE_MAX) + "].";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

Status
Config::CheckLogsLogRotateNum(const std::string& value) {
    auto exist_error = !ValidateStringIsNumber(value).ok();
    fiu_do_on("check_logs_log_rotate_num_fail", exist_error = true);

    if (exist_error) {
        std::string msg =
            "Invalid log_rotate_num: " + value + ". Possible reason: logs.log_rotate_num is not a positive integer.";
        return Status(SERVER_INVALID_ARGUMENT, msg);
    } else {
        int64_t log_rotate_num = std::stoll(value);
        if (log_rotate_num < CONFIG_LOGS_LOG_ROTATE_NUM_MIN || log_rotate_num > CONFIG_LOGS_LOG_ROTATE_NUM_MAX) {
            std::string msg = "Invalid log_rotate_num: " + value +
                              ". Possible reason: logs.log_rotate_num is not in range [" +
                              std::to_string(CONFIG_LOGS_LOG_ROTATE_NUM_MIN) + ", " +
                              std::to_string(CONFIG_LOGS_LOG_ROTATE_NUM_MAX) + "].";
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    }
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
ConfigNode&
Config::GetConfigRoot() {
    ConfigMgr* mgr = YamlConfigMgr::GetInstance();
    return mgr->GetRootNode();
}

ConfigNode&
Config::GetConfigNode(const std::string& name) {
    return GetConfigRoot().GetChild(name);
}

bool
Config::ConfigNodeValid(const std::string& parent_key, const std::string& child_key) {
    if (config_map_.find(parent_key) == config_map_.end()) {
        return false;
    }
    return config_map_[parent_key].count(child_key) != 0;
}

Status
Config::GetConfigValueInMem(const std::string& parent_key, const std::string& child_key, std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_map_.find(parent_key) != config_map_.end() &&
        config_map_[parent_key].find(child_key) != config_map_[parent_key].end()) {
        value = config_map_[parent_key][child_key];
        return Status::OK();
    }
    return Status(SERVER_UNEXPECTED_ERROR, "key not exist");
}

Status
Config::SetConfigValueInMem(const std::string& parent_key, const std::string& child_key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[parent_key][child_key] = value;
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
std::string
Config::GetConfigStr(const std::string& parent_key, const std::string& child_key, const std::string& default_value) {
    std::string value;
    if (!GetConfigValueInMem(parent_key, child_key, value).ok()) {
        value = GetConfigNode(parent_key).GetValue(child_key, default_value);
        SetConfigValueInMem(parent_key, child_key, value);
    }
    return value;
}

std::string
Config::GetConfigSequenceStr(const std::string& parent_key, const std::string& child_key, const std::string& delim,
                             const std::string& default_value) {
    std::string value;
    if (!GetConfigValueInMem(parent_key, child_key, value).ok()) {
        std::vector<std::string> sequence = GetConfigNode(parent_key).GetSequence(child_key);
        if (sequence.empty()) {
            value = default_value;
        } else {
            StringHelpFunctions::MergeStringWithDelimeter(sequence, delim, value);
        }
        SetConfigValueInMem(parent_key, child_key, value);
    }
    return value;
}

Status
Config::GetConfigVersion(std::string& value) {
    value = GetConfigRoot().GetValue(CONFIG_VERSION);
    return CheckConfigVersion(value);
}

Status
Config::ExecCallBacks(const std::string& node, const std::string& sub_node, const std::string& value) {
    auto status = Status::OK();
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (config_callback_.empty()) {
        return Status(SERVER_UNEXPECTED_ERROR, "Callback map is empty. Cannot take effect in-service");
    }

    std::string cb_node = node + "." + sub_node;
    if (config_callback_.find(cb_node) == config_callback_.end()) {
        return Status(SERVER_UNEXPECTED_ERROR,
                      "Cannot find " + cb_node + " in callback map, cannot take effect in-service");
    }

    auto& cb_map = config_callback_.at(cb_node);
    for (auto& cb_kv : cb_map) {
        auto& cd = cb_kv.second;
        status = cd(value);
        if (!status.ok()) {
            break;
        }
    }

    return status;
}

/* cluster config */
Status
Config::GetClusterConfigEnable(bool& value) {
    std::string str = GetConfigStr(CONFIG_CLUSTER, CONFIG_CLUSTER_ENABLE, CONFIG_CLUSTER_ENABLE_DEFAULT);
    STATUS_CHECK(CheckClusterConfigEnable(str));
    return StringHelpFunctions::ConvertToBoolean(str, value);
}

Status
Config::GetClusterConfigRole(std::string& value) {
    value = GetConfigStr(CONFIG_CLUSTER, CONFIG_CLUSTER_ROLE, CONFIG_CLUSTER_ROLE_DEFAULT);
    return CheckClusterConfigRole(value);
}

/* general config */
Status
Config::GetGeneralConfigTimezone(std::string& value) {
    value = GetConfigStr(CONFIG_GENERAL, CONFIG_GENERAL_TIMEZONE, CONFIG_GENERAL_TIMEZONE_DEFAULT);
    return CheckGeneralConfigTimezone(value);
}

Status
Config::GetGeneralConfigMetaURI(std::string& value) {
    value = GetConfigStr(CONFIG_GENERAL, CONFIG_GENERAL_METAURI, CONFIG_GENERAL_METAURI_DEFAULT);
    return CheckGeneralConfigMetaURI(value);
}

/* network config */
Status
Config::GetNetworkConfigBindAddress(std::string& value) {
    value = GetConfigStr(CONFIG_NETWORK, CONFIG_NETWORK_BIND_ADDRESS, CONFIG_NETWORK_BIND_ADDRESS_DEFAULT);
    return CheckNetworkConfigBindAddress(value);
}

Status
Config::GetNetworkConfigBindPort(std::string& value) {
    value = GetConfigStr(CONFIG_NETWORK, CONFIG_NETWORK_BIND_PORT, CONFIG_NETWORK_BIND_PORT_DEFAULT);
    return CheckNetworkConfigBindPort(value);
}

Status
Config::GetNetworkConfigHTTPEnable(bool& value) {
    std::string str = GetConfigStr(CONFIG_NETWORK, CONFIG_NETWORK_HTTP_ENABLE, CONFIG_NETWORK_HTTP_ENABLE_DEFAULT);
    STATUS_CHECK(CheckNetworkConfigHTTPEnable(str));
    return StringHelpFunctions::ConvertToBoolean(str, value);
}

Status
Config::GetNetworkConfigHTTPPort(std::string& value) {
    value = GetConfigStr(CONFIG_NETWORK, CONFIG_NETWORK_HTTP_PORT, CONFIG_NETWORK_HTTP_PORT_DEFAULT);
    return CheckNetworkConfigHTTPPort(value);
}

/* DB config */
Status
Config::GetDBConfigArchiveDiskThreshold(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_DB, CONFIG_DB_ARCHIVE_DISK_THRESHOLD, CONFIG_DB_ARCHIVE_DISK_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckDBConfigArchiveDiskThreshold(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetDBConfigArchiveDaysThreshold(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_DB, CONFIG_DB_ARCHIVE_DAYS_THRESHOLD, CONFIG_DB_ARCHIVE_DAYS_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckDBConfigArchiveDaysThreshold(str));
    value = std::stoll(str);
    return Status::OK();
}

/* storage config */
Status
Config::GetStorageConfigPath(std::string& value) {
    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_PATH, CONFIG_STORAGE_PATH_DEFAULT);
    return CheckStorageConfigPath(value);
}

Status
Config::GetStorageConfigAutoFlushInterval(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_AUTO_FLUSH_INTERVAL, CONFIG_STORAGE_AUTO_FLUSH_INTERVAL_DEFAULT);
    STATUS_CHECK(CheckStorageConfigAutoFlushInterval(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetStorageConfigFileCleanupTimeup(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT, CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT_DEFAULT);
    STATUS_CHECK(CheckStorageConfigFileCleanupTimeout(str));
    value = std::stoll(str);
    return Status::OK();
}

// Status
// Config::GetStorageConfigS3Enable(bool& value) {
//    std::string str = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_ENABLE, CONFIG_STORAGE_S3_ENABLE_DEFAULT);
//    STATUS_CHECK(CheckStorageConfigS3Enable(str));
//    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, value));
//    return Status::OK();
// }
//
// Status
// Config::GetStorageConfigS3Address(std::string& value) {
//    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_ADDRESS, CONFIG_STORAGE_S3_ADDRESS_DEFAULT);
//    return CheckStorageConfigS3Address(value);
// }
//
// Status
// Config::GetStorageConfigS3Port(std::string& value) {
//    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_PORT, CONFIG_STORAGE_S3_PORT_DEFAULT);
//    return CheckStorageConfigS3Port(value);
// }
//
// Status
// Config::GetStorageConfigS3AccessKey(std::string& value) {
//    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_ACCESS_KEY, CONFIG_STORAGE_S3_ACCESS_KEY_DEFAULT);
//    return Status::OK();
// }
//
// Status
// Config::GetStorageConfigS3SecretKey(std::string& value) {
//    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_SECRET_KEY, CONFIG_STORAGE_S3_SECRET_KEY_DEFAULT);
//    return Status::OK();
// }
//
// Status
// Config::GetStorageConfigS3Bucket(std::string& value) {
//    value = GetConfigStr(CONFIG_STORAGE, CONFIG_STORAGE_S3_BUCKET, CONFIG_STORAGE_S3_BUCKET_DEFAULT);
//    return Status::OK();
// }

/* metric config */
Status
Config::GetMetricConfigEnableMonitor(bool& value) {
    std::string str = GetConfigStr(CONFIG_METRIC, CONFIG_METRIC_ENABLE_MONITOR, CONFIG_METRIC_ENABLE_MONITOR_DEFAULT);
    STATUS_CHECK(CheckMetricConfigEnableMonitor(str));
    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, value));
    return Status::OK();
}

Status
Config::GetMetricConfigAddress(std::string& value) {
    value = GetConfigStr(CONFIG_METRIC, CONFIG_METRIC_ADDRESS, CONFIG_METRIC_ADDRESS_DEFAULT);
    return Status::OK();
}

Status
Config::GetMetricConfigPort(std::string& value) {
    value = GetConfigStr(CONFIG_METRIC, CONFIG_METRIC_PORT, CONFIG_METRIC_PORT_DEFAULT);
    return CheckMetricConfigPort(value);
}

/* cache config */
Status
Config::GetCacheConfigCpuCacheCapacity(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_CAPACITY, CONFIG_CACHE_CPU_CACHE_CAPACITY_DEFAULT);
    STATUS_CHECK(CheckCacheConfigCpuCacheCapacity(str));
    std::string err;
    value = parse_bytes(str, err);
    // value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetCacheConfigCpuCacheThreshold(float& value) {
    std::string str =
        GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_THRESHOLD, CONFIG_CACHE_CPU_CACHE_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckCacheConfigCpuCacheThreshold(str));
    value = std::stof(str);
    return Status::OK();
}

Status
Config::GetCacheConfigInsertBufferSize(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_INSERT_BUFFER_SIZE, CONFIG_CACHE_INSERT_BUFFER_SIZE_DEFAULT);
    STATUS_CHECK(CheckCacheConfigInsertBufferSize(str));
    std::string err;
    value = parse_bytes(str, err);
    // value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetCacheConfigCacheInsertData(bool& value) {
    std::string str =
        GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_CACHE_INSERT_DATA, CONFIG_CACHE_CACHE_INSERT_DATA_DEFAULT);
    STATUS_CHECK(CheckCacheConfigCacheInsertData(str));
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    value = (str == "true" || str == "on" || str == "yes" || str == "1");
    return Status::OK();
}

Status
Config::GetCacheConfigPreloadCollection(std::string& value) {
    value = GetConfigStr(CONFIG_CACHE, CONFIG_CACHE_PRELOAD_COLLECTION);
    return Status::OK();
}

/* engine config */
Status
Config::GetEngineConfigUseBlasThreshold(int64_t& value) {
    std::string str =
        GetConfigStr(CONFIG_ENGINE, CONFIG_ENGINE_USE_BLAS_THRESHOLD, CONFIG_ENGINE_USE_BLAS_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckEngineConfigUseBlasThreshold(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetEngineConfigOmpThreadNum(int64_t& value) {
    std::string str = GetConfigStr(CONFIG_ENGINE, CONFIG_ENGINE_OMP_THREAD_NUM, CONFIG_ENGINE_OMP_THREAD_NUM_DEFAULT);
    STATUS_CHECK(CheckEngineConfigOmpThreadNum(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetEngineConfigSimdType(std::string& value) {
    value = GetConfigStr(CONFIG_ENGINE, CONFIG_ENGINE_SIMD_TYPE, CONFIG_ENGINE_SIMD_TYPE_DEFAULT);
    return CheckEngineConfigSimdType(value);
}

/* gpu resource config */
#ifdef MILVUS_GPU_VERSION
Status
Config::GetGpuResourceConfigEnable(bool& value) {
    std::string str = GetConfigStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_ENABLE, CONFIG_GPU_RESOURCE_ENABLE_DEFAULT);
    STATUS_CHECK(CheckGpuResourceConfigEnable(str));
    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, value));
    return Status::OK();
}

Status
Config::GetGpuResourceConfigCacheCapacity(int64_t& value) {
    bool gpu_resource_enable = false;
    STATUS_CHECK(GetGpuResourceConfigEnable(gpu_resource_enable));
    fiu_do_on("Config.GetGpuResourceConfigCacheCapacity.diable_gpu_resource", gpu_resource_enable = false);
    if (!gpu_resource_enable) {
        std::string msg = "GPU not supported. Possible reason: gpu.enable is set to false.";
        return Status(SERVER_UNSUPPORTED_ERROR, msg);
    }
    std::string str = GetConfigStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_CACHE_CAPACITY,
                                   CONFIG_GPU_RESOURCE_CACHE_CAPACITY_DEFAULT);
    STATUS_CHECK(CheckGpuResourceConfigCacheCapacity(str));
    std::string err;
    value = parse_bytes(str, err);
    // value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetGpuResourceConfigCacheThreshold(float& value) {
    bool gpu_resource_enable = false;
    STATUS_CHECK(GetGpuResourceConfigEnable(gpu_resource_enable));
    fiu_do_on("Config.GetGpuResourceConfigCacheThreshold.diable_gpu_resource", gpu_resource_enable = false);
    if (!gpu_resource_enable) {
        std::string msg = "GPU not supported. Possible reason: gpu.enable is set to false.";
        return Status(SERVER_UNSUPPORTED_ERROR, msg);
    }
    std::string str = GetConfigStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_CACHE_THRESHOLD,
                                   CONFIG_GPU_RESOURCE_CACHE_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckGpuResourceConfigCacheThreshold(str));
    value = std::stof(str);
    return Status::OK();
}

Status
Config::GetGpuResourceConfigGpuSearchThreshold(int64_t& value) {
    std::string str = GetConfigStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD,
                                   CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD_DEFAULT);
    STATUS_CHECK(CheckGpuResourceConfigGpuSearchThreshold(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetGpuResourceConfigSearchResources(std::vector<int64_t>& value) {
    bool gpu_resource_enable = false;
    STATUS_CHECK(GetGpuResourceConfigEnable(gpu_resource_enable));
    fiu_do_on("get_gpu_config_search_resources.disable_gpu_resource_fail", gpu_resource_enable = false);
    if (!gpu_resource_enable) {
        std::string msg = "GPU not supported. Possible reason: gpu.enable is set to false.";
        return Status(SERVER_UNSUPPORTED_ERROR, msg);
    }
    std::string str = GetConfigSequenceStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_SEARCH_RESOURCES,
                                           CONFIG_GPU_RESOURCE_DELIMITER, CONFIG_GPU_RESOURCE_SEARCH_RESOURCES_DEFAULT);
    std::vector<std::string> res_vec;
    StringHelpFunctions::SplitStringByDelimeter(str, CONFIG_GPU_RESOURCE_DELIMITER, res_vec);
    STATUS_CHECK(CheckGpuResourceConfigSearchResources(res_vec));
    value.clear();
    for (std::string& res : res_vec) {
        value.push_back(std::stoll(res.substr(3)));
    }
    return Status::OK();
}

Status
Config::GetGpuResourceConfigBuildIndexResources(std::vector<int64_t>& value) {
    bool gpu_resource_enable = false;
    STATUS_CHECK(GetGpuResourceConfigEnable(gpu_resource_enable));
    fiu_do_on("get_gpu_config_build_index_resources.disable_gpu_resource_fail", gpu_resource_enable = false);
    if (!gpu_resource_enable) {
        std::string msg = "GPU not supported. Possible reason: gpu.enable is set to false.";
        return Status(SERVER_UNSUPPORTED_ERROR, msg);
    }
    std::string str =
        GetConfigSequenceStr(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES,
                             CONFIG_GPU_RESOURCE_DELIMITER, CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES_DEFAULT);
    std::vector<std::string> res_vec;
    StringHelpFunctions::SplitStringByDelimeter(str, CONFIG_GPU_RESOURCE_DELIMITER, res_vec);
    STATUS_CHECK(CheckGpuResourceConfigBuildIndexResources(res_vec));
    value.clear();
    for (std::string& res : res_vec) {
        value.push_back(std::stoll(res.substr(3)));
    }
    return Status::OK();
}
#endif

/* tracing config */
Status
Config::GetTracingConfigJsonConfigPath(std::string& value) {
    value = GetConfigStr(CONFIG_TRACING, CONFIG_TRACING_JSON_CONFIG_PATH, "");
    fiu_do_on("get_config_json_config_path_fail", value = "error_config_json_path");
    if (!value.empty()) {
        std::ifstream tracer_config(value);
        Status s = tracer_config.good() ? Status::OK()
                                        : Status(SERVER_INVALID_ARGUMENT, "Failed to open tracer config file " + value +
                                                                              ": " + std::strerror(errno));
        tracer_config.close();
        return s;
    }
    return Status::OK();
}

/* wal config */
Status
Config::GetWalConfigEnable(bool& wal_enable) {
    std::string str = GetConfigStr(CONFIG_WAL, CONFIG_WAL_ENABLE, CONFIG_WAL_ENABLE_DEFAULT);
    STATUS_CHECK(CheckWalConfigEnable(str));
    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, wal_enable));
    return Status::OK();
}

Status
Config::GetWalConfigRecoveryErrorIgnore(bool& recovery_error_ignore) {
    std::string str =
        GetConfigStr(CONFIG_WAL, CONFIG_WAL_RECOVERY_ERROR_IGNORE, CONFIG_WAL_RECOVERY_ERROR_IGNORE_DEFAULT);
    STATUS_CHECK(CheckWalConfigRecoveryErrorIgnore(str));
    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, recovery_error_ignore));
    return Status::OK();
}

Status
Config::GetWalConfigBufferSize(int64_t& buffer_size) {
    std::string str = GetConfigStr(CONFIG_WAL, CONFIG_WAL_BUFFER_SIZE, CONFIG_WAL_BUFFER_SIZE_DEFAULT);
    STATUS_CHECK(CheckWalConfigBufferSize(str));
    std::string err;
    buffer_size = parse_bytes(str, err);
    // buffer_size = std::stoll(str);
    if (buffer_size > CONFIG_WAL_BUFFER_SIZE_MAX) {
        buffer_size = CONFIG_WAL_BUFFER_SIZE_MAX;
    } else if (buffer_size < CONFIG_WAL_BUFFER_SIZE_MIN) {
        buffer_size = CONFIG_WAL_BUFFER_SIZE_MIN;
    }
    return Status::OK();
}

Status
Config::GetWalConfigWalPath(std::string& wal_path) {
    wal_path = GetConfigStr(CONFIG_WAL, CONFIG_WAL_WAL_PATH, CONFIG_WAL_WAL_PATH_DEFAULT);
    STATUS_CHECK(CheckWalConfigWalPath(wal_path));
    return Status::OK();
}

/* logs config */
Status
Config::GetLogsLevel(std::string& value) {
    value = GetConfigStr(CONFIG_LOGS, CONFIG_LOGS_LEVEL, CONFIG_LOGS_LEVEL_DEFAULT);
    STATUS_CHECK(CheckLogsLevel(value));
    return Status::OK();
}

Status
Config::GetLogsTraceEnable(bool& value) {
    std::string str = GetConfigStr(CONFIG_LOGS, CONFIG_LOGS_TRACE_ENABLE, CONFIG_LOGS_TRACE_ENABLE_DEFAULT);
    STATUS_CHECK(CheckLogsTraceEnable(str));
    STATUS_CHECK(StringHelpFunctions::ConvertToBoolean(str, value));
    return Status::OK();
}

Status
Config::GetLogsPath(std::string& value) {
    value = GetConfigStr(CONFIG_LOGS, CONFIG_LOGS_PATH, CONFIG_LOGS_PATH_DEFAULT);
    STATUS_CHECK(CheckLogsPath(value));
    return Status::OK();
}

Status
Config::GetLogsMaxLogFileSize(int64_t& value) {
    std::string str = GetConfigStr(CONFIG_LOGS, CONFIG_LOGS_MAX_LOG_FILE_SIZE, CONFIG_LOGS_MAX_LOG_FILE_SIZE_DEFAULT);
    STATUS_CHECK(CheckLogsMaxLogFileSize(str));
    std::string err;
    value = parse_bytes(str, err);
    // value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetLogsLogRotateNum(int64_t& value) {
    std::string str = GetConfigStr(CONFIG_LOGS, CONFIG_LOGS_LOG_ROTATE_NUM, CONFIG_LOGS_LOG_ROTATE_NUM_DEFAULT);
    STATUS_CHECK(CheckLogsLogRotateNum(str));
    value = std::stoll(str);
    return Status::OK();
}

Status
Config::GetServerRestartRequired(bool& required) {
    required = restart_required_;
    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
/* cluster config */
Status
Config::SetClusterConfigEnable(const std::string& value) {
    STATUS_CHECK(CheckClusterConfigEnable(value));
    return SetConfigValueInMem(CONFIG_CLUSTER, CONFIG_CLUSTER_ENABLE, value);
}

Status
Config::SetClusterConfigRole(const std::string& value) {
    STATUS_CHECK(CheckClusterConfigRole(value));
    return SetConfigValueInMem(CONFIG_CLUSTER, CONFIG_CLUSTER_ROLE, value);
}

/* general config */
Status
Config::SetGeneralConfigTimezone(const std::string& value) {
    STATUS_CHECK(CheckGeneralConfigTimezone(value));
    return SetConfigValueInMem(CONFIG_GENERAL, CONFIG_GENERAL_TIMEZONE, value);
}

Status
Config::SetGeneralConfigMetaURI(const std::string& value) {
    STATUS_CHECK(CheckGeneralConfigMetaURI(value));
    return SetConfigValueInMem(CONFIG_GENERAL, CONFIG_GENERAL_METAURI, value);
}

/* network config */
Status
Config::SetNetworkConfigBindAddress(const std::string& value) {
    STATUS_CHECK(CheckNetworkConfigBindAddress(value));
    return SetConfigValueInMem(CONFIG_NETWORK, CONFIG_NETWORK_BIND_ADDRESS, value);
}

Status
Config::SetNetworkConfigBindPort(const std::string& value) {
    STATUS_CHECK(CheckNetworkConfigBindPort(value));
    return SetConfigValueInMem(CONFIG_NETWORK, CONFIG_NETWORK_BIND_PORT, value);
}

Status
Config::SetNetworkConfigHTTPEnable(const std::string& value) {
    STATUS_CHECK(CheckNetworkConfigHTTPEnable(value));
    return SetConfigValueInMem(CONFIG_NETWORK, CONFIG_NETWORK_HTTP_ENABLE, value);
}

Status
Config::SetNetworkConfigHTTPPort(const std::string& value) {
    STATUS_CHECK(CheckNetworkConfigHTTPPort(value));
    return SetConfigValueInMem(CONFIG_NETWORK, CONFIG_NETWORK_HTTP_PORT, value);
}

/* db config */
Status
Config::SetDBConfigArchiveDiskThreshold(const std::string& value) {
    STATUS_CHECK(CheckDBConfigArchiveDiskThreshold(value));
    return SetConfigValueInMem(CONFIG_DB, CONFIG_DB_ARCHIVE_DISK_THRESHOLD, value);
}

Status
Config::SetDBConfigArchiveDaysThreshold(const std::string& value) {
    STATUS_CHECK(CheckDBConfigArchiveDaysThreshold(value));
    return SetConfigValueInMem(CONFIG_DB, CONFIG_DB_ARCHIVE_DAYS_THRESHOLD, value);
}

/* storage config */
Status
Config::SetStorageConfigPath(const std::string& value) {
    STATUS_CHECK(CheckStorageConfigPath(value));
    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_PATH, value);
}

Status
Config::SetStorageConfigAutoFlushInterval(const std::string& value) {
    STATUS_CHECK(CheckStorageConfigAutoFlushInterval(value));
    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_AUTO_FLUSH_INTERVAL, value);
}

Status
Config::SetStorageConfigFileCleanupTimeout(const std::string& value) {
    STATUS_CHECK(CheckStorageConfigFileCleanupTimeout(value));
    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_FILE_CLEANUP_TIMEOUT, value);
}

// Status
// Config::SetStorageConfigS3Enable(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3Enable(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_ENABLE, value);
// }
//
// Status
// Config::SetStorageConfigS3Address(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3Address(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_ADDRESS, value);
// }
//
// Status
// Config::SetStorageConfigS3Port(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3Port(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_PORT, value);
// }
//
// Status
// Config::SetStorageConfigS3AccessKey(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3AccessKey(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_ACCESS_KEY, value);
// }
//
// Status
// Config::SetStorageConfigS3SecretKey(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3SecretKey(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_SECRET_KEY, value);
// }
//
// Status
// Config::SetStorageConfigS3Bucket(const std::string& value) {
//    STATUS_CHECK(CheckStorageConfigS3Bucket(value));
//    return SetConfigValueInMem(CONFIG_STORAGE, CONFIG_STORAGE_S3_BUCKET, value);
// }

/* metric config */
Status
Config::SetMetricConfigEnableMonitor(const std::string& value) {
    STATUS_CHECK(CheckMetricConfigEnableMonitor(value));
    return SetConfigValueInMem(CONFIG_METRIC, CONFIG_METRIC_ENABLE_MONITOR, value);
}

Status
Config::SetMetricConfigAddress(const std::string& value) {
    STATUS_CHECK(CheckMetricConfigAddress(value));
    return SetConfigValueInMem(CONFIG_METRIC, CONFIG_METRIC_ADDRESS, value);
}

Status
Config::SetMetricConfigPort(const std::string& value) {
    STATUS_CHECK(CheckMetricConfigPort(value));
    return SetConfigValueInMem(CONFIG_METRIC, CONFIG_METRIC_PORT, value);
}

/* cache config */
Status
Config::SetCacheConfigCpuCacheCapacity(const std::string& value) {
    STATUS_CHECK(CheckCacheConfigCpuCacheCapacity(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_CAPACITY, value));
    return ExecCallBacks(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_CAPACITY, value);
}

Status
Config::SetCacheConfigCpuCacheThreshold(const std::string& value) {
    STATUS_CHECK(CheckCacheConfigCpuCacheThreshold(value));
    return SetConfigValueInMem(CONFIG_CACHE, CONFIG_CACHE_CPU_CACHE_THRESHOLD, value);
}

Status
Config::SetCacheConfigInsertBufferSize(const std::string& value) {
    STATUS_CHECK(CheckCacheConfigInsertBufferSize(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_CACHE, CONFIG_CACHE_INSERT_BUFFER_SIZE, value));
    return ExecCallBacks(CONFIG_CACHE, CONFIG_CACHE_INSERT_BUFFER_SIZE, value);
}

Status
Config::SetCacheConfigCacheInsertData(const std::string& value) {
    STATUS_CHECK(CheckCacheConfigCacheInsertData(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_CACHE, CONFIG_CACHE_CACHE_INSERT_DATA, value));
    return ExecCallBacks(CONFIG_CACHE, CONFIG_CACHE_CACHE_INSERT_DATA, value);
}

Status
Config::SetCacheConfigPreloadCollection(const std::string& value) {
    STATUS_CHECK(CheckCacheConfigPreloadCollection(value));
    std::string cor_value = value == "*" ? "\'*\'" : value;
    return SetConfigValueInMem(CONFIG_CACHE, CONFIG_CACHE_PRELOAD_COLLECTION, cor_value);
}

/* engine config */
Status
Config::SetEngineConfigUseBlasThreshold(const std::string& value) {
    STATUS_CHECK(CheckEngineConfigUseBlasThreshold(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_ENGINE, CONFIG_ENGINE_USE_BLAS_THRESHOLD, value));
    return ExecCallBacks(CONFIG_ENGINE, CONFIG_ENGINE_USE_BLAS_THRESHOLD, value);
}

Status
Config::SetEngineConfigOmpThreadNum(const std::string& value) {
    STATUS_CHECK(CheckEngineConfigOmpThreadNum(value));
    return SetConfigValueInMem(CONFIG_ENGINE, CONFIG_ENGINE_OMP_THREAD_NUM, value);
}

Status
Config::SetEngineConfigSimdType(const std::string& value) {
    STATUS_CHECK(CheckEngineConfigSimdType(value));
    return SetConfigValueInMem(CONFIG_ENGINE, CONFIG_ENGINE_SIMD_TYPE, value);
}

/* gpu resource config */
#ifdef MILVUS_GPU_VERSION
Status
Config::SetGpuResourceConfigEnable(const std::string& value) {
    STATUS_CHECK(CheckGpuResourceConfigEnable(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_ENABLE, value));
    return ExecCallBacks(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_ENABLE, value);
}

Status
Config::SetGpuResourceConfigCacheCapacity(const std::string& value) {
    STATUS_CHECK(CheckGpuResourceConfigCacheCapacity(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_CACHE_CAPACITY, value));
    return ExecCallBacks(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_CACHE_CAPACITY, value);
}

Status
Config::SetGpuResourceConfigCacheThreshold(const std::string& value) {
    STATUS_CHECK(CheckGpuResourceConfigCacheThreshold(value));
    return SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_CACHE_THRESHOLD, value);
}

Status
Config::SetGpuResourceConfigGpuSearchThreshold(const std::string& value) {
    STATUS_CHECK(CheckGpuResourceConfigGpuSearchThreshold(value));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD, value));
    return ExecCallBacks(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_GPU_SEARCH_THRESHOLD, value);
}

Status
Config::SetGpuResourceConfigSearchResources(const std::string& value) {
    std::vector<std::string> res_vec;
    StringHelpFunctions::SplitStringByDelimeter(value, CONFIG_GPU_RESOURCE_DELIMITER, res_vec);
    STATUS_CHECK(CheckGpuResourceConfigSearchResources(res_vec));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_SEARCH_RESOURCES, value));
    return ExecCallBacks(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_SEARCH_RESOURCES, value);
}

Status
Config::SetGpuResourceConfigBuildIndexResources(const std::string& value) {
    std::vector<std::string> res_vec;
    StringHelpFunctions::SplitStringByDelimeter(value, CONFIG_GPU_RESOURCE_DELIMITER, res_vec);
    STATUS_CHECK(CheckGpuResourceConfigBuildIndexResources(res_vec));
    STATUS_CHECK(SetConfigValueInMem(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES, value));
    return ExecCallBacks(CONFIG_GPU_RESOURCE, CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES, value);
}
#endif

/* tracing config */
Status
Config::SetTracingConfigJsonConfigPath(const std::string& value) {
    STATUS_CHECK(CheckTracingConfigJsonConfigPath(value));
    return SetConfigValueInMem(CONFIG_TRACING, CONFIG_TRACING_JSON_CONFIG_PATH, value);
}

/* wal config */
Status
Config::SetWalConfigEnable(const std::string& value) {
    STATUS_CHECK(CheckWalConfigEnable(value));
    return SetConfigValueInMem(CONFIG_WAL, CONFIG_WAL_ENABLE, value);
}

Status
Config::SetWalConfigRecoveryErrorIgnore(const std::string& value) {
    STATUS_CHECK(CheckWalConfigRecoveryErrorIgnore(value));
    return SetConfigValueInMem(CONFIG_WAL, CONFIG_WAL_RECOVERY_ERROR_IGNORE, value);
}

Status
Config::SetWalConfigBufferSize(const std::string& value) {
    STATUS_CHECK(CheckWalConfigBufferSize(value));
    return SetConfigValueInMem(CONFIG_WAL, CONFIG_WAL_BUFFER_SIZE, value);
}

Status
Config::SetWalConfigWalPath(const std::string& value) {
    STATUS_CHECK(CheckWalConfigWalPath(value));
    return SetConfigValueInMem(CONFIG_WAL, CONFIG_WAL_WAL_PATH, value);
}

/* logs config */
Status
Config::SetLogsLevel(const std::string& value) {
    STATUS_CHECK(CheckLogsLevel(value));
    return SetConfigValueInMem(CONFIG_LOGS, CONFIG_LOGS_LEVEL, value);
}

Status
Config::SetLogsTraceEnable(const std::string& value) {
    STATUS_CHECK(CheckLogsTraceEnable(value));
    return SetConfigValueInMem(CONFIG_LOGS, CONFIG_LOGS_TRACE_ENABLE, value);
}

Status
Config::SetLogsPath(const std::string& value) {
    STATUS_CHECK(CheckLogsPath(value));
    return SetConfigValueInMem(CONFIG_LOGS, CONFIG_LOGS_PATH, value);
}

Status
Config::SetLogsMaxLogFileSize(const std::string& value) {
    STATUS_CHECK(CheckLogsMaxLogFileSize(value));
    return SetConfigValueInMem(CONFIG_LOGS, CONFIG_LOGS_MAX_LOG_FILE_SIZE, value);
}

Status
Config::SetLogsLogRotateNum(const std::string& value) {
    STATUS_CHECK(CheckLogsLogRotateNum(value));
    return SetConfigValueInMem(CONFIG_LOGS, CONFIG_LOGS_LOG_ROTATE_NUM, value);
}

}  // namespace server
}  // namespace milvus
