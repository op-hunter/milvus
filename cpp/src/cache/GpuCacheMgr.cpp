////////////////////////////////////////////////////////////////////////////////
// Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include "utils/Log.h"
#include "GpuCacheMgr.h"
#include "server/ServerConfig.h"

namespace zilliz {
namespace milvus {
namespace cache {

std::mutex GpuCacheMgr::mutex_;
std::unordered_map<uint64_t, GpuCacheMgrPtr> GpuCacheMgr::instance_;

namespace {
    constexpr int64_t unit = 1024 * 1024 * 1024;

    std::vector<uint64_t> load() {
        server::ConfigNode& config = server::ServerConfig::GetInstance().GetConfig(server::CONFIG_CACHE);
        auto conf_gpu_ids = config.GetSequence(server::CONFIG_GPU_IDS);

        std::vector<uint64_t > gpu_ids;

        for (auto gpu_id : conf_gpu_ids) {
            gpu_ids.push_back(std::atoi(gpu_id.c_str()));
        }

        return gpu_ids;
    }
}


bool GpuCacheMgr::GpuIdInConfig(uint64_t gpu_id) {
    static std::vector<uint64_t > ids = load();
    for (auto id : ids) {
        if (gpu_id == id) return true;
    }
    return false;
}

GpuCacheMgr::GpuCacheMgr() {
    server::ConfigNode& config = server::ServerConfig::GetInstance().GetConfig(server::CONFIG_CACHE);

    int64_t cap = config.GetInt64Value(server::CONFIG_GPU_CACHE_CAPACITY, 2);
    cap *= unit;
    cache_ = std::make_shared<Cache>(cap, 1UL<<32);

    double free_percent = config.GetDoubleValue(server::GPU_CACHE_FREE_PERCENT, 0.85);
    if (free_percent > 0.0 && free_percent <= 1.0) {
        cache_->set_freemem_percent(free_percent);
    } else {
        SERVER_LOG_ERROR << "Invalid gpu_cache_free_percent: " << free_percent <<
                         ", defaultly set to " << cache_->freemem_percent();
    }
}

void GpuCacheMgr::InsertItem(const std::string& key, const DataObjPtr& data) {
    //TODO: copy data to gpu
    if (cache_ == nullptr) {
        SERVER_LOG_ERROR << "Cache doesn't exist";
        return;
    }

    cache_->insert(key, data);
}

}
}
}