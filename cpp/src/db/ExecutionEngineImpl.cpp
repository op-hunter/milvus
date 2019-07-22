/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/
#include <stdexcept>

#include <src/server/ServerConfig.h>
#include <src/metrics/Metrics.h>
#include "Log.h"

#include "src/cache/CpuCacheMgr.h"
#include "ExecutionEngineImpl.h"
#include "wrapper/knowhere/vec_index.h"
#include "wrapper/knowhere/vec_impl.h"
#include "knowhere/common/exception.h"
#include "Exception.h"


namespace zilliz {
namespace milvus {
namespace engine {

ExecutionEngineImpl::ExecutionEngineImpl(uint16_t dimension,
                                         const std::string &location,
                                         EngineType type)
    : location_(location), dim(dimension), build_type(type) {
    current_type = EngineType::FAISS_IDMAP;

    index_ = CreatetVecIndex(EngineType::FAISS_IDMAP);
    if (!index_) throw Exception("Create Empty VecIndex");

    auto ec = std::static_pointer_cast<BFIndex>(index_)->Build(dimension);
    if (ec != server::KNOWHERE_SUCCESS) { throw Exception("Build index error"); }
}

ExecutionEngineImpl::ExecutionEngineImpl(VecIndexPtr index,
                                         const std::string &location,
                                         EngineType type)
    : index_(std::move(index)), location_(location), build_type(type) {
    current_type = type;
}

VecIndexPtr ExecutionEngineImpl::CreatetVecIndex(EngineType type) {
    std::shared_ptr<VecIndex> index;
    switch (type) {
        case EngineType::FAISS_IDMAP: {
            index = GetVecIndexFactory(IndexType::FAISS_IDMAP);
            break;
        }
        case EngineType::FAISS_IVFFLAT_GPU: {
            index = GetVecIndexFactory(IndexType::FAISS_IVFFLAT_MIX);
            break;
        }
        case EngineType::FAISS_IVFFLAT_CPU: {
            index = GetVecIndexFactory(IndexType::FAISS_IVFFLAT_CPU);
            break;
        }
        case EngineType::SPTAG_KDT_RNT_CPU: {
            index = GetVecIndexFactory(IndexType::SPTAG_KDT_RNT_CPU);
            break;
        }
        default: {
            ENGINE_LOG_ERROR << "Invalid engine type";
            return nullptr;
        }
    }
    return index;
}

Status ExecutionEngineImpl::AddWithIds(long n, const float *xdata, const long *xids) {
    auto ec = index_->Add(n, xdata, xids);
    if (ec != server::KNOWHERE_SUCCESS) {
        return Status::Error("Add error");
    }
    return Status::OK();
}

size_t ExecutionEngineImpl::Count() const {
    return index_->Count();
}

size_t ExecutionEngineImpl::Size() const {
    return (size_t) (Count() * Dimension()) * sizeof(float);
}

size_t ExecutionEngineImpl::Dimension() const {
    return index_->Dimension();
}

size_t ExecutionEngineImpl::PhysicalSize() const {
    return (size_t) (Count() * Dimension()) * sizeof(float);
}

Status ExecutionEngineImpl::Serialize() {
    auto ec = write_index(index_, location_);
    if (ec != server::KNOWHERE_SUCCESS) {
        return Status::Error("Serialize: write to disk error");
    }
    return Status::OK();
}

Status ExecutionEngineImpl::Load() {
    index_ = zilliz::milvus::cache::CpuCacheMgr::GetInstance()->GetIndex(location_);
    bool to_cache = false;
    auto start_time = METRICS_NOW_TIME;
    if (!index_) {
        try {
            index_ = read_index(location_);
            to_cache = true;
            ENGINE_LOG_DEBUG << "Disk io from: " << location_;
        } catch (knowhere::KnowhereException &e) {
            ENGINE_LOG_ERROR << e.what();
            return Status::Error(e.what());
        } catch (std::exception &e) {
            return Status::Error(e.what());
        }
    }

    if (to_cache) {
        Cache();
        auto end_time = METRICS_NOW_TIME;
        auto total_time = METRICS_MICROSECONDS(start_time, end_time);

        server::Metrics::GetInstance().FaissDiskLoadDurationSecondsHistogramObserve(total_time);
        double total_size = Size();

        server::Metrics::GetInstance().FaissDiskLoadSizeBytesHistogramObserve(total_size);
        server::Metrics::GetInstance().FaissDiskLoadIOSpeedGaugeSet(total_size / double(total_time));
    }
    return Status::OK();
}

Status ExecutionEngineImpl::Merge(const std::string &location) {
    if (location == location_) {
        return Status::Error("Cannot Merge Self");
    }
    ENGINE_LOG_DEBUG << "Merge index file: " << location << " to: " << location_;

    auto to_merge = zilliz::milvus::cache::CpuCacheMgr::GetInstance()->GetIndex(location);
    if (!to_merge) {
        try {
            to_merge = read_index(location);
        } catch (knowhere::KnowhereException &e) {
            ENGINE_LOG_ERROR << e.what();
            return Status::Error(e.what());
        } catch (std::exception &e) {
            return Status::Error(e.what());
        }
    }

    if (auto file_index = std::dynamic_pointer_cast<BFIndex>(to_merge)) {
        auto ec = index_->Add(file_index->Count(), file_index->GetRawVectors(), file_index->GetRawIds());
        if (ec != server::KNOWHERE_SUCCESS) {
            ENGINE_LOG_ERROR << "Merge: Add Error";
            return Status::Error("Merge: Add Error");
        }
        return Status::OK();
    } else {
        return Status::Error("file index type is not idmap");
    }
}

ExecutionEnginePtr
ExecutionEngineImpl::BuildIndex(const std::string &location) {
    ENGINE_LOG_DEBUG << "Build index file: " << location << " from: " << location_;

    auto from_index = std::dynamic_pointer_cast<BFIndex>(index_);
    auto to_index = CreatetVecIndex(build_type);
    if (!to_index) {
        throw Exception("Create Empty VecIndex");
    }

    Config build_cfg;
    build_cfg["dim"] = Dimension();
    build_cfg["gpu_id"] = gpu_num;
    AutoGenParams(to_index->GetType(), Count(), build_cfg);

    auto ec = to_index->BuildAll(Count(),
                                 from_index->GetRawVectors(),
                                 from_index->GetRawIds(),
                                 build_cfg);
    if (ec != server::KNOWHERE_SUCCESS) { throw Exception("Build index error"); }

    return std::make_shared<ExecutionEngineImpl>(to_index, location, build_type);
}

Status ExecutionEngineImpl::Search(long n,
                                   const float *data,
                                   long k,
                                   float *distances,
                                   long *labels) const {
    ENGINE_LOG_DEBUG << "Search Params: [k]  " << k << " [nprobe] " << nprobe_;
    auto ec = index_->Search(n, data, distances, labels, Config::object{{"k", k}, {"nprobe", nprobe_}});
    if (ec != server::KNOWHERE_SUCCESS) {
        ENGINE_LOG_ERROR << "Search error";
        return Status::Error("Search: Search Error");
    }
    return Status::OK();
}

Status ExecutionEngineImpl::Cache() {
    zilliz::milvus::cache::CpuCacheMgr::GetInstance()->InsertItem(location_, index_);

    return Status::OK();
}

Status ExecutionEngineImpl::Init() {
    using namespace zilliz::milvus::server;
    ServerConfig &config = ServerConfig::GetInstance();
    ConfigNode server_config = config.GetConfig(CONFIG_SERVER);
    gpu_num = server_config.GetInt32Value("gpu_index", 0);

    switch (build_type) {
        case EngineType::FAISS_IVFFLAT_GPU: {
        }
        case EngineType::FAISS_IVFFLAT_CPU: {
            ConfigNode engine_config = config.GetConfig(CONFIG_ENGINE);
            nprobe_ = engine_config.GetInt32Value(CONFIG_NPROBE, 1);
            break;
        }
    }

    return Status::OK();
}


} // namespace engine
} // namespace milvus
} // namespace zilliz
