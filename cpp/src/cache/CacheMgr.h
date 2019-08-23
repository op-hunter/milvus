////////////////////////////////////////////////////////////////////////////////
// Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Cache.h"

namespace zilliz {
namespace milvus {
namespace cache {

class CacheMgr {
public:
    virtual uint64_t ItemCount() const;

    virtual bool ItemExists(const std::string& key);

    virtual DataObjPtr GetItem(const std::string& key);
    virtual engine::VecIndexPtr GetIndex(const std::string& key);

    virtual void InsertItem(const std::string& key, const DataObjPtr& data);
    virtual void InsertItem(const std::string& key, const engine::VecIndexPtr& index);

    virtual void EraseItem(const std::string& key);

    virtual void PrintInfo();

    virtual void ClearCache();

    int64_t CacheUsage() const;
    int64_t CacheCapacity() const;
    void SetCapacity(int64_t capacity);
    std::vector<uint64_t > GpuIds() const;
    void SetGpuIds(std::vector<uint64_t> gpu_ids);

protected:
    CacheMgr();
    virtual ~CacheMgr();

protected:
    CachePtr cache_;
};


}
}
}
