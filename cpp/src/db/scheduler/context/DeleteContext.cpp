/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/

#include "DeleteContext.h"


namespace zilliz {
namespace milvus {
namespace engine {

DeleteContext::DeleteContext(const std::string &table_id, meta::MetaPtr &meta_ptr, uint64_t num_resource)
    : IScheduleContext(ScheduleContextType::kDelete),
      table_id_(table_id),
      meta_ptr_(meta_ptr),
      num_resource_(num_resource) {

}

void DeleteContext::WaitAndDelete() {
#ifdef NEW_SCHEDULER
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return done_resource == num_resource_; });
    meta_ptr_->DeleteTableFiles(table_id_);
#endif
}

void DeleteContext::ResourceDone() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++done_resource;
    }
    cv_.notify_one();
}

}
}
}