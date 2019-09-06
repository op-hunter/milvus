/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/

#include "TaskConvert.h"
#include "scheduler/tasklabel/DefaultLabel.h"
#include "scheduler/tasklabel/BroadcastLabel.h"


namespace zilliz {
namespace milvus {
namespace engine {

TaskPtr
TaskConvert(const ScheduleTaskPtr &schedule_task) {
    switch (schedule_task->type()) {
        case ScheduleTaskType::kIndexLoad: {
            auto load_task = std::static_pointer_cast<IndexLoadTask>(schedule_task);
            auto task = std::make_shared<XSearchTask>(load_task->file_);
            task->label() = std::make_shared<DefaultLabel>();
            task->search_contexts_ = load_task->search_contexts_;
            return task;
        }
        case ScheduleTaskType::kDelete: {
            auto delete_task = std::static_pointer_cast<DeleteTask>(schedule_task);
            auto task = std::make_shared<XDeleteTask>(delete_task->context_);
            task->label() = std::make_shared<BroadcastLabel>();
            return task;
        }
        default: {
            // TODO: unexpected !!!
            return nullptr;
        }
    }
}

}
}
}
