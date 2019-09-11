#pragma once

#include "db/Status.h"
#include "db/Types.h"

#include <set>
#include <memory>

namespace zilliz {
namespace milvus {
namespace engine {

class MemManager {
 public:

    virtual Status InsertVectors(const std::string &table_id,
                                 size_t n, const float *vectors, IDNumbers &vector_ids) = 0;

    virtual Status Serialize(std::set<std::string> &table_ids) = 0;

    virtual Status EraseMemVector(const std::string &table_id) = 0;

    virtual size_t GetCurrentMutableMem() = 0;

    virtual size_t GetCurrentImmutableMem() = 0;

    virtual size_t GetCurrentMem() = 0;

}; // MemManagerAbstract

using MemManagerPtr = std::shared_ptr<MemManager>;

} // namespace engine
} // namespace milvus
} // namespace zilliz