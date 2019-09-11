#pragma once

#include "db/Status.h"
#include "db/meta/Meta.h"
#include "db/IDGenerator.h"
#include "db/engine/ExecutionEngine.h"


namespace zilliz {
namespace milvus {
namespace engine {

class VectorSource {
 public:
    VectorSource(const size_t &n, const float *vectors);

    Status Add(const ExecutionEnginePtr &execution_engine,
               const meta::TableFileSchema &table_file_schema,
               const size_t &num_vectors_to_add,
               size_t &num_vectors_added,
               IDNumbers &vector_ids);

    size_t GetNumVectorsAdded();

    bool AllAdded();

    IDNumbers GetVectorIds();

 private:

    const size_t n_;
    const float *vectors_;
    IDNumbers vector_ids_;

    size_t current_num_vectors_added;

    std::shared_ptr<IDGenerator> id_generator_;

}; //VectorSource

using VectorSourcePtr = std::shared_ptr<VectorSource>;

} // namespace engine
} // namespace milvus
} // namespace zilliz