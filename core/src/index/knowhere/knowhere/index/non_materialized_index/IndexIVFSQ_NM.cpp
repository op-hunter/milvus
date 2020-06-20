// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <memory>
#include <string>

#ifdef MILVUS_GPU_VERSION
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#endif
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/clone_index.h>
#include <faiss/index_factory.h>

#include "knowhere/common/Exception.h"
#include "knowhere/index/non_materialized_index/IndexIVFSQ_NM.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#ifdef MILVUS_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace milvus {
namespace knowhere {

void
IVFSQ_NM::Load(const BinarySet& binary_set, const void* p_data, size_t nb) {
    std::lock_guard<std::mutex> lk(mutex_);
    LoadImpl(binary_set, index_type_);

    // Construct arranged data from original data
    const float *original_data = (const float *) p_data;
    auto ivfsq_index = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(index_.get());
    auto invlists = ivfsq_index->invlists;
    auto ails = dynamic_cast<faiss::ArrayInvertedLists *> (invlists);
    auto d = ivfsq_index->d;
    auto sq = ivfsq_index->sq;
    auto code_size = ivfsq_index->code_size;
    std::unique_ptr<Quantizer> squant(sq.select_quantizer());
    arranged_data = new uint8_t[code_size * nb];
    prefix_sum.resize(ails->nlist);
    std::vector<float> residual (d);
    std::vector<uint8_t> one_code (code_size);

    size_t curr_index = 0;
    for (int i = 0; i < ails->nlist; i++) {
        auto list_size = ails->ids[i].size();
        for (int j = 0; j < list_size; j++) {
            const float *x_j = original_data + d * (curr_index + j);
            if (ivfsq_index->by_residual) {
                ivfsq_index->quantizer->compute_residual (x_j, residual.data(), i);
                x_j = residual.data();
            }

            memset (one_code.data(), 0, code_size);
            squant->encode_vector (x_j, one_code.data());

            memcpy(arranged_data + code_size * (curr_index + j), one_code.data(), code_size);
        }
        prefix_sum[i] = curr_index;
        curr_index += list_size;
    }
}

void
IVFSQ_NM::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GETTENSOR(dataset_ptr)

    std::stringstream index_type;
    index_type << "IVF" << config[IndexParams::nlist] << ","
               << "SQ" << config[IndexParams::nbits];
    auto build_index =
        faiss::index_factory(dim, index_type.str().c_str(), GetMetricType(config[Metric::TYPE].get<std::string>()));
    build_index->train(rows, (float*)p_data);

    index_.reset(faiss::clone_index(build_index));
}

VecIndexPtr
IVFSQ_NM::CopyCpuToGpu(const int64_t device_id, const Config& config) {
#ifdef MILVUS_GPU_VERSION
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
        ResScope rs(res, device_id, false);

        auto gpu_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), device_id, index_.get());

        std::shared_ptr<faiss::Index> device_index;
        device_index.reset(gpu_index);
        return std::make_shared<GPUIVFSQ>(device_index, device_id, res);
    } else {
        KNOWHERE_THROW_MSG("CopyCpuToGpu Error, can't get gpu_resource");
    }
#else
    KNOWHERE_THROW_MSG("Calling IVFSQ::CopyCpuToGpu when we are using CPU version");
#endif
}

}  // namespace knowhere
}  // namespace milvus
