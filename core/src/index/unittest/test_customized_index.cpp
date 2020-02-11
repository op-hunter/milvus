// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <fiu-control.h>
#include <fiu-local.h>
#include <gtest/gtest.h>
#include <thread>
#include "knowhere/index/vector_index/helpers/Cloner.h"

#include "unittest/Helper.h"
#include "unittest/utils.h"

#include "knowhere/common/Timer.h"

class SingleIndexTest : public DataGen, public TestGpuIndexBase {
 protected:
    void
    SetUp() override {
        TestGpuIndexBase::SetUp();
        nb = 1000000;
        nq = 1000;
        dim = DIM;
        Generate(dim, nb, nq);
        k = 1000;
    }

    void
    TearDown() override {
        TestGpuIndexBase::TearDown();
    }

 protected:
    std::string index_type;
    knowhere::IVFIndexPtr index_ = nullptr;
};

#ifdef CUSTOMIZATION
TEST_F(SingleIndexTest, IVFSQHybrid) {
    assert(!xb.empty());

    index_type = "IVFSQHybrid";
    index_ = IndexFactory(index_type);
    auto conf = ParamGenerator::GetInstance().Gen(ParameterType::ivfsq);
    auto preprocessor = index_->BuildPreprocessor(base_dataset, conf);
    index_->set_preprocessor(preprocessor);

    fiu_init(0);
    auto model = index_->Train(base_dataset, conf);
    index_->set_index_model(model);
    index_->Add(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dimension(), dim);

    auto binaryset = index_->Serialize();
    {
        // copy cpu to gpu
        auto cpu_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
        cpu_idx->Load(binaryset);

        {
            for (int i = 0; i < 3; ++i) {
                auto gpu_idx = cpu_idx->CopyCpuToGpu(DEVICEID, conf);
                auto result = gpu_idx->Search(query_dataset, conf);
                AssertAnns(result, nq, conf->k);
                // PrintResult(result, nq, k);
            }
        }
    }

    {
        // quantization already in gpu, only copy data
        auto cpu_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
        cpu_idx->Load(binaryset);

        ASSERT_ANY_THROW(cpu_idx->CopyCpuToGpuWithQuantizer(-1, conf));
        auto pair = cpu_idx->CopyCpuToGpuWithQuantizer(DEVICEID, conf);
        auto gpu_idx = pair.first;
        auto quantization = pair.second;

        auto result = gpu_idx->Search(query_dataset, conf);
        AssertAnns(result, nq, conf->k);
        //        PrintResult(result, nq, k);

        auto quantizer_conf = std::make_shared<knowhere::QuantizerCfg>();
        quantizer_conf->mode = 2;  // only copy data
        quantizer_conf->gpu_id = DEVICEID;
        for (int i = 0; i < 2; ++i) {
            auto hybrid_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
            hybrid_idx->Load(binaryset);
            auto new_idx = hybrid_idx->LoadData(quantization, quantizer_conf);
            auto result = new_idx->Search(query_dataset, conf);
            AssertAnns(result, nq, conf->k);
            //            PrintResult(result, nq, k);
        }

        {
            // invalid quantizer config
            quantizer_conf = std::make_shared<knowhere::QuantizerCfg>();
            auto hybrid_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
            ASSERT_ANY_THROW(hybrid_idx->LoadData(quantization, nullptr));
            ASSERT_ANY_THROW(hybrid_idx->LoadData(quantization, quantizer_conf));
            quantizer_conf->mode = 2;  // only copy data
            ASSERT_ANY_THROW(hybrid_idx->LoadData(quantization, quantizer_conf));
        }
    }

    {
        // quantization already in gpu, only set quantization
        auto cpu_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
        cpu_idx->Load(binaryset);

        auto pair = cpu_idx->CopyCpuToGpuWithQuantizer(DEVICEID, conf);
        auto quantization = pair.second;

        for (int i = 0; i < 2; ++i) {
            auto hybrid_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
            hybrid_idx->Load(binaryset);

            hybrid_idx->SetQuantizer(quantization);
            auto result = hybrid_idx->Search(query_dataset, conf);
            AssertAnns(result, nq, conf->k);
            //            PrintResult(result, nq, k);
            hybrid_idx->UnsetQuantizer();
        }
    }
}

// TEST_F(SingleIndexTest, thread_safe) {
//    assert(!xb.empty());
//
//    index_type = "IVFSQHybrid";
//    index_ = IndexFactory(index_type);
//    auto base = ParamGenerator::GetInstance().Gen(ParameterType::ivfsq);
//    auto conf = std::dynamic_pointer_cast<knowhere::IVFSQCfg>(base);
//    conf->nlist = 16384;
//    conf->k = k;
//    conf->nprobe = 10;
//    conf->d = dim;
//    auto preprocessor = index_->BuildPreprocessor(base_dataset, conf);
//    index_->set_preprocessor(preprocessor);
//
//    auto model = index_->Train(base_dataset, conf);
//    index_->set_index_model(model);
//    index_->Add(base_dataset, conf);
//    EXPECT_EQ(index_->Count(), nb);
//    EXPECT_EQ(index_->Dimension(), dim);
//
//    auto binaryset = index_->Serialize();
//
//
//
//    auto cpu_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
//    cpu_idx->Load(binaryset);
//    auto pair = cpu_idx->CopyCpuToGpuWithQuantizer(DEVICEID, conf);
//    auto quantizer = pair.second;
//
//    auto quantizer_conf = std::make_shared<knowhere::QuantizerCfg>();
//    quantizer_conf->mode = 2;  // only copy data
//    quantizer_conf->gpu_id = DEVICEID;
//
//    auto CopyAllToGpu = [&](int64_t search_count, bool do_search = false) {
//        for (int i = 0; i < search_count; ++i) {
//            auto gpu_idx = cpu_idx->CopyCpuToGpu(DEVICEID, conf);
//            if (do_search) {
//                auto result = gpu_idx->Search(query_dataset, conf);
//                AssertAnns(result, nq, conf->k);
//            }
//        }
//    };
//
//    auto hybrid_qt_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
//    hybrid_qt_idx->Load(binaryset);
//    auto SetQuantizerDoSearch = [&](int64_t search_count) {
//        for (int i = 0; i < search_count; ++i) {
//            hybrid_qt_idx->SetQuantizer(quantizer);
//            auto result = hybrid_qt_idx->Search(query_dataset, conf);
//            AssertAnns(result, nq, conf->k);
//            //            PrintResult(result, nq, k);
//            hybrid_qt_idx->UnsetQuantizer();
//        }
//    };
//
//    auto hybrid_data_idx = std::make_shared<knowhere::IVFSQHybrid>(DEVICEID);
//    hybrid_data_idx->Load(binaryset);
//    auto LoadDataDoSearch = [&](int64_t search_count, bool do_search = false) {
//        for (int i = 0; i < search_count; ++i) {
//            auto hybrid_idx = hybrid_data_idx->LoadData(quantizer, quantizer_conf);
//            if (do_search) {
//                auto result = hybrid_idx->Search(query_dataset, conf);
////                AssertAnns(result, nq, conf->k);
//            }
//        }
//    };
//
//    knowhere::TimeRecorder tc("");
//    CopyAllToGpu(2000/2, false);
//    tc.RecordSection("CopyAllToGpu witout search");
//    CopyAllToGpu(400/2, true);
//    tc.RecordSection("CopyAllToGpu with search");
//    SetQuantizerDoSearch(6);
//    tc.RecordSection("SetQuantizer with search");
//    LoadDataDoSearch(2000/2, false);
//    tc.RecordSection("LoadData without search");
//    LoadDataDoSearch(400/2, true);
//    tc.RecordSection("LoadData with search");
//
//    {
//        std::thread t1(CopyAllToGpu, 2000, false);
//        std::thread t2(CopyAllToGpu, 400, true);
//        t1.join();
//        t2.join();
//    }
//
//    {
//        std::thread t1(SetQuantizerDoSearch, 12);
//        std::thread t2(CopyAllToGpu, 400, true);
//        t1.join();
//        t2.join();
//    }
//
//    {
//        std::thread t1(SetQuantizerDoSearch, 12);
//        std::thread t2(LoadDataDoSearch, 400, true);
//        t1.join();
//        t2.join();
//    }
//
//    {
//        std::thread t1(LoadDataDoSearch, 2000, false);
//        std::thread t2(LoadDataDoSearch, 400, true);
//        t1.join();
//        t2.join();
//    }
//
//}

#endif
