////////////////////////////////////////////////////////////////////////////////
// Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <fstream>


class DataGenBase;

using DataGenPtr = std::shared_ptr<DataGenBase>;


class DataGenBase {
 public:
    virtual void GenData(const int &dim, const int &nb, const int &nq, float *xb, float *xq, long *ids,
                         const int &k, long *gt_ids, float *gt_dis);

    virtual void GenData(const int &dim,
                         const int &nb,
                         const int &nq,
                         std::vector<float> &xb,
                         std::vector<float> &xq,
                         std::vector<long> &ids,
                         const int &k,
                         std::vector<long> &gt_ids,
                         std::vector<float> &gt_dis);
};


//class SanityCheck : public DataGenBase {
// public:
//    void GenData(const int &dim, const int &nb, const int &nq, float *xb, float *xq, long *ids,
//                 const int &k, long *gt_ids, float *gt_dis) override;
//};

