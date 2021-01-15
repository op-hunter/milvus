// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <string>
#include <cstring>
#include <src/index/thirdparty/faiss/impl/AuxIndexStructures.h>
#include "DynamicResultSet.h"

namespace milvus {
namespace knowhere {

/***********************************************************************
 * RangeSearchResult
 ***********************************************************************/

RangeSearchSet::RangeSearchSet (idx_t nq): nq (nq) {
    labels = nullptr;
    distances = nullptr;
    buffer_size = 1024 * 256;
}

RangeSearchSet::~RangeSearchSet () {
    delete [] labels;
    delete [] distances;
}

/***********************************************************************
 * BufferPool
 ***********************************************************************/


BufferPool::BufferPool (size_t buffer_size):
    buffer_size (buffer_size)
{
    wp = buffer_size;
}

BufferPool::~BufferPool ()
{
    for (int i = 0; i < buffers.size(); i++) {
        delete [] buffers[i].ids;
        delete [] buffers[i].dis;
    }
}

void BufferPool::add (idx_t id, float dis) {
    if (wp == buffer_size) { // need new buffer
        append();
    }
    Buffer & buf = buffers.back();
    buf.ids [wp] = id;
    buf.dis [wp] = dis;
    wp++;
}


void BufferPool::append ()
{
    Buffer buf = {new idx_t [buffer_size], new float [buffer_size]};
    buffers.push_back (buf);
    wp = 0;
}

/// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
/// tables dest_ids, dest_dis
void BufferPool::copy_range (size_t ofs, size_t n,
                             idx_t * dest_ids, float *dest_dis)
{
    size_t bno = ofs / buffer_size;
    ofs -= bno * buffer_size;
    while (n > 0) {
        size_t ncopy = ofs + n < buffer_size ? n : buffer_size - ofs;
        Buffer buf = buffers [bno];
        memcpy (dest_ids, buf.ids + ofs, ncopy * sizeof(*dest_ids));
        memcpy (dest_dis, buf.dis + ofs, ncopy * sizeof(*dest_dis));
        dest_ids += ncopy;
        dest_dis += ncopy;
        ofs = 0;
        bno ++;
        n -= ncopy;
    }
}


/***********************************************************************
 * RangeSearchPartialResult
 ***********************************************************************/

void RangeQueryResult::add (float dis, idx_t id) {
    qnr++;
    pdr->add (id, dis);
}



RangeSearchPartialResult::RangeSearchPartialResult (size_t buffer_size):
    BufferPool(buffer_size)
{}


/// begin a new result
RangeQueryResult &
    RangeSearchPartialResult::new_result (idx_t qid, size_t qnr)
{
    RangeQueryResult qres = {qid, qnr, this};
    queries.push_back (qres);
    return queries.back();
}

/// called by range_search after do_allocation
void RangeSearchPartialResult::copy_result (RangeSearchSet *res, std::vector<size_t> &lims, bool incremental)
{
    size_t ofs = 0;
    for (int i = 0; i < queries.size(); i++) {
        RangeQueryResult & qres = queries[i];

        copy_range (ofs, qres.qnr,
                    res->labels + lims[qres.qid],
                    res->distances + lims[qres.qid]);
        if (incremental) {
            lims[qres.qid] += qres.qnr;
        }
        ofs += qres.qnr;
    }
}

void DynamicResultSet::finalize ()
{
    set_lims ();

    allocation ();

    merge ();
}


/// called by range_search before do_allocation
void DynamicResultSet::set_lims ()
{
    query_boundaries.resize(res->nq + 1, 0);
    for (auto &seg : seg_res) {
        for (auto &prspr : seg) {
            for (auto &query : prspr->queries) {
                query_boundaries[query.qid] += query.qnr;
            }
        }
    }
}

void DynamicResultSet::allocation() {
    size_t offset = 0;
    for (auto i = 1; i < res->nq; ++ i) {
        size_t tmp = query_boundaries[i];
        query_boundaries[i] = offset;
        offset += tmp;
    }
    query_boundaries[res->nq] = offset;
    res->labels = new idx_t[offset];
    res->distances = new float[offset];
}

void DynamicResultSet::merge() {
    for (auto &seg : seg_res) {
        for (auto &prspr : seg) {
            prspr->copy_result(res, query_boundaries, true);
        }
    }
}

void ExchangeDataset(std::vector<RangeSearchPartialResult*> &milvus_dataset,
                     std::vector<faiss::RangeSearchPartialResult*> &faiss_dataset) {
    for (auto &prspr: faiss_dataset) {
        auto mrspr = new RangeSearchPartialResult(prspr->res->buffer_size);
        mrspr->wp = prspr->wp;
        for (auto &query : prspr->queries) {
            auto qres = mrspr->new_result(query.qno, query.nres);
        }
        mrspr->buffers.resize(prspr->buffers.size());
        for (auto i = 0; i < prspr->buffers.size(); ++ i) {
            mrspr->buffers[i].ids = prspr->buffers[i].ids;
            mrspr->buffers[i].dis = prspr->buffers[i].dis;
            prspr->buffers[i].ids = nullptr;
            prspr->buffers[i].dis = nullptr;
        }
        delete prspr->res;
        milvus_dataset.push_back(mrspr);
    }
}

void MapUids(std::vector<RangeSearchPartialResult*> &milvus_dataset, std::shared_ptr<std::vector<IDType>> uids) {
    if (uids) {
        for (auto &mrspr : milvus_dataset) {
            for (auto j = 0; j < mrspr->buffers.size() - 1; ++ j) {
                auto buf = mrspr->buffers[j];
                for (auto i = 0; i < mrspr->buffer_size; ++ i) {
                    if (buf.ids[i] >= 0)
                        buf.ids[i] = uids->at(buf.ids[i]);
                }
            }
            auto buf = mrspr->buffers[mrspr->buffers.size() - 1];
            for (auto i = 0; i < mrspr->wp; ++ i) {
                if (buf.ids[i] >= 0)
                    buf.ids[i] = uids->at(buf.ids[i]);
            }
        }
    }
}

}  // namespace knowhere
}  // namespace milvus
