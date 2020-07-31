/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
#include "RHNSW.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexRHNSW.h>



namespace faiss {

    using storage_idx_t = RHNSW::storage_idx_t ;
    using NodeDistCloser = RHNSW::NodeDistCloser;
/**************************************************************
 * hnsw structure implementation
 **************************************************************/

    inline int *RHNSW::get_neighbor_link(idx_t no, int layer_no) const {
      return layer_no == 0 ? (int *) (level0_links + no * level0_link_size) : (int *) (linkLists[no] +
                                                                                       (layer_no - 1) * link_size);
    }

    inline unsigned short int RHNSW::get_neighbors_num(int *p) const {
      return *((unsigned short int *) p);
    }

    inline void RHNSW::set_neighbors_num(int *p, unsigned short int num) const {
      *((unsigned short int *) (p)) = *((unsigned short int *) (&num));
    }

    inline int RHNSW::random_level(double arg) {
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      double r = -log(distribution(level_generator)) * arg;
      return (int) r;
    }

    inline int RHNSW::prepare_level_tab(size_t n, bool preset_levels) {
//  size_t n0 = offsets.size() - 1;
      size_t n0 = levels.size();

      if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
      } else {
        FAISS_ASSERT(n0 == levels.size());
        double random_level_arg = 1 / log(double(M));
//    levels.resize(n0 + n);
        for (int i = 0; i < n; i++) {
          int pt_level = random_level(random_level_arg);
          levels.push_back(pt_level);
        }
      }

      char *level0_links_new = (char *) malloc((n0 + n) * level0_link_size);
      if (level0_links_new == nullptr) {
        throw std::runtime_error("No enough memory 4 level0_links!");
      }
      memset(level0_links_new, 0, (n0 + n) * level0_link_size);
//  printf("level0_links_new addr: %p has malloc %d bytes space, level0_link_size = %d\n", level0_links_new, level0_link_size * (n0 + n), level0_link_size);
      if (level0_links) {
        memcpy(level0_links_new, level0_links, n0 * level0_link_size);
        free(level0_links);
      }
      level0_links = level0_links_new;

      char **linkLists_new = (char **) malloc(sizeof(void *) * (n0 + n));
      if (linkLists_new == nullptr) {
        throw std::runtime_error("No enough memory 4 level0_links_new!");
      }
//  printf("linkLists_new addr: %p has malloc %d bytes space, link_size = %d\n", linkLists_new, sizeof(void*) * (n0 + n), link_size);
      if (linkLists) {
        memcpy(linkLists_new, linkLists, n0 * sizeof(void *));
        free(linkLists);
      }
      linkLists = linkLists_new;

      int max_level = 0;
      int debug_space = 0;
      for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0];
        if (pt_level > max_level) max_level = pt_level;
        if (pt_level) {
          linkLists[n0 + i] = (char *) malloc(link_size * pt_level + 1);
          if (linkLists[n0 + i] == nullptr) {
            throw std::runtime_error("No enough memory 4 linkLists!");
          }
          memset(linkLists[n0 + i], 0, link_size * pt_level + 1);
        }
      }

      return max_level;
    }

    inline void RHNSW::MinimaxHeap::push(storage_idx_t i, float v) {
      if (k == n) {
        if (v >= dis[0]) return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
      }
      faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
      ++nvalid;
    }

    inline float RHNSW::MinimaxHeap::max() const {
      return dis[0];
    }

    inline int RHNSW::MinimaxHeap::size() const {
      return nvalid;
    }

    inline void RHNSW::MinimaxHeap::clear() {
      nvalid = k = 0;
    }

    inline int RHNSW::MinimaxHeap::pop_min(float *vmin_out) {
      assert(k > 0);
      // returns min. This is an O(n) operation
      int i = k - 1;
      while (i >= 0) {
        if (ids[i] != -1) break;
        i--;
      }
      if (i == -1) return -1;
      int imin = i;
      float vmin = dis[i];
      i--;
      while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
          vmin = dis[i];
          imin = i;
        }
        i--;
      }
      if (vmin_out) *vmin_out = vmin;
      int ret = ids[imin];
      ids[imin] = -1;
      --nvalid;

      return ret;
    }

    inline int RHNSW::MinimaxHeap::count_below(float thresh) {
      int n_below = 0;
      for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
          n_below++;
        }
      }

      return n_below;
    }

    /*
    namespace greedy {
        /// greedily update a nearest vector at a given level
        void greedy_update_nearest(const RHNSW &hnsw,
                                   DistanceComputer &qdis,
                                   int level,
                                   storage_idx_t &nearest,
                                   float &d_nearest) {
          for (;;) {
            storage_idx_t prev_nearest = nearest;

            int *cur_links = hnsw.get_neighbor_link(nearest, level);
            int *cur_neighbors = cur_links + 1;
            auto cur_neighbor_num = hnsw.get_neighbors_num(cur_links);
            for (auto i = 0; i < cur_neighbor_num; ++i) {
              storage_idx_t v = cur_neighbors[i];
              if (v < 0) break;
              float dis = qdis(v);
              if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
              }
            }
            if (nearest == prev_nearest) {
              return;
            }
          }
        }
    }
    */



/**************************************************************
 * new implementation of hnsw ispired by hnswlib
 * by cmli@zilliz   July 30, 2020
 **************************************************************/
    inline void RHNSW::addPoint(DistanceComputer &ptdis, int pt_level, int pt_id,
                                std::vector <omp_lock_t> &locks,
                                VisitedTable &vt) {
      storage_idx_t nearest;
#pragma omp critical
      {
        nearest = entry_point;

        if (nearest == -1) {
          max_level = pt_level;
          entry_point = pt_id;
        }
      }

      if (nearest < 0) {
        return;
      }

      omp_set_lock(&locks[pt_id]);

      int level = max_level; // level at which we start adding neighbors
      float d_nearest = ptdis(nearest);

      for (; level > pt_level; level--) {
//        faiss::greedy::greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        for (;;) {
          storage_idx_t prev_nearest = nearest;

          int *cur_links = get_neighbor_link(nearest, level);
          int *cur_neighbors = cur_links + 1;
          auto cur_neighbor_num = get_neighbors_num(cur_links);
          for (auto i = 0; i < cur_neighbor_num; ++i) {
            storage_idx_t v = cur_neighbors[i];
            if (v < 0) break;
            float dis = ptdis(v);
            if (dis < d_nearest) {
              nearest = v;
              d_nearest = dis;
            }
          }
          if (nearest == prev_nearest) {
            break;
          }
        }
      }

      for (; level >= 0; level--) {
//    add_links_starting_from(ptdis, pt_id, nearest, d_nearest,
//                            level, locks.data(), vt);
        std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> top_cand =
                search_layer(ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt);
        nearest = top_cand.top().id;
        d_nearest = ptdis(nearest);
        make_connection(ptdis, pt_id, top_cand, locks.data(), level);
        vt.advance();
      }

      omp_unset_lock(&locks[pt_id]);

      if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
      }
    }

    inline
    std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>>
    RHNSW::search_layer(DistanceComputer &ptdis,
                        storage_idx_t pt_id,
                        storage_idx_t nearest,
                        float d_nearest,
                        int level,
                        omp_lock_t *locks,
                        VisitedTable &vt) {
      std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> top_candidates;
      std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> candidate_set;

      float lb = d_nearest;
      top_candidates.emplace(d_nearest, nearest);
      candidate_set.emplace(-d_nearest, nearest);
      vt.set(nearest);
      while (!candidate_set.empty()) {
        NodeDistCloser currNode = candidate_set.top();
        if ((-currNode.d) > lb)
          break;
        candidate_set.pop();
        int cur_id = currNode.id;
        omp_set_lock(&locks[cur_id]);
        int *cur_link = get_neighbor_link(cur_id, level);
        auto cur_neighbor_num = get_neighbors_num(cur_link);
        for (auto i = 1; i <= cur_neighbor_num; ++i) {
          int candidate_id = cur_link[i];
          if (vt.get(candidate_id)) continue;
          vt.set(candidate_id);
          float dcand = ptdis(candidate_id);
          if (top_candidates.size() < efConstruction || lb > dcand) {
            candidate_set.emplace(-dcand, candidate_id);
            top_candidates.emplace(dcand, candidate_id);
            if (top_candidates.size() > efConstruction)
              top_candidates.pop();
            if (!top_candidates.empty())
              lb = top_candidates.top().d;
          }
        }
        omp_unset_lock(&locks[cur_id]);
      }
      return top_candidates;
    }

    inline
    std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>>
    RHNSW::search_base_layer(DistanceComputer &ptdis,
                             storage_idx_t nearest,
                             float d_nearest,
                             VisitedTable &vt) const {
      std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> top_candidates;
      std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> candidate_set;

      float lb = d_nearest;
      top_candidates.emplace(d_nearest, nearest);
      candidate_set.emplace(-d_nearest, nearest);
      vt.set(nearest);
      while (!candidate_set.empty()) {
        NodeDistCloser currNode = candidate_set.top();
        if ((-currNode.d) > lb)
          break;
        candidate_set.pop();
        int cur_id = currNode.id;
        int *cur_link = get_neighbor_link(cur_id, 0);
        auto cur_neighbor_num = get_neighbors_num(cur_link);
        for (auto i = 1; i <= cur_neighbor_num; ++i) {
          int candidate_id = cur_link[i];
          if (vt.get(candidate_id)) continue;
          vt.set(candidate_id);
          float dcand = ptdis(candidate_id);
          if (top_candidates.size() < efSearch || lb > dcand) {
            candidate_set.emplace(-dcand, candidate_id);
            top_candidates.emplace(dcand, candidate_id);
            if (top_candidates.size() > efSearch)
              top_candidates.pop();
            if (!top_candidates.empty())
              lb = top_candidates.top().d;
          }
        }
      }
      return top_candidates;
    }

    inline void
    RHNSW::make_connection(DistanceComputer &ptdis,
                           storage_idx_t pt_id,
                           std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> &cand,
                           omp_lock_t *locks,
                           int level) {
      int maxM = level ? M : M << 1;
      int *selectedNeighbors = (int *) malloc(sizeof(int) * maxM);
      int selectedNeighborsNum = 0;
      prune_neighbors(cand, ptdis, maxM, selectedNeighbors, selectedNeighborsNum);
      if (selectedNeighborsNum > maxM)
        throw std::runtime_error("Wrong size of candidates returned by prune_neighbors!");

      int *cur_link = get_neighbor_link(pt_id, level);
      if (*cur_link)
        throw std::runtime_error("The newly inserted element should have blank link");

      set_neighbors_num(cur_link, selectedNeighborsNum);
      for (auto i = 1; i <= selectedNeighborsNum; ++i) {
        if (cur_link[i])
          throw std::runtime_error("Possible memory corruption.");
        if (level > levels[selectedNeighbors[i - 1]])
          throw std::runtime_error("Trying to make a link on a non-exisitent level.");
        cur_link[i] = selectedNeighbors[i - 1];
      }

      for (auto i = 0; i < selectedNeighborsNum; ++i) {
        omp_set_lock(&locks[selectedNeighbors[i]]);
        int *selected_link = get_neighbor_link(selectedNeighbors[i], level);
        auto selected_neighbor_num = get_neighbors_num(selected_link);
        if (selected_neighbor_num > maxM)
          throw std::runtime_error("Bad value of selected_neighbor_num.");
        if (selectedNeighbors[i] == pt_id)
          throw std::runtime_error("Trying to connect an element to itself.");
        if (level > levels[selectedNeighbors[i]])
          throw std::runtime_error("Trying to make a link on a non-exisitent level.");
        if (selected_neighbor_num < maxM) {
          selected_link[selected_neighbor_num + 1] = pt_id;
          set_neighbors_num(selected_link, selected_neighbor_num + 1);
        } else {
          double d_max = ptdis(selectedNeighbors[i]);
          std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> candi;
          candi.emplace(d_max, pt_id);
          for (auto j = 1; j <= selected_neighbor_num; ++j) {
            candi.emplace(ptdis.symmetric_dis(selectedNeighbors[i], selected_link[j]), selected_link[j]);
          }
          int indx = 0;
          prune_neighbors(candi, ptdis, maxM, selected_link + 1, indx);
          set_neighbors_num(selected_link, indx);
        }
        omp_unset_lock(&locks[selectedNeighbors[i]]);
      }

      free(selectedNeighbors);
    }

    inline void RHNSW::prune_neighbors(std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> &cand,
                                       DistanceComputer &ptdis,
                                       const int maxM, int *ret, int &ret_len) {
      if (cand.size() < maxM) {
        while (!cand.empty()) {
          ret[ret_len++] = cand.top().id;
          cand.pop();
        }
        return;
      }
      std::priority_queue <NodeDistCloser> closest;

      while (!cand.empty()) {
        closest.emplace(-cand.top().d, cand.top().id);
        cand.pop();
      }

      while (closest.size()) {
        if (ret_len >= maxM)
          break;
        NodeDistCloser curr = closest.top();
        float dist_to_query = -curr.d;
        closest.pop();
        bool good = true;
        for (auto i = 0; i < ret_len; ++i) {
          float cur_dist = ptdis.symmetric_dis(curr.id, ret[i]);
          if (cur_dist < dist_to_query) {
            good = false;
            break;
          }
        }
        if (good) {
          ret[ret_len++] = curr.id;
        }
      }
    }

    inline void RHNSW::searchKnn(DistanceComputer &qdis, int k,
                                 idx_t *I, float *D,
                                 VisitedTable &vt) const {
      if (levels.size() == 0)
        return;
      int ep = entry_point;
      float dist = qdis(ep);

      for (auto i = max_level; i > 0; --i) {
        bool good = true;
        while (good) {
          good = false;
          int *ep_link = get_neighbor_link(ep, i);
          auto ep_neighbors_cnt = get_neighbors_num(ep_link);
          for (auto j = 1; j <= ep_neighbors_cnt; ++j) {
            int cand = ep_link[j];
            if (cand < 0 || cand > levels.size())
              throw std::runtime_error("cand error");
            float d = qdis(cand);
            if (d < dist) {
              dist = d;
              ep = cand;
              good = true;
            }
          }
        }
      }
      std::priority_queue <NodeDistCloser, std::vector<NodeDistCloser>> top_candidates = search_base_layer(qdis, ep,
                                                                                                           dist, vt);
      while (top_candidates.size() > k)
        top_candidates.pop();
      int i = 0;
      while (!top_candidates.empty()) {
        I[i] = top_candidates.top().id;
        D[i] = top_candidates.top().d;
        i++;
        top_candidates.pop();
      }
      vt.advance();
    }

}
