/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/

#include "resource/Resource.h"

#include <vector>
#include <string>

namespace zilliz {
namespace milvus {
namespace engine {

std::vector<std::string>
ShortestPath(const ResourcePtr &src, const ResourcePtr& dest);

}
}
}