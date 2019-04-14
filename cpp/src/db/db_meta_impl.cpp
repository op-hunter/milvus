#include "db_meta_impl.h"

namespace zilliz {
namespace vecwise {
namespace engine {

Status DBMetaImpl::DBMetaImpl(DBMetaOptions options_)
    : _options(options_) {
    initialize();
}

Status DBMetaImpl::initialize() {
    // PXU TODO: Create DB Connection
    return Status.OK();
}

Status DBMetaImpl::add_group(const std::string& group_id_, GroupSchema& group_info_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::get_group(const std::string& group_id_, GroupSchema& group_info_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::has_group(const std::string& group_id_, bool& has_or_not_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::add_group_file(const std::string& group_id_,
                              GroupFileSchema& group_file_info_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::has_group_file(const std::string& group_id_,
                              const std::string& file_id_,
                              bool& has_or_not_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::get_group_file(const std::string& group_id_,
                              const std::string& file_id_,
                              GroupFileSchema& group_file_info_) {
    //PXU TODO
    return Status.OK();
}

Status DBMetaImpl::get_group_files(const std::string& group_id_,
                               GroupFilesSchema& group_files_info_) {
    // PXU TODO
    return Status.OK();
}

Status DBMetaImpl::mark_group_file_as_index(const std::string& group_id_,
                                        const std::string& file_id_) {
    //PXU TODO
    return Status.OK();
}

} // namespace engine
} // namespace vecwise
} // namespace zilliz
