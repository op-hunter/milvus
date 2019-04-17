#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <fstream>
#include "DBMetaImpl.h"
#include "IDGenerator.h"

namespace zilliz {
namespace vecwise {
namespace engine {
namespace meta {

long GetFileSize(const std::string& filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

DBMetaImpl::DBMetaImpl(const DBMetaOptions& options_)
    : _options(options_) {
    initialize();
}

Status DBMetaImpl::initialize() {
    // PXU TODO: Create DB Connection
    return Status::OK();
}

Status DBMetaImpl::add_group(const GroupOptions& options_,
            const std::string& group_id_,
            GroupSchema& group_info_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::get_group(const std::string& group_id_, GroupSchema& group_info_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::has_group(const std::string& group_id_, bool& has_or_not_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::add_group_file(const std::string& group_id,
                              GroupFileSchema& group_file_info,
                              GroupFileSchema::FILE_TYPE file_type) {
    return add_group_file(group_id, Meta::GetDate(), group_file_info);
}

Status DBMetaImpl::add_group_file(const std::string& group_id,
                              DateT date,
                              GroupFileSchema& group_file_info,
                              GroupFileSchema::FILE_TYPE file_type) {
    //PXU TODO
    std::stringstream ss;
    SimpleIDGenerator g;
    std::string suffix = (file_type == GroupFileSchema::RAW) ? ".raw" : ".index";
    ss << "/tmp/test/" << date
                       << "/" << g.getNextIDNumber()
                       << suffix;
    group_file_info.group_id = "1";
    group_file_info.dimension = 64;
    group_file_info.location = ss.str();
    group_file_info.date = date;
    return Status::OK();
}

Status DBMetaImpl::files_to_index(GroupFilesSchema& files) {
    // PXU TODO
    files.clear();
    std::stringstream ss;
    ss << "/tmp/test/" << Meta::GetDate();
    boost::filesystem::path path(ss.str().c_str());
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr) {
        /* std::cout << itr->path().string() << std::endl; */
        GroupFileSchema f;
        f.location = itr->path().string();
        std::string suffixStr = f.location.substr(f.location.find_last_of('.') + 1);
        if (suffixStr == "index") continue;
        if (1024*1024*1000 >= GetFileSize(f.location)) continue;
        std::cout << "[About to index] " << f.location << std::endl;
        f.date = Meta::GetDate();
        files.push_back(f);
    }
    return Status::OK();
}

Status DBMetaImpl::files_to_merge(const std::string& group_id,
        DatePartionedGroupFilesSchema& files) {
    //PXU TODO
    files.clear();
    std::stringstream ss;
    ss << "/tmp/test/" << Meta::GetDate();
    boost::filesystem::path path(ss.str().c_str());
    boost::filesystem::directory_iterator end_itr;
    GroupFilesSchema gfiles;
    DateT date = Meta::GetDate();
    files[date] = gfiles;
    for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr) {
        /* std::cout << itr->path().string() << std::endl; */
        GroupFileSchema f;
        f.location = itr->path().string();
        std::string suffixStr = f.location.substr(f.location.find_last_of('.') + 1);
        if (suffixStr == "index") continue;
        if (1024*1024*1000 < GetFileSize(f.location)) continue;
        std::cout << "About to merge " << f.location << std::endl;
        files[date].push_back(f);
    }

    return Status::OK();
}

Status DBMetaImpl::has_group_file(const std::string& group_id_,
                              const std::string& file_id_,
                              bool& has_or_not_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::get_group_file(const std::string& group_id_,
                              const std::string& file_id_,
                              GroupFileSchema& group_file_info_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::get_group_files(const std::string& group_id_,
                               const int date_delta_,
                               GroupFilesSchema& group_files_info_) {
    // PXU TODO
    return Status::OK();
}

Status DBMetaImpl::update_group_file(const GroupFileSchema& group_file_) {
    //PXU TODO
    return Status::OK();
}

Status DBMetaImpl::update_files(const GroupFilesSchema& files) {
    //PXU TODO
    return Status::OK();
}

} // namespace meta
} // namespace engine
} // namespace vecwise
} // namespace zilliz
