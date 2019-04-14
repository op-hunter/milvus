#ifndef VECENGINE_OPTIONS_H_
#define VECENGINE_OPTIONS_H_

#include <string>

namespace vecengine {

struct Options {
    uint16_t  memory_sync_interval = 10;
    uint16_t  raw_file_merge_trigger_number = 100;
    size_t  raw_to_index_trigger_size = 100000;
}; // Options


struct GroupOptions {
    size_t dimension;
    bool has_id = false;
}; // GroupOptions


struct MetaOptions {
}; // MetaOptions


struct DBMetaOptions : public MetaOptions {
    std::string backend_uri;
    std::string dbname;
}; // DBMetaOptions


} // namespace vecengine

#endif // VECENGINE_OPTIONS_H_
