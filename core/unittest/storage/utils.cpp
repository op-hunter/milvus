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

#include <fstream>
#include <string>

#include "config/ServerConfig.h"
#include "storage/utils.h"
#include "utils/CommonUtil.h"

namespace {

static const char* CONFIG_STR =
    "storage_config:\n"
    "  primary_path: /tmp/milvus\n"
    "  secondary_path:\n"
    "  s3_enable: true\n"
    "  s3_address: 127.0.0.1\n"
    "  s3_port: 9000\n"
    "  s3_access_key: minioadmin\n"
    "  s3_secret_key: minioadmin\n"
    "  s3_bucket: milvus-bucket\n"
    "\n";

void
WriteToFile(const std::string& file_path, const char* content) {
    std::fstream fs(file_path.c_str(), std::ios_base::out);

    // write data to file
    fs << content;
    fs.close();
}

}  // namespace

void
StorageTest::SetUp() {
    std::string config_path(CONFIG_PATH);
    milvus::CommonUtil::CreateDirectory(config_path);
    config_path += CONFIG_FILE;
    WriteToFile(config_path, CONFIG_STR);

    // milvus::server::Config& config = milvus::server::Config::GetInstance();
    // ASSERT_TRUE(config.LoadConfigFile(config_path).ok());
}

void
StorageTest::TearDown() {
    std::string config_path(CONFIG_PATH);
    milvus::CommonUtil::DeleteDirectory(config_path);
}
