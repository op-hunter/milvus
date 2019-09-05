/*******************************************************************************
* Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
******************************************************************************/
#include "milvus.grpc.pb.h"
#include "GrpcMilvusServer.h"
#include "../ServerConfig.h"
#include "../DBWrapper.h"
#include "utils/Log.h"
#include "faiss/utils.h"
#include "GrpcRequestHandler.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/grpcpp.h>


namespace zilliz {
namespace milvus {
namespace server {
namespace grpc {

static std::unique_ptr<::grpc::Server> server;

constexpr long MESSAGE_SIZE = -1;

class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
    void UpdateArguments(::grpc::ChannelArguments *args) override {
        args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
    }

    void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>> *
    plugins) override {}
};


void
GrpcMilvusServer::StartService() {
    if (server != nullptr) {
        std::cout << "stop service!\n";
        StopService();
    }

    ServerConfig &config = ServerConfig::GetInstance();
    ConfigNode server_config = config.GetConfig(CONFIG_SERVER);
    ConfigNode engine_config = config.GetConfig(CONFIG_ENGINE);
    std::string address = server_config.GetValue(CONFIG_SERVER_ADDRESS, "127.0.0.1");
    int32_t port = server_config.GetInt32Value(CONFIG_SERVER_PORT, 19530);

    faiss::distance_compute_blas_threshold = engine_config.GetInt32Value(CONFIG_DCBT, 20);

    std::string server_address(address + ":" + std::to_string(port));

    ::grpc::ServerBuilder builder;
    builder.SetOption(std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
    builder.SetMaxReceiveMessageSize(MESSAGE_SIZE); //default 4 * 1024 * 1024
    builder.SetMaxSendMessageSize(MESSAGE_SIZE);

    builder.SetCompressionAlgorithmSupportStatus(GRPC_COMPRESS_STREAM_GZIP, true);
    builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_STREAM_GZIP);
    builder.SetDefaultCompressionLevel(GRPC_COMPRESS_LEVEL_HIGH);

    GrpcRequestHandler service;

    builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    server = builder.BuildAndStart();
    server->Wait();

}

void
GrpcMilvusServer::StopService() {
    if (server != nullptr) {
        server->Shutdown();
    }
}

}
}
}
}