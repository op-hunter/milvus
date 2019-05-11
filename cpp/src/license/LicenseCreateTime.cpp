//
// Created by zilliz on 19-5-11.
//

#include <gtest/gtest.h>
#include <iostream>
#include <map>

#include "utils/Error.h"
#include "license/License.h"
#include "license/LicensePublic.h"

using namespace zilliz::vecwise;

//TEST(LicenseTest, LICENSE_TEST) {
//
//    std::string path1 = "/tmp/vecwise_engine.sha";
//    std::string path2 = "/tmp/vecwise_engine.license";
//    std::cout << "This is create  licenseTime " << std::endl;
//
//    server::ServerError err;
//    int deviceCount=0;
//
//    std::vector<std::string> shas;
//    err = server::LicenseLoad(path1,deviceCount,shas);
//    ASSERT_EQ(err, server::SERVER_SUCCESS);
//
//    std::map<int,std::string> uuidEncryption;
//    std::cout<< "deviceCount : " << deviceCount << std::endl;
//    for(int i=0;i<deviceCount;i++)
//    {
//        std::cout<< "uuidshas : " << shas[i] << std::endl;
//        uuidEncryption.insert(std::make_pair(i,shas[i]));
//    }
//    int64_t  RemainingTime;
//    std::cout<< "cin  RemainingTime: (Hours)"  << std::endl;
//    std::cin >> RemainingTime ;
//
//    err = server::LiSave(path2,deviceCount,uuidEncryption,RemainingTime);
//    ASSERT_EQ(err, server::SERVER_SUCCESS);
//
//    int64_t  RemainingTimecheck;
//    std::map<int,std::string> uuidEncryptioncheck;
//    int deviceCountcheck;
//    err = server::LiLoad(path2,deviceCountcheck,uuidEncryptioncheck,RemainingTimecheck);
//
//    printf("\n  deviceCountcheck  = %d\n",deviceCountcheck);
//    std::cout<< "RemainingTimecheck: "  << RemainingTimecheck<< std::endl;
//    std::cout<< "success"  << std::endl;
//
//}