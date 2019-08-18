#include "scheduler/ResourceFactory.h"
#include "scheduler/ResourceMgr.h"
#include "scheduler/Scheduler.h"
#include "scheduler/task/TestTask.h"
#include "utils/Log.h"
#include <gtest/gtest.h>


using namespace zilliz::milvus::engine;

TEST(normal_test, test1) {
    // ResourceMgr only compose resources, provide unified event
    auto res_mgr = std::make_shared<ResourceMgr>();
    auto disk = res_mgr->Add(ResourceFactory::Create("disk", "ssd"));
    auto cpu = res_mgr->Add(ResourceFactory::Create("cpu"));
    auto gpu1 = res_mgr->Add(ResourceFactory::Create("gpu"));
    auto gpu2 = res_mgr->Add(ResourceFactory::Create("gpu"));

    auto IO = Connection("IO", 500.0);
    auto PCIE = Connection("IO", 11000.0);
    res_mgr->Connect(disk, cpu, IO);
    res_mgr->Connect(cpu, gpu1, PCIE);
    res_mgr->Connect(cpu, gpu2, PCIE);

    res_mgr->Start();

    auto scheduler = new Scheduler(res_mgr);
    scheduler->Start();

    auto task1 = std::make_shared<TestTask>();
    auto task2 = std::make_shared<TestTask>();
    auto task3 = std::make_shared<TestTask>();
    auto task4 = std::make_shared<TestTask>();
    if (auto observe = disk.lock()) {
        observe->task_table().Put(task1);
        observe->task_table().Put(task2);
        observe->task_table().Put(task3);
        observe->task_table().Put(task4);
        std::cout << "disk:" << std::endl;
        std::cout << observe->task_table().Dump() << std::endl;
    }

    sleep(5);

    if (auto observe = disk.lock()) {
        std::cout << "disk:" << std::endl;
        std::cout << observe->task_table().Dump() << std::endl;
    }
    if (auto observe = cpu.lock()) {
        std::cout << "cpu:" << std::endl;
        std::cout << observe->task_table().Dump() << std::endl;
    }
    scheduler->Stop();
    res_mgr->Stop();

    ASSERT_EQ(task1->load_count_, 1);
    ASSERT_EQ(task1->exec_count_, 1);
}
