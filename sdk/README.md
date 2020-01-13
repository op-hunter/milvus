# Milvus C++ SDK

### Get C++ SDK

If you compile Milvus from source, C++ SDK is already in `[Milvus root path]/sdk`. If you install Milvus from Docker images, you need to download the whole `sdk` folder to your host.

### Requirements

CMake 3.14 or higher

### Build C++ SDK

You must build the C++ SDK before using it:

```shell
 # build C++ SDK
 $ cd [Milvus root path]/sdk
 $ ./build.sh
```

### Try C++ example

You must have a running Milvus server to try the C++ example. Refer to [Milvus Documentation](https://milvus.io/docs/guides/get_started/install_milvus/install_milvus.md) to learn how to install and run a Milvus server.

Run C++ example:

 ```shell
 # run Milvus C++ example
 $ cd [Milvus root path]/sdk/cmake_build/examples/simple
 $ ./sdk_simple
 ```

### Create your own C++ client project

Create a folder for the project, and copy C++ SDK header and library files into it.

```shell
 # create project folder
 $ mkdir MyMilvusClient
 $ cd MyMilvusClient
 
 # copy necessary files
 $ cp [Milvus root path]/sdk/cmake_build/libmilvus_sdk.so .
 $ cp [Milvus root path]/sdk/include/MilvusApi.h .
 $ cp [Milvus root path]/sdk/include/Status.h .
```

Create file `main.cpp` in the project folder, and copy the following code into it:

```c++
#include "./MilvusApi.h"
#include "./Status.h"

int main() {
  // connect to milvus server
  std::shared_ptr<milvus::Connection> conn = milvus::Connection::Create();
  milvus::ConnectParam param = {"127.0.0.1", "19530"};
  conn->Connect(param);
  
  // put your client code here
  
  milvus::Connection::Destroy(conn);
  return 0;
}
```

Create file `CMakeLists.txt` in the project folder, and copy the following code into it:

```bash
 cmake_minimum_required(VERSION 3.14)
 project(test)
 set(CMAKE_CXX_STANDARD 17)

 add_executable(milvus_client main.cpp)
 target_link_libraries(milvus_client
         ${PROJECT_SOURCE_DIR}/libmilvus_sdk.so
         pthread)
```

Now there are 5 files in your project:

```shell
MyMilvusClient
 |-CMakeLists.txt
 |-main.cpp
 |-libmilvus_sdk.so
 |-MilvusApi.h
 |-Status.h
```

Build the project:

```shell
 $ mkdir cmake_build
 $ cd cmake_build
 $ cmake ..
 $ make
```

Run your client program:

```shell
 $ ./milvus_client
```
