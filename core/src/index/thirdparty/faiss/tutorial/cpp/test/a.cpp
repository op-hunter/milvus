#include <iostream>
#include <cstdio>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "wrong argc!" << std::endl;
        return EXIT_FAILURE;
    }
    int arg = atoi(argv[1]);
    std::cout << "hello ";
    printf("%s ", argv[0]);
    std::cout << "arg: " << arg << std::endl;
    return EXIT_SUCCESS;
}
