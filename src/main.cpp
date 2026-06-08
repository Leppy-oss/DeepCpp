#include <iostream>
#include <nn/tensor.h>
#include <utils.h>
#include <vector>

int main()
{
    auto t1 = std::make_shared<Tensor>(5.0f);
    auto t2 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    auto t3 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}}
    );
    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;
    std::cout << *t3 << std::endl;
    auto t4 = t1 * t2;
    auto t5 = t2 * t3;
    std::cout << *t4 << std::endl;
    std::cout << *t5 << std::endl;

    return 0;
}