#include "nn/tensor.h"
#include <iostream>
#include <vector>

int main()
{
    auto weights = std::make_shared<Tensor>(2.0f, true);
    std::cout << "Memory leak test begin" << std::endl;
    {
        auto inputs = std::make_shared<Tensor>(3.0f);

        auto loss = weights + inputs;

        std::cout << "Exit block scope" << std::endl;
    }

    std::cout << "Memory leak test end" << std::endl;

    return 0;
}