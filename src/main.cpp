#include "nn/tensor.h"
#include <iostream>
#include <vector>

int main()
{
    std::vector<float> v{0.0, 1.0, 2.0, 3.0};
    Tensor t1(v);
    Tensor t2({v, v, v, v});

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;

    return 0;
}