#include "nn/modules/softmax.h"
#include "nn/tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    auto x = Tensor::zeros({128, 10});
    auto sm = Softmax();

    std::cout << *sm(x) << std::endl;

    return 0;
}