#include "nn/modules/cel.h"
#include "nn/tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    auto preds = Tensor::zeros({128, 10});
    auto gts = Tensor::zeros({128});
    for (std::size_t idx = 0; idx < 128; idx++)
    {
        gts->at(idx) = 1;
    }
    auto loss_fn = Cel();

    std::cout << *loss_fn(preds, gts) << std::endl;

    return 0;
}