#include "nn/cel.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    auto preds = std::make_shared<Tensor>(std::vector{0.0f, 0.0f, 1.0f, 0.0f}, true);
    auto gts = std::make_shared<Tensor>(2.0f);

    auto loss_fn = Cel(0);
    auto loss = loss_fn(preds, gts);
    loss->backward();

    std::cout << *loss << std::endl;
    std::cout << *loss->grad() << std::endl;

    return 0;
}