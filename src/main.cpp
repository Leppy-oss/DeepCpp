#include "nn/modules/flatten.h"
#include "nn/modules/linear.h"
#include "nn/modules/module.h"
#include "nn/modules/relu.h"
#include "nn/tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    std::vector<std::shared_ptr<Module>> layers = {
        std::make_shared<Flatten>(),
        std::make_shared<Linear>(784, 128),
        std::make_shared<Relu>(),
        std::make_shared<Linear>(128, 64),
        std::make_shared<Relu>(),
        std::make_shared<Linear>(64, 10)
    };
    auto x = Tensor::zeros({128, 28, 28});
    auto y = x;
    for (auto &m : layers)
    {
        std::cout << m->name() << std::endl;
        y = (*m)(y);
    }

    // std::cout << *(layers[1]->parameters()[0].param()) << std::endl;
    std::cout << y->shape() << std::endl;

    return 0;
}