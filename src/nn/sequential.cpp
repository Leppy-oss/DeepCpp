#include "nn/sequential.h"
#include "nn/module.h"
#include "tensor.h"
#include <memory>
#include <vector>

Sequential::Sequential(const std::vector<std::shared_ptr<Module>> &modules) : Module("Sequential")
{
    if (modules.size() < 1)
    {
        throw std::invalid_argument("Sequential must contain at least 1 module");
    }
    for (const auto &module : modules)
    {
        add_module(module);
    }
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> x)
{
    auto y = std::make_shared<Tensor>(*x);
    for (const auto &module : modules())
    {
        y = module->forward(y);
    }
    return y;
}