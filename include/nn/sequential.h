#pragma once
#include "nn/module.h"
#include "tensor.h"
#include <memory>
#include <vector>

class Sequential : public Module
{
public:
    Sequential(const std::vector<std::shared_ptr<Module>> &modules);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
};