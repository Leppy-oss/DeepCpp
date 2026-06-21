#pragma once
#include "module.h"
#include "tensor.h"

class Relu : public Module
{
public:
    Relu();
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
};