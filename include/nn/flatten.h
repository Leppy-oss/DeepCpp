#pragma once
#include "module.h"
#include "tensor.h"
#include <memory>

class Flatten : public Module
{
    std::size_t dim_;

public:
    Flatten(std::size_t dim = 1);
    Flatten(std::string name, std::size_t dim = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
};