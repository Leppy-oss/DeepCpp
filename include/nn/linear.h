#pragma once
#include "module.h"
#include "tensor.h"
#include <memory>

class Linear : public Module
{
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
    std::size_t seed_;

public:
    Linear(std::size_t in_features, std::size_t out_features, std::size_t seed = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    void reset_parameters();
};