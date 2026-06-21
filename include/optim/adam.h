#pragma once
#include "optimizer.h"
#include "tensor.h"
#include <memory>
#include <vector>

class Adam : public Optimizer
{
private:
    float b1_;
    float b2_;
    float eps_;
    std::size_t t_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;

public:
    Adam(std::vector<Parameter> params, float lr = 1e-3f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    void step() override;
};