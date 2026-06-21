#pragma once
#include "nn/module.h"
#include <vector>

class Optimizer
{
protected:
    std::vector<Parameter> params_;
    float lr_;

public:
    Optimizer(std::vector<Parameter> params, float lr);
    virtual void step();
    virtual void zero_grad();
};