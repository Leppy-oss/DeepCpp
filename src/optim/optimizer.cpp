#include "optim/optimizer.h"
#include "nn/module.h"
#include <vector>

Optimizer::Optimizer(std::vector<Parameter> params, float lr) : params_{params}, lr_{lr} {}

void Optimizer::step() { throw std::runtime_error("Base optimizer has no step implementation"); }

void Optimizer::zero_grad()
{
    for (Parameter &p : params_)
    {
        p.param()->zero_grad();
    }
}