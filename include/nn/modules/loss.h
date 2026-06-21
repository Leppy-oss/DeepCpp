#pragma once
#include "module.h"
#include "nn/tensor.h"
#include <memory>

class Loss : public Module
{
public:
    using Module::Module;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y);
};