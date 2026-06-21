#include "nn/modules/loss.h"
#include "nn/modules/module.h"
#include "nn/tensor.h"
#include <memory>

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> x)
{
    throw std::runtime_error("Loss must be called with two tensors (predicted and target)");
}

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y)
{
    throw std::runtime_error("Cannot call forward on base Loss class");
}

std::shared_ptr<Tensor> Loss::operator()(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y)
{
    return forward(y_hat, y);
}