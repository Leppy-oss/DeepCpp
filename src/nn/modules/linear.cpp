#include "nn/modules/linear.h"
#include "nn/tensor.h"
#include <memory>
#include <random>

Linear::Linear(std::size_t in_features, std::size_t out_features, std::size_t seed) :
    weight_{Tensor::zeros({in_features, out_features}, true)},
    bias_{Tensor::zeros({out_features}, true)},
    seed_{seed}
{
    add_parameter("weight", weight_);
    add_parameter("bias", bias_);
    reset_parameters();
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) { return x->mm(weight_) + bias_; }

void Linear::reset_parameters()
{
    float g = std::sqrt(2.0f);
    std::size_t fan_in = weight_->shape()[0];
    float b = g * std::sqrt(3.0f / fan_in);
    std::mt19937 gen(seed_);

    for (std::size_t i = 0; i < weight_->numel(); i++)
    {
        weight_->storage(i) = std::uniform_real_distribution<float>(-b, b)(gen);
    }
}