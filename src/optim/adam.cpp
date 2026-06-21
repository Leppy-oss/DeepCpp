#include "optim/adam.h"
#include "nn/module.h"
#include "tensor.h"
#include <memory>
#include <vector>

Adam::Adam(std::vector<Parameter> params, float lr, float b1, float b2, float eps) :
    Optimizer(params, lr),
    b1_{b1},
    b2_{b2},
    eps_{eps},
    t_{0}
{
    for (const Parameter &p : params_)
    {
        m_.push_back(Tensor::zeros(p.param()->shape()));
        v_.push_back(Tensor::zeros(p.param()->shape()));
    }
}

void Adam::step()
{
    t_++;
    for (std::size_t i = 0; i < params_.size(); i++)
    {
        auto param = params_[i].param();
        auto grad = param->grad();

        if (!grad)
        {
            continue;
        }

        if (param->numel() != grad->numel())
        {
            throw std::runtime_error(
                "Mismatched param and grad sizes of " + std::to_string(param->numel()) + " and " +
                std::to_string(grad->numel()) + " (respectively)"
            );
        }

        float bc1 = 1.0f - std::pow(b1_, static_cast<float>(t_));
        float bc2 = 1.0f - std::pow(b2_, static_cast<float>(t_));

        for (std::size_t grad_idx = 0; grad_idx < grad->numel(); grad_idx++)
        {
            float g = grad->at(grad_idx);

            m_[i]->at(grad_idx) = b1_ * m_[i]->at(grad_idx) + (1.0f - b1_) * g;
            v_[i]->at(grad_idx) = b2_ * v_[i]->at(grad_idx) + (1.0f - b2_) * g * g;

            float m_hat = m_[i]->at(grad_idx) / bc1;
            float v_hat = v_[i]->at(grad_idx) / bc2;

            param->at(grad_idx) -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}