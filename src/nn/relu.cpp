#include "nn/relu.h"
#include "nn/module.h"
#include "tensor.h"
#include <functional>
#include <memory>

Relu::Relu() : Module("Relu") {}

std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> x)
{
    std::vector<float> out_data(x->numel());
    for (std::size_t i = 0; i < x->numel(); i++)
    {
        out_data[i] = std::max(0.0f, x->at(i));
    }

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (x->requires_grad())
    {
        parents = {x};
        gradfn = [x](std::shared_ptr<Tensor> grad_prev)
        {
            auto grad_update = Tensor::zeros(x->shape());
            for (std::size_t i = 0; i < x->numel(); i++)
            {
                grad_update->at(i) = x->at(i) > 0.0f ? grad_prev->at(i) : 0.0f;
            }
            x->add_grad(grad_update);
        };
    }

    return std::make_shared<Tensor>(out_data, x->shape(), x->requires_grad(), gradfn, parents);
}