#include "nn/softmax.h"
#include "tensor.h"
#include <functional>
#include <limits>
#include <memory>
#include <vector>

Softmax::Softmax(std::size_t dim) : Module("Softmax"), dim_{dim} {}

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> x)
{
    if (x->ndim() == 0)
    {
        throw std::invalid_argument("Softmax requires tensor to be at least 1d");
    }

    if (dim_ >= x->ndim())
    {
        throw std::runtime_error(
            "Dim " + std::to_string(dim_) + " invalid for tensor with " + std::to_string(x->ndim()) + " dimensions"
        );
    }

    tensor::Shape shape = x->shape();
    std::size_t shape_dim = shape[dim_];

    std::size_t outer = 1, inner = 1;
    for (std::size_t dim = 0; dim < x->ndim(); dim++)
    {
        if (dim < dim_)
        {
            outer *= shape[dim];
        }
        else if (dim > dim_)
        {
            inner *= shape[dim];
        }
    }

    std::vector<float> out_data(x->numel());

    for (std::size_t outer_idx = 0; outer_idx < outer; outer_idx++)
    {
        for (std::size_t inner_idx = 0; inner_idx < inner; inner_idx++)
        {
            std::size_t outer_inner_idx = outer_idx * shape_dim * inner + inner_idx;
            float max_val = std::numeric_limits<float>::lowest();

            for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
            {
                max_val = std::max(max_val, x->at(dim_idx * inner + outer_inner_idx));
            }

            float sum = 0.0f;

            for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
            {
                float ve = std::exp(x->at(dim_idx * inner + outer_inner_idx) - max_val);
                out_data[dim_idx * inner + outer_inner_idx] = ve;
                sum += ve;
            }

            for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
            {
                out_data[dim_idx * inner + outer_inner_idx] /= sum;
            }
        }
    }

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (x->requires_grad())
    {
        gradfn = [x, shape, inner, outer, shape_dim, out_data](std::shared_ptr<Tensor> grad_prev)
        {
            auto grad_update = Tensor::zeros(shape);
            for (std::size_t outer_idx = 0; outer_idx < outer; outer_idx++)
            {
                for (std::size_t inner_idx = 0; inner_idx < inner; inner_idx++)
                {
                    std::size_t outer_inner_idx = outer_idx * shape_dim * inner + inner_idx;
                    float dp = 0.0f;

                    for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
                    {
                        std::size_t idx = dim_idx * inner + outer_inner_idx;
                        dp += grad_prev->at(idx) * out_data[idx];
                    }

                    for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
                    {
                        std::size_t idx = dim_idx * inner + outer_inner_idx;
                        grad_update->at(idx) = out_data[idx] * (grad_prev->at(idx) - dp);
                    }
                }
            }
            x->add_grad(grad_update);
        };
        parents = {x};
    }
    return std::make_shared<Tensor>(out_data, x->shape(), x->requires_grad(), gradfn, parents);
}