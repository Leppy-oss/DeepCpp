#include "nn/argmax.h"
#include "tensor.h"
#include <functional>
#include <limits>
#include <memory>
#include <vector>

Argmax::Argmax(std::size_t dim) : Module("Argmax"), dim_{dim} {}

std::shared_ptr<Tensor> Argmax::forward(std::shared_ptr<Tensor> x)
{
    if (x->ndim() == 0)
    {
        throw std::invalid_argument("Argmax requires tensor to be at least 1d");
    }

    if (dim_ >= x->ndim())
    {
        throw std::out_of_range(
            "Dim " + std::to_string(dim_) + " invalid for tensor with " + std::to_string(x->ndim()) + " dimensions"
        );
    }

    tensor::Shape in_shape = x->shape();
    std::size_t shape_dim = in_shape[dim_];

    tensor::Shape out_shape;
    std::size_t outer = 1, inner = 1;
    for (std::size_t dim = 0; dim < x->ndim(); dim++)
    {
        if (dim < dim_)
        {
            outer *= in_shape[dim];
            out_shape.push_back(in_shape[dim]);
        }
        else if (dim > dim_)
        {
            inner *= in_shape[dim];
            out_shape.push_back(in_shape[dim]);
        }
    }
    if (out_shape.empty())
    {
        out_shape.push_back(1);
    }

    std::vector<float> out_data(outer * inner);

    for (std::size_t outer_idx = 0; outer_idx < outer; outer_idx++)
    {
        for (std::size_t inner_idx = 0; inner_idx < inner; inner_idx++)
        {
            std::size_t outer_inner_idx = outer_idx * shape_dim * inner + inner_idx;
            std::size_t max_idx = 0;

            for (std::size_t dim_idx = 0; dim_idx < shape_dim; dim_idx++)
            {
                if (x->at(dim_idx * inner + outer_inner_idx) > x->at(max_idx * inner + outer_inner_idx))
                {
                    max_idx = dim_idx;
                }
            }

            out_data[outer_idx * inner + inner_idx] = static_cast<float>(max_idx);
        }
    }

    return std::make_shared<Tensor>(out_data, out_shape);
}