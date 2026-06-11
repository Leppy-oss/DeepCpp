#include "nn/modules/flatten.h"
#include "nn/tensor.h"
#include <memory>

Flatten::Flatten(std::size_t dim) : Module("Flatten"), dim_{dim} {}

Flatten::Flatten(std::string name, std::size_t dim) : Module(std::move(name)), dim_{dim} {}

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> x)
{
    if (x->ndim() == 0)
    {
        throw std::invalid_argument("Flatten requires tensor to be at least 1d");
    }

    if (dim_ >= x->ndim())
    {
        throw std::invalid_argument(
            "Dim " + std::to_string(dim_) + " invalid for tensor with " + std::to_string(x->ndim()) + " dimensions"
        );
    }

    tensor::Shape new_shape;
    for (std::size_t i = 0; i < dim_; i++)
    {
        new_shape.push_back(x->shape()[i]);
    }

    std::size_t flattened_size = 1;
    for (std::size_t i = dim_; i < x->shape().size(); i++)
    {
        flattened_size *= x->shape()[i];
    }

    new_shape.push_back(flattened_size);

    if (new_shape == x->shape())
    {
        return x;
    }

    return x->reshape(new_shape);
}