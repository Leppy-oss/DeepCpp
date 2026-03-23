#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {};

Tensor::Tensor(std::vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {};

Tensor::Tensor(std::vector<std::vector<float>> data)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}
{
    std::size_t n_expected_cols = data[0].size();
    for (std::size_t i = 0; i < data.size(); i++)
    {
        if (data[i].size() != n_expected_cols)
            throw std::invalid_argument("Dimensions are inconsistent");
    }
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < data[i].size(); j++)
        {
            _data.push_back(data[i][j]);
        }
    }
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }

const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

const float &Tensor::item() const
{
    if (_data.size() != 1)
        throw std::runtime_error("item() can only be called on tensors with a single element");

    return _data[0];
}

float &Tensor::item() { return const_cast<float &>(static_cast<const Tensor *>(this)->item()); }

const float &Tensor::operator()(std::size_t i) const
{
    if (_shape.size() == 0)
        throw std::invalid_argument("Cannot index into scalar value, please use item() instead");
    if (_shape.size() != 1)
        throw std::invalid_argument("Dimensional mismatch between single index and non-1d tensor");
    if (i >= _shape[0])
    {
        throw std::invalid_argument("Index " + std::to_string(i) +
                                    " out of bounds for array of size " +
                                    std::to_string(_shape[0]));
    }
    return _data[i];
}

float &Tensor::operator()(std::size_t i)
{
    return const_cast<float &>(static_cast<const Tensor *>(this)->operator()(i));
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (_shape.size() != 2)
        throw std::invalid_argument("Dimensional mismatch between double index and non-2d tensor");
    if (i >= _shape[0])
        throw std::invalid_argument("Row index " + std::to_string(i) +
                                    " out of bounds for tensor with " + std::to_string(_shape[0]) +
                                    " rows");
    if (j >= _shape[1])
        throw std::invalid_argument("Col index " + std::to_string(i) +
                                    " out of bounds for tensor with " + std::to_string(_shape[0]) +
                                    " cols");

    return _data[i * _stride[0] + j * _stride[1]];
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    return const_cast<float &>(static_cast<const Tensor *>(this)->operator()(i, j));
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    std::string repr = "tensor(";
    if (obj.shape().size() == 0)
        repr += std::to_string(obj.item());
    else if (obj.shape().size() == 1)
    {
        for (std::size_t i = 0; i < obj.shape()[0] - 1; i++)
        {
            repr += std::to_string(obj(i)) + ", ";
        }
        if (obj.shape()[0] > 0)
            repr += std::to_string(obj(obj.shape()[0] - 1));
    }
    else
    {
        repr += "\n";
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            repr += "(";
            for (std::size_t j = 0; j < obj.shape()[1] - 1; j++)
            {
                repr += std::to_string(obj(i, j)) + ", ";
            }
            if (obj.shape()[1] > 0)
                repr += std::to_string(obj(i, obj.shape()[1] - 1)) + ")\n";
        }
    }
    repr += ")";
    os << repr;
    return os;
}