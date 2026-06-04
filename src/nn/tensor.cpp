#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(float data, bool requires_grad, std::function<void(const std::vector<float> &)> gradfn, std::vector<std::shared_ptr<Tensor>> parents)
    : _data{data}, _shape{}, _stride{}, _requires_grad(requires_grad), _gradfn(gradfn), _parents(parents)
{
    if (_requires_grad)
    {
        zero_grad();
    }
};

Tensor::Tensor(std::vector<float> data, bool requires_grad, std::function<void(const std::vector<float> &)> gradfn, std::vector<std::shared_ptr<Tensor>> parents)
    : _data(data), _shape{data.size()}, _stride{1}, _requires_grad(requires_grad), _gradfn(gradfn), _parents(parents)
{
    if (_requires_grad)
    {
        zero_grad();
    }
};

Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad, std::function<void(const std::vector<float> &)> gradfn, std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}, _requires_grad(requires_grad), _gradfn(gradfn), _parents(parents)
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
    if (_requires_grad)
    {
        zero_grad();
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
        throw std::invalid_argument("Index " + std::to_string(i) + " out of bounds for array of size " + std::to_string(_shape[0]));
    }
    return _data[i];
}

float &Tensor::operator()(std::size_t i) { return const_cast<float &>(static_cast<const Tensor *>(this)->operator()(i)); }

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (_shape.size() != 2)
        throw std::invalid_argument("Dimensional mismatch between double index and non-2d tensor");
    if (i >= _shape[0])
        throw std::invalid_argument("Row index " + std::to_string(i) + " out of bounds for tensor with " + std::to_string(_shape[0]) + " rows");
    if (j >= _shape[1])
        throw std::invalid_argument("Col index " + std::to_string(j) + " out of bounds for tensor with " + std::to_string(_shape[1]) + " cols");

    return _data[i * _stride[0] + j * _stride[1]];
}

float &Tensor::operator()(std::size_t i, std::size_t j) { return const_cast<float &>(static_cast<const Tensor *>(this)->operator()(i, j)); }

bool Tensor::requires_grad() const { return _requires_grad; }
const std::vector<float> &Tensor::grad() const { return _grad; }

void Tensor::add_to_grad(const std::vector<float> &grad_update)
{
    if (!_requires_grad)
    {
        return;
    }
    if (_grad.size() != grad_update.size())
    {
        throw std::runtime_error("Shape mismatch during gradient accumulation");
    }
    for (std::size_t i = 0; i < _grad.size(); i++)
    {
        _grad[i] += grad_update[i];
    }
}

void Tensor::zero_grad() { _grad = std::vector<float>(_data.size()); }

std::size_t Tensor::numel() const { return _data.size(); }

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

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2)
{
    if (t1->shape().size() == 0 && t2->shape().size() == 0)
    {
        float result = t1->item() + t2->item();
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 0 && t2->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t2->shape()[0]; i++)
        {
            result.push_back(t1->item() + ((*t2)(i)));
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 0 && t2->shape().size() == 2)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t2->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < t2->shape()[1]; j++)
            {
                result.push_back(t1->item() + (*t2)(i, j));
            }
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 1 && t2->shape().size() == 0)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            result.push_back((*t1)(i) + t2->item());
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 2 && t2->shape().size() == 0)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < t1->shape()[1]; j++)
            {
                result.push_back((*t1)(i, j) + t2->item());
            }
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape()[0] != t2->shape()[0])
    {
        throw std::invalid_argument("Tensors with size " + std::to_string(t1->shape()[0]) + " and " + std::to_string(t2->shape()[0]) + " along dimension 0 cannot be added");
    }
    if (t1->shape().size() == 1 && t2->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            result.push_back((*t1)(i) + (*t2)(i));
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape()[1] != t2->shape()[1])
    {
        throw std::invalid_argument("Tensors with size " + std::to_string(t1->shape()[1]) + " and " + std::to_string(t2->shape()[1]) + " along dimension 1 cannot be added");
    }
    if (t1->shape().size() == 2 && t2->shape().size() == 2)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < t1->shape()[1]; j++)
            {
                result.push_back((*t1)(i, j) + (*t2)(i, j));
            }
        }
        return std::make_shared<Tensor>(result);
    }
    throw std::invalid_argument("Tensor with " + std::to_string(t1->shape().size()) + " dimensions cannot be added to tensor with " + std::to_string(t2->shape().size()) +
                                " dimensions");
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2)
{
    if (t1->shape().size() == 0)
    {
        if (t2->shape().size() == 0)
        {
            return std::make_shared<Tensor>(t1->item() * t2->item());
        }
        if (t2->shape().size() == 1)
        {
            std::vector<float> result;
            for (std::size_t i = 0; i < t2->shape()[0]; i++)
            {
                result.push_back(t1->item() * (*t2)(i));
            }
            return std::make_shared<Tensor>(result);
        }
        std::vector<float> result;
        for (std::size_t i = 0; i < t2->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < t2->shape()[1]; j++)
            {
                result.push_back(t1->item() * (*t2)(i, j));
            }
        }
        return std::make_shared<Tensor>(result);
    }
    if (t2->shape().size() == 0)
    {
        if (t1->shape().size() == 1)
        {
            std::vector<float> result;
            for (std::size_t i = 0; i < t1->shape()[0]; i++)
            {
                result.push_back(t2->item() * (*t1)(i));
            }
            return std::make_shared<Tensor>(result);
        }
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < t1->shape()[1]; j++)
            {
                result.push_back(t2->item() * (*t1)(i, j));
            }
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape()[t1->shape().size() - 1] != t2->shape()[0])
    {
        throw std::invalid_argument("Last dimension of first tensor (" + std::to_string(t1->shape()[t1->shape().size() - 1]) +
                                    ") does not match first dimension of second tensor (" + std::to_string(t2->shape()[0]) + ")");
    }
    if (t1->shape().size() == 1 && t2->shape().size() == 1)
    {
        float result = 0;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            result += (*t1)(i) * (*t2)(i);
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 2 && t2->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t1->shape()[0]; i++)
        {
            float acc = 0;
            for (std::size_t j = 0; j < t1->shape()[1]; j++)
            {
                acc += (*t1)(i, j) * (*t2)(j);
            }
            result.push_back(acc);
        }
        return std::make_shared<Tensor>(result);
    }
    if (t1->shape().size() == 1 && t2->shape().size() == 2)
    {
        std::vector<float> result;
        for (std::size_t j = 0; j < t2->shape()[1]; j++)
        {
            float acc = 0;
            for (std::size_t i = 0; i < t2->shape()[0]; i++)
            {
                acc += (*t2)(i, j) * (*t1)(i);
            }
            result.push_back(acc);
        }
        return std::make_shared<Tensor>(result);
    }
    std::vector<std::vector<float>> result;
    for (std::size_t i = 0; i < t1->shape()[0]; i++)
    {
        std::vector<float> result_i;
        for (std::size_t j = 0; j < t2->shape()[1]; j++)
        {
            float acc = 0;
            for (std::size_t k = 0; k < t1->shape()[0]; k++)
            {
                acc += (*t1)(i, k) * (*t2)(k, j);
            }
            result_i.push_back(acc);
        }
        result.push_back(result_i);
    }
    return std::make_shared<Tensor>(result);
}