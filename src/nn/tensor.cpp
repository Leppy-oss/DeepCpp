#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(
    float data,
    bool requires_grad,
    std::function<void(const std::vector<float> &)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
) :
    storage_{std::make_shared<std::vector<float>>(std::initializer_list<float>{data})},
    shape_{},
    stride_{},
    offset_{0},
    requires_grad_{requires_grad},
    gradfn_{gradfn},
    parents_(parents)
{
    if (requires_grad_)
    {
        zero_grad();
    }
};

Tensor::Tensor(
    std::vector<float> data,
    bool requires_grad,
    std::function<void(const std::vector<float> &)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
) :
    storage_{std::make_shared<std::vector<float>>(std::move(data))},
    shape_{storage_->size()},
    stride_{1},
    offset_{0},
    requires_grad_{requires_grad},
    gradfn_{gradfn},
    parents_(parents)
{
    if (requires_grad_)
    {
        zero_grad();
    }
};

Tensor::Tensor(
    std::vector<std::vector<float>> data,
    bool requires_grad,
    std::function<void(const std::vector<float> &)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
) :
    storage_{std::make_shared<std::vector<float>>()},
    shape_{data.size(), data.empty() ? 0 : data[0].size()},
    stride_{data.empty() ? 0 : data[0].size(), 1},
    offset_{0},
    requires_grad_(requires_grad),
    gradfn_(gradfn),
    parents_(parents)
{
    if (!data.empty())
    {
        std::size_t n_expected_cols = data[0].size();
        for (const auto &row : data)
        {
            if (row.size() != n_expected_cols)
            {
                throw std::invalid_argument("Inconsistent dimensions");
                for (float d : row)
                {
                    storage_->push_back(d);
                }
            }
        }
    }
    if (requires_grad_)
    {
        zero_grad();
    }
}

Tensor::Tensor(
    std::shared_ptr<std::vector<float>> storage,
    std::vector<std::size_t> shape,
    std::vector<std::size_t> stride,
    std::size_t offset,
    bool requires_grad,
    std::function<void(const std::vector<float> &)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
) :
    storage_{std::move(storage)},
    shape_(std::move(shape)),
    stride_(std::move(stride)),
    offset_{offset},
    requires_grad_{requires_grad},
    gradfn_{std::move(gradfn)},
    parents_(std::move(parents))
{
    if (shape_.size() != stride_.size())
    {
        throw std::invalid_argument("Shape and stride must have the same rank");
    }
    if (requires_grad_)
    {
        zero_grad();
    }
}

float Tensor::item() const
{
    if (storage_->size() != 1)
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }

    return (*storage_)[0];
}

float &Tensor::item()
{
    if (storage_->size() != 1)
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }

    return (*storage_)[0];
}

float &Tensor::operator()(std::vector<std::size_t> idx) { return (*storage_)[flat_idx(idx)]; }

float Tensor::operator()(std::size_t i) const
{
    if (shape_.size() == 0)
    {
        throw std::invalid_argument("Cannot index into scalar value, please use item() instead");
    }
    if (shape_.size() != 1)
    {
        throw std::invalid_argument("Dimensional mismatch between single index and non-1d tensor");
    }
    if (i >= shape_[0])
    {
        throw std::invalid_argument(
            "Index " + std::to_string(i) + " out of bounds for array of size " + std::to_string(shape_[0])
        );
    }
    return storage_[i];
}

float &Tensor::operator()(std::size_t i)
{
    if (shape_.size() == 0)
    {
        throw std::invalid_argument("Cannot index into scalar value, please use item() instead");
    }
    if (shape_.size() != 1)
    {
        throw std::invalid_argument("Dimensional mismatch between single index and non-1d tensor");
    }
    if (i >= shape_[0])
    {
        throw std::invalid_argument(
            "Index " + std::to_string(i) + " out of bounds for array of size " + std::to_string(shape_[0])
        );
    }
    return storage_[i];
}

float Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (shape_.size() != 2)
    {
        throw std::invalid_argument("Dimensional mismatch between double index and non-2d tensor");
    }
    if (i >= shape_[0])
    {
        throw std::invalid_argument(
            "Row index " + std::to_string(i) + " out of bounds for tensor with " + std::to_string(shape_[0]) + " rows"
        );
    }
    if (j >= shape_[1])
    {
        throw std::invalid_argument(
            "Col index " + std::to_string(j) + " out of bounds for tensor with " + std::to_string(shape_[1]) + " cols"
        );
    }

    return storage_[i * stride_[0] + j * stride_[1]];
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    if (shape_.size() != 2)
    {
        throw std::invalid_argument("Dimensional mismatch between double index and non-2d tensor");
    }
    if (i >= shape_[0])
    {
        throw std::invalid_argument(
            "Row index " + std::to_string(i) + " out of bounds for tensor with " + std::to_string(shape_[0]) + " rows"
        );
    }
    if (j >= shape_[1])
    {
        throw std::invalid_argument(
            "Col index " + std::to_string(j) + " out of bounds for tensor with " + std::to_string(shape_[1]) + " cols"
        );
    }

    return storage_[i * stride_[0] + j * stride_[1]];
}

const std::vector<std::size_t> &Tensor::shape() const { return shape_; }
const std::vector<std::size_t> &Tensor::stride() const { return stride_; }

std::size_t Tensor::offset() const { return offset_; }
std::size_t Tensor::numel() const { return storage_->size(); }
std::size_t Tensor::ndim() const { return shape_.size(); }

bool Tensor::requires_grad() const { return requires_grad_; }
const std::vector<float> &Tensor::grad() const { return grad_; }

void Tensor::zero_grad() { grad_ = std::vector<float>(numel()); }
void Tensor::add_to_grad(const std::vector<float> &grad_update)
{
    if (!requires_grad_)
    {
        return;
    }
    if (grad_.size() != grad_update.size())
    {
        throw std::runtime_error("Shape mismatch during gradient accumulation");
    }
    for (std::size_t i = 0; i < grad_.size(); i++)
    {
        grad_[i] += grad_update[i];
    }
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    std::string repr = "tensor(";
    if (obj.shape().size() == 0)
    {
        repr += std::to_string(obj.item());
    }
    else if (obj.shape().size() == 1)
    {
        for (std::size_t i = 0; i < obj.shape()[0] - 1; i++)
        {
            repr += std::to_string(obj(i)) + ", ";
        }
        if (obj.shape()[0] > 0)
        {
            repr += std::to_string(obj(obj.shape()[0] - 1));
        }
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
            {
                repr += std::to_string(obj(i, obj.shape()[1] - 1)) + ")\n";
            }
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
        std::vector<std::shared_ptr<Tensor>> parents{t1, t2};
        auto gradfn = [t1, t2](const std::vector<float> &grad_output)
        {
            if (t1->requires_grad())
            {
                t1->add_to_grad({grad_output[0]});
            }
            if (t2->requires_grad())
            {
                t2->add_to_grad({grad_output[0]});
            }
        };
        return std::make_shared<Tensor>(result, t1->requires_grad() || t2->requires_grad(), gradfn, parents);
    }
    if (t1->shape().size() == 0 && t2->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < t2->shape()[0]; i++)
        {
            result.push_back(t1->item() + ((*t2)(i)));
        }
        std::vector<std::shared_ptr<Tensor>> parents{t1, t2};
        auto gradfn = [t1, t2](const std::vector<float> &grad_output)
        {
            if (t1->requires_grad())
            {
                float t1_grad = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    t1_grad += grad_output[i];
                }
                t1->add_to_grad({t1_grad});
            }
            if (t2->requires_grad())
            {
                t2->add_to_grad(grad_output);
            }
        };
        return std::make_shared<Tensor>(result, t1->requires_grad() || t2->requires_grad(), gradfn, parents);
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
        throw std::invalid_argument(
            "Tensors with size " + std::to_string(t1->shape()[0]) + " and " + std::to_string(t2->shape()[0]) +
            " along dimension 0 cannot be added"
        );
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
        throw std::invalid_argument(
            "Tensors with size " + std::to_string(t1->shape()[1]) + " and " + std::to_string(t2->shape()[1]) +
            " along dimension 1 cannot be added"
        );
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
    throw std::invalid_argument(
        "Tensor with " + std::to_string(t1->shape().size()) + " dimensions cannot be added to tensor with " +
        std::to_string(t2->shape().size()) + " dimensions"
    );
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
        throw std::invalid_argument(
            "Last dimension of first tensor (" + std::to_string(t1->shape()[t1->shape().size() - 1]) +
            ") does not match first dimension of second tensor (" + std::to_string(t2->shape()[0]) + ")"
        );
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