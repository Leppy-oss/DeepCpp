#include <iostream>
#include <nn/tensor.h>
#include <sstream>
#include <string>
#include <utils.h>
#include <vector>

namespace tensor
{
    std::vector<std::size_t> make_stride(const std::vector<std::size_t> &shape)
    {
        std::vector<std::size_t> stride(shape.size());
        std::size_t s = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
        {
            stride[i] = s;
            s *= shape[i];
        }
        return stride;
    }

    std::size_t numel_shape(const std::vector<std::size_t> shape)
    {
        std::size_t numel = 1;
        for (std::size_t dim : shape)
        {
            numel *= dim;
        }
        return numel;
    }

    std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t> &a, const std::vector<std::size_t> &b)
    {
        std::size_t max_dim = std::max(a.size(), b.size());
        std::vector<std::size_t> result(max_dim);
        for (int i = 0; i < max_dim; i++)
        {
            std::size_t from_dim = 1;
            std::size_t to_dim = 1;

            if (i >= max_dim - a.size())
            {
                from_dim = a[i - (max_dim - a.size())];
            }

            if (i >= max_dim - b.size())
            {
                to_dim = b[i - (max_dim - b.size())];
            }

            if (from_dim != to_dim && from_dim != 1 && to_dim != 1)
            {
                throw std::invalid_argument(
                    "Shapes " + utils::to_string(a) + " and " + utils::to_string(b) + " cannot be broadcast"
                );
            }

            result[i] = std::max(from_dim, to_dim);
        }
        return result;
    }

    std::size_t inv_broadcast_idx(
        std::size_t out_idx,
        const std::vector<std::size_t> &out_shape,
        const std::vector<std::size_t> &in_shape,
        const std::vector<std::size_t> &in_stride,
        std::size_t in_offset
    )
    {
        std::size_t in_ndim = in_shape.size();
        std::size_t out_ndim = out_shape.size();

        if (out_ndim < in_ndim)
        {
            throw std::invalid_argument(
                "in_ndim of " + std::to_string(in_ndim) + " greater than out_ndim of " + std::to_string(out_ndim)
            );
        }

        std::size_t in_idx = in_offset;

        for (std::size_t out_dim = out_ndim; out_dim-- > out_ndim - in_ndim;)
        {
            std::size_t out_dim_idx = out_idx % out_shape[out_dim];
            out_idx /= out_shape[out_dim];

            std::size_t in_dim = out_dim - (out_ndim - in_ndim);

            if (in_shape[in_dim] == 1)
            {
                out_dim_idx = 0;
            }
            in_idx += out_dim_idx * in_stride[in_dim];
        }

        return in_idx;
    }
}

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
    gradfn_{std::move(gradfn)},
    parents_(std::move(parents))
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
    gradfn_{std::move(gradfn)},
    parents_(std::move(parents))
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
    gradfn_(std::move(gradfn)),
    parents_(std::move(parents))
{
    if (!data.empty())
    {
        std::size_t n_expected_cols = data[0].size();
        for (const auto &row : data)
        {
            if (row.size() != n_expected_cols)
            {
                throw std::invalid_argument("Inconsistent dimensions for initializer data");
            }
            for (float d : row)
            {
                storage_->push_back(d);
            }
        }
    }
    if (requires_grad_)
    {
        zero_grad();
    }
}

Tensor::Tensor(
    std::vector<float> data,
    std::vector<std::size_t> shape,
    bool requires_grad,
    std::function<void(const std::vector<float> &)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
) :
    storage_{std::make_shared<std::vector<float>>(std::move(data))},
    shape_{std::move(shape)},
    stride_{tensor::make_stride(shape_)},
    offset_{0},
    requires_grad_{requires_grad},
    gradfn_{std::move(gradfn)},
    parents_{std::move(parents)}
{
    if (storage_->size() != tensor::numel_shape(shape_))
    {
        throw std::invalid_argument(
            "Number of elements in data (" + std::to_string(storage_->size()) + ") does not match shape " +
            utils::to_string(shape_)
        );
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

std::shared_ptr<std::vector<float>> Tensor::storage() const { return storage_; }
float Tensor::storage(std::size_t idx) const { return (*storage_)[idx]; }
float &Tensor::storage(std::size_t idx) { return (*storage_)[idx]; }

float Tensor::item() const
{
    if (numel() != 1)
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
    return (*storage_)[0];
}

float &Tensor::item()
{
    if (numel() != 1)
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
    return (*storage_)[0];
}

std::size_t Tensor::flat_idx(const std::vector<std::size_t> &idx) const
{
    if (idx.size() != ndim())
    {
        throw std::invalid_argument(
            "Cannot access index with " + std::to_string(idx.size()) + " dimensions for tensor with " +
            std::to_string(ndim()) + " dimensions"
        );
    }
    std::size_t flattened_idx = offset_;
    for (std::size_t dim = 0; dim < idx.size(); dim++)
    {
        if (idx[dim] >= shape_[dim])
        {
            throw std::out_of_range(
                "Index " + std::to_string(idx[dim]) + " out of bounds on dimension " + std::to_string(dim) +
                " (max value " + std::to_string(shape_[dim]) + ")"
            );
        }
        flattened_idx += idx[dim] * stride_[dim];
    }
    return flattened_idx;
}

float Tensor::operator()(const std::vector<std::size_t> &idx) const { return (*storage_)[flat_idx(idx)]; }
float &Tensor::operator()(const std::vector<std::size_t> &idx) { return (*storage_)[flat_idx(idx)]; }

const std::vector<std::size_t> &Tensor::shape() const { return shape_; }
const std::vector<std::size_t> &Tensor::stride() const { return stride_; }

std::size_t Tensor::idx_at_flat(std::size_t flat) const
{
    std::size_t idx = offset_;
    for (std::size_t dim = ndim(); dim-- > 0;)
    {
        std::size_t dim_idx = flat % shape_[dim];
        flat /= shape_[dim];

        idx += dim_idx * stride_[dim];
    }

    return idx;
}

float Tensor::at(std::size_t idx) const { return (*storage_)[idx_at_flat(idx)]; }
float &Tensor::at(std::size_t idx) { return (*storage_)[idx_at_flat(idx)]; }

std::size_t Tensor::offset() const { return offset_; }
std::size_t Tensor::numel() const { return tensor::numel_shape(shape_); }
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
        throw std::runtime_error(
            "Shape mismatch during gradient accumulation (" + std::to_string(grad_.size()) + " vs " +
            std::to_string(grad_update.size()) + ")"
        );
    }
    for (std::size_t i = 0; i < grad_.size(); i++)
    {
        grad_[i] += grad_update[i];
    }
}

std::ostream &Tensor::printf(std::ostream &os, std::size_t dim, std::vector<std::size_t> &idx) const
{
    os << "[";
    for (std::size_t i = 0; i < shape_[dim]; i++)
    {
        idx[dim] = i;
        if (dim >= ndim() - 1)
        {
            os << operator()(idx);
        }
        else
        {
            printf(os, dim + 1, idx);
        }
        if (i < shape_[dim] - 1)
        {
            if (dim >= ndim() - 1)
            {
                os << ", ";
            }
            else
            {
                os << ",\n       ";
                for (std::size_t j = 0; j <= dim; j++)
                {
                    os << " ";
                }
            }
        }
    }
    os << "]";
    return os;
}

std::shared_ptr<Tensor> Tensor::broadcast(const std::vector<std::size_t> &target_shape) const
{
    if (target_shape == shape_)
    {
        return std::const_pointer_cast<Tensor>(shared_from_this());
    }
    if (target_shape.size() < ndim())
    {
        throw std::invalid_argument(
            "Cannot broadcast to fewer dimensions (shape " + utils::to_string(shape_) + " to shape " +
            utils::to_string(target_shape) + ")"
        );
    }

    std::vector<std::size_t> out_stride(target_shape.size());

    for (std::size_t out_dim = target_shape.size(); out_dim-- > target_shape.size() - ndim();)
    {
        std::size_t in_dim = out_dim - (target_shape.size() - ndim());

        if (shape_[in_dim] == target_shape[out_dim])
        {
            out_stride[out_dim] = stride_[in_dim];
        }
        else if (shape_[in_dim] == 1)
        {
            out_stride[out_dim] = 0;
        }
        else
        {
            throw std::invalid_argument(
                "Tensor with shape " + utils::to_string(shape_) + " cannot be broadcast to shape " +
                utils::to_string(target_shape)
            );
        }
    }

    std::function<void(const std::vector<float> &)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents;

    if (requires_grad_)
    {
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto in_shape = shape_;
        auto out_shape = target_shape;
        auto in_stride = tensor::make_stride(shape_);
        parents = {self};

        gradfn = [self, in_shape, out_shape, in_stride](const std::vector<float> &grad_output)
        {
            std::vector<float> grad_update(self->numel());
            for (std::size_t out_idx = 0; out_idx < grad_output.size(); out_idx++)
            {
                std::size_t in_idx = tensor::inv_broadcast_idx(out_idx, out_shape, in_shape, in_stride, 0);
                grad_update[in_idx] += grad_output[out_idx];
            }
            self->add_to_grad(grad_update);
        };
    }

    return std::make_shared<Tensor>(storage_, target_shape, out_stride, offset_, requires_grad_, gradfn, parents);
}

std::shared_ptr<Tensor> bin_elementwise(
    std::shared_ptr<Tensor> t1,
    std::shared_ptr<Tensor> t2,
    std::function<float(float, float)> fwd_op,
    std::function<float(float, float, float)> grad_t1,
    std::function<float(float, float, float)> grad_t2
)
{
    auto out_shape = tensor::broadcast_shape(t1->shape(), t2->shape());

    auto b1 = t1->broadcast(out_shape);
    auto b2 = t2->broadcast(out_shape);

    std::size_t out_numel = tensor::numel_shape(out_shape);
    std::vector<float> out_data;
    out_data.reserve(out_numel);

    for (std::size_t out_idx = 0; out_idx < out_numel; out_idx++)
    {
        out_data.push_back(fwd_op(b1->at(out_idx), b2->at(out_idx)));
    }

    bool requires_grad = b1->requires_grad() || b2->requires_grad();
    std::function<void(const std::vector<float> &)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents;

    if (requires_grad)
    {
        parents = {b1, b2};
        gradfn = [b1, b2, grad_t1, grad_t2](const std::vector<float> &grad_output)
        {
            std::vector<float> grad_b1(grad_output.size());
            std::vector<float> grad_b2(grad_output.size());

            for (std::size_t idx = 0; idx < grad_output.size(); idx++)
            {
                grad_b1[idx] = grad_t1(b1->at(idx), b2->at(idx), grad_output[idx]);
                grad_b2[idx] = grad_t1(b1->at(idx), b2->at(idx), grad_output[idx]);
            }

            if (b1->requires_grad())
            {
                b1->add_to_grad(grad_b1);
            }

            if (b2->requires_grad())
            {
                b2->add_to_grad(grad_b2);
            }
        };
    }

    return std::make_shared<Tensor>(std::move(out_data), out_shape, requires_grad, gradfn, parents);
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    os << "tensor(";
    if (obj.shape().size() == 0)
    {
        os << obj.item();
    }
    else
    {
        auto idx = std::vector<std::size_t>(obj.shape().size());
        obj.printf(os, 0, idx);
    }
    os << ")";
    return os;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2)
{
    return bin_elementwise(
        t1,
        t2,
        [](float a, float b) { return a + b; },
        [](float a, float b, float g) { return g; },
        [](float a, float b, float g) { return g; }
    );
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2)
{
    return bin_elementwise(
        t1,
        t2,
        [](float a, float b) { return a * b; },
        [](float a, float b, float g) { return b * g; },
        [](float a, float b, float g) { return a * g; }
    );
}

// std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {}

// std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) { return ::matmul(shared_from_this(), other); }