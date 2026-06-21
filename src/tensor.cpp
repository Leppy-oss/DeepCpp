#include "tensor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <utils.h>
#include <vector>

namespace tensor
{
    tensor::Stride make_stride(const tensor::Shape &shape)
    {
        tensor::Stride stride(shape.size());
        std::size_t s = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
        {
            stride[i] = s;
            s *= shape[i];
        }
        return stride;
    }

    std::size_t numel_shape(const tensor::Shape &shape)
    {
        std::size_t numel = 1;
        for (std::size_t dim : shape)
        {
            numel *= dim;
        }
        return numel;
    }

    tensor::Shape broadcast_shape(const tensor::Shape &a, const tensor::Shape &b)
    {
        std::size_t max_dim = std::max(a.size(), b.size());
        tensor::Shape result(max_dim);
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
        const tensor::Shape &out_shape,
        const tensor::Shape &in_shape,
        const tensor::Stride &in_stride,
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
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
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
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
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
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
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
    tensor::Shape shape,
    bool requires_grad,
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
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
    tensor::Shape shape,
    tensor::Stride stride,
    std::size_t offset,
    bool requires_grad,
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
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

std::shared_ptr<Tensor> Tensor::zeros(
    tensor::Shape shape,
    bool requires_grad,
    std::function<void(std::shared_ptr<Tensor>)> gradfn,
    std::vector<std::shared_ptr<Tensor>> parents
)
{
    return std::make_shared<Tensor>(
        std::vector<float>(tensor::numel_shape(shape)), shape, requires_grad, gradfn, parents
    );
}

void Tensor::load_data(std::shared_ptr<std::vector<float>> data)
{
    if (data->size() != storage_->size())
    {
        throw std::runtime_error(
            "Size mismatch when loading data (" + std::to_string(data->size()) + " vs " +
            std::to_string(storage_->size()) + ")"
        );
    }
    *storage_ = *data;
}

std::shared_ptr<Tensor> Tensor::deep_copy()
{
    std::vector<float> new_data;
    new_data.reserve(numel());
    for (float v : *storage_)
    {
        new_data.push_back(v);
    }
    return std::make_shared<Tensor>(std::make_shared<std::vector<float>>(new_data), shape_, stride_, offset_);
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
    return storage(0);
}

float &Tensor::item()
{
    if (numel() != 1)
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
    return storage(0);
}

std::size_t Tensor::storage_idx(const std::vector<std::size_t> &idx) const
{
    if (idx.size() != ndim())
    {
        throw std::invalid_argument(
            "Cannot access index with " + std::to_string(idx.size()) + " dimensions for tensor with " +
            std::to_string(ndim()) + " dimensions"
        );
    }
    std::size_t storage_idx = offset_;
    for (std::size_t dim = 0; dim < idx.size(); dim++)
    {
        if (idx[dim] >= shape_[dim])
        {
            throw std::out_of_range(
                "Index " + std::to_string(idx[dim]) + " out of bounds on dimension " + std::to_string(dim) +
                " (max value " + std::to_string(shape_[dim]) + ")"
            );
        }
        storage_idx += idx[dim] * stride_[dim];
    }
    return storage_idx;
}

float Tensor::operator()(const std::vector<std::size_t> &idx) const { return storage(storage_idx(idx)); }
float &Tensor::operator()(const std::vector<std::size_t> &idx) { return storage(storage_idx(idx)); }

const tensor::Shape &Tensor::shape() const { return shape_; }
const tensor::Stride &Tensor::stride() const { return stride_; }

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

float Tensor::at(std::size_t idx) const { return storage(idx_at_flat(idx)); }
float &Tensor::at(std::size_t idx) { return storage(idx_at_flat(idx)); }

std::size_t Tensor::offset() const { return offset_; }
std::size_t Tensor::numel() const { return tensor::numel_shape(shape_); }
std::size_t Tensor::ndim() const { return shape_.size(); }

bool Tensor::requires_grad() const { return requires_grad_; }
std::shared_ptr<Tensor> Tensor::grad() const { return grad_; }

void Tensor::zero_grad()
{
    if (!requires_grad_)
    {
        return;
    }
    grad_ = Tensor::zeros(shape_);
}
void Tensor::add_grad(std::shared_ptr<Tensor> grad_update)
{
    if (!requires_grad_)
    {
        return;
    }
    if (!grad_)
    {
        throw std::runtime_error("grad_ does not exist yet");
    }
    if (grad_->numel() != grad_update->numel())
    {
        throw std::runtime_error(
            "Shape mismatch during gradient accumulation (" + std::to_string(grad_->numel()) + " vs " +
            std::to_string(grad_update->numel()) + ")"
        );
    }
    for (std::size_t i = 0; i < grad_->numel(); i++)
    {
        grad_->at(i) += grad_update->at(i);
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

std::shared_ptr<Tensor> Tensor::broadcast(const tensor::Shape &target_shape) const
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

    tensor::Stride out_stride(target_shape.size());

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

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents;

    if (requires_grad_)
    {
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto in_shape = shape_;
        auto out_shape = target_shape;
        auto in_stride = tensor::make_stride(shape_);
        parents = {self};

        gradfn = [self, in_shape, out_shape, in_stride](std::shared_ptr<Tensor> grad_prev)
        {
            auto grad_update = Tensor::zeros(self->shape());
            for (std::size_t grad_idx = 0; grad_idx < grad_prev->numel(); grad_idx++)
            {
                std::size_t ten_idx = tensor::inv_broadcast_idx(grad_idx, out_shape, in_shape, in_stride, 0);
                grad_update->at(ten_idx) += grad_prev->at(grad_idx);
            }
            self->add_grad(grad_update);
        };
    }

    return std::make_shared<Tensor>(storage_, target_shape, out_stride, offset_, requires_grad_, gradfn, parents);
}

std::shared_ptr<Tensor> bin_elementwise(
    std::shared_ptr<Tensor> t1,
    std::shared_ptr<Tensor> t2,
    std::function<float(float, float)> fwd_op,
    std::function<float(float, float, float)> gradfn_t1,
    std::function<float(float, float, float)> gradfn_t2
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
    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents;

    if (requires_grad)
    {
        parents = {b1, b2};
        gradfn = [b1, b2, gradfn_t1, gradfn_t2](std::shared_ptr<Tensor> grad_prev)
        {
            auto g1 = Tensor::zeros(grad_prev->shape());
            auto g2 = Tensor::zeros(grad_prev->shape());

            for (std::size_t idx = 0; idx < grad_prev->numel(); idx++)
            {
                if (b1->requires_grad())
                {
                    g1->at(idx) = gradfn_t1(b1->at(idx), b2->at(idx), grad_prev->at(idx));
                }
                if (b2->requires_grad())
                {
                    g2->at(idx) = gradfn_t2(b1->at(idx), b2->at(idx), grad_prev->at(idx));
                }
            }

            if (b1->requires_grad())
            {
                b1->add_grad(g1);
            }

            if (b2->requires_grad())
            {
                b2->add_grad(g2);
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

std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2)
{
    if (t1->ndim() == 0 || t2->ndim() == 0)
    {
        throw std::invalid_argument("Matmul tensors must be at least 1d");
    }

    bool t1_vec = t1->ndim() == 1;
    bool t2_vec = t2->ndim() == 1;

    auto t1_shape = t1->shape();
    auto t2_shape = t2->shape();

    auto t1_stride = t1->stride();
    auto t2_stride = t2->stride();

    if (t1_vec)
    {
        t1_shape.insert(t1_shape.begin(), 1);
        t1_stride.insert(t1_stride.begin(), 0);
    }
    if (t2_vec)
    {
        t2_shape.push_back(1);
        t2_stride.push_back(0);
    }

    auto t1_batch_shape = tensor::Shape(t1_shape.begin(), t1_shape.end() - 2);
    auto t2_batch_shape = tensor::Shape(t2_shape.begin(), t2_shape.end() - 2);

    auto batch_shape = tensor::broadcast_shape(t1_batch_shape, t2_batch_shape);

    std::size_t M = t1_shape[t1_shape.size() - 2];
    std::size_t N1 = t1_shape[t1_shape.size() - 1];
    std::size_t N2 = t2_shape[t2_shape.size() - 2];
    std::size_t K = t2_shape[t2_shape.size() - 1];

    if (N1 != N2)
    {
        throw std::invalid_argument(
            "Inner dimensions of matmul tensors must match (found " + std::to_string(N1) + " and " +
            std::to_string(N2) + ")"
        );
    }

    tensor::Shape output_shape = batch_shape;
    output_shape.push_back(M);
    output_shape.push_back(K);

    if (t1_vec)
    {
        output_shape.erase(output_shape.end() - 2);
    }
    if (t2_vec)
    {
        output_shape.erase(output_shape.end() - 1);
    }

    std::vector<float> output_data;
    output_data.reserve(tensor::numel_shape(output_shape));

    auto t1_batch_stride = tensor::Stride(t1_stride.begin(), t1_stride.end() - 2);
    auto t2_batch_stride = tensor::Stride(t2_stride.begin(), t2_stride.end() - 2);

    std::size_t num_batches = tensor::numel_shape(batch_shape);

    for (std::size_t batch_idx = 0; batch_idx < num_batches; batch_idx++)
    {
        std::size_t t1_batch_idx =
            tensor::inv_broadcast_idx(batch_idx, batch_shape, t1_batch_shape, t1_batch_stride, t1->offset());
        std::size_t t2_batch_idx =
            tensor::inv_broadcast_idx(batch_idx, batch_shape, t2_batch_shape, t2_batch_stride, t2->offset());

        for (std::size_t i = 0; i < M; i++)
        {
            for (std::size_t j = 0; j < K; j++)
            {
                float dot_product = 0.0f;
                for (std::size_t k = 0; k < N1; k++)
                {
                    std::size_t t1_ik =
                        t1_batch_idx + i * t1_stride[t1_stride.size() - 2] + k * t1_stride[t1_stride.size() - 1];
                    std::size_t t2_kj =
                        t2_batch_idx + k * t2_stride[t2_stride.size() - 2] + j * t2_stride[t2_stride.size() - 1];
                    dot_product += t1->storage(t1_ik) * t2->storage(t2_kj);
                }
                output_data.push_back(dot_product);
            }
        }
    }

    bool requires_grad = t1->requires_grad() || t2->requires_grad();
    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (requires_grad)
    {
        parents = {t1, t2};

        auto g1_batch_stride = tensor::make_stride(t1_batch_shape);
        auto g2_batch_stride = tensor::make_stride(t2_batch_shape);

        gradfn = [=](std::shared_ptr<Tensor> grad_prev)
        {
            auto g1 = Tensor::zeros(t1->shape());
            auto g2 = Tensor::zeros(t2->shape());

            for (std::size_t batch_idx = 0; batch_idx < num_batches; batch_idx++)
            {
                std::size_t t1_batch_idx =
                    tensor::inv_broadcast_idx(batch_idx, batch_shape, t1_batch_shape, t1_batch_stride, t1->offset());
                std::size_t t2_batch_idx =
                    tensor::inv_broadcast_idx(batch_idx, batch_shape, t2_batch_shape, t2_batch_stride, t2->offset());
                std::size_t g1_batch_idx =
                    tensor::inv_broadcast_idx(batch_idx, batch_shape, t1_batch_shape, g1_batch_stride, 0);
                std::size_t g2_batch_idx =
                    tensor::inv_broadcast_idx(batch_idx, batch_shape, t2_batch_shape, g2_batch_stride, 0);

                for (std::size_t i = 0; i < M; i++)
                {
                    for (std::size_t j = 0; j < K; j++)
                    {
                        float dy_ij = grad_prev->at(batch_idx * M * K + i * K + j);

                        for (std::size_t k = 0; k < N1; k++)
                        {
                            std::size_t t1_ik = t1_batch_idx + i * t1_stride[t1_stride.size() - 2] +
                                                k * t1_stride[t1_stride.size() - 1];
                            std::size_t t2_kj = t2_batch_idx + k * t2_stride[t2_stride.size() - 2] +
                                                j * t2_stride[t2_stride.size() - 1];

                            if (t1->requires_grad())
                            {
                                std::size_t g1_ik = t1_vec ? k : g1_batch_idx * M * N1 + i * N1 + k;
                                g1->at(g1_ik) += dy_ij * t2->storage(t2_kj);
                            }

                            if (t2->requires_grad())
                            {
                                std::size_t g2_kj = t2_vec ? k : g2_batch_idx * N1 * K + k * K + j;
                                g2->at(g2_kj) += dy_ij * t1->storage(t1_ik);
                            }
                        }
                    }
                }
            }

            if (t1->requires_grad())
            {
                t1->add_grad(g1);
            }
            if (t2->requires_grad())
            {
                t2->add_grad(g2);
            }
        };
    }

    return std::make_shared<Tensor>(output_data, output_shape, requires_grad, gradfn, parents);
}

std::shared_ptr<Tensor> Tensor::mm(std::shared_ptr<Tensor> other) { return ::mm(shared_from_this(), other); }

std::shared_ptr<Tensor> Tensor::squeeze(const std::size_t dim)
{
    if (dim >= ndim())
    {
        throw std::invalid_argument(
            "Squeeze dimension " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim()) +
            " dimensions"
        );
    }
    if (shape_[dim] != 1)
    {
        throw std::invalid_argument(
            "Shape is not 1 at dimension " + std::to_string(dim) + " (found " + std::to_string(shape_[dim]) + ")"
        );
    }
    auto out_shape = shape_;
    auto out_stride = stride_;
    out_shape.erase(out_shape.begin() + dim);
    out_stride.erase(out_stride.begin() + dim);

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (requires_grad_)
    {
        auto self = shared_from_this();
        gradfn = [self, dim](std::shared_ptr<Tensor> grad_prev) { self->add_grad(grad_prev->unsqueeze(dim)); };
        parents = {self};
    };

    return std::make_shared<Tensor>(storage_, out_shape, out_stride, offset_, requires_grad_, gradfn, parents);
}

std::shared_ptr<Tensor> Tensor::unsqueeze(const std::size_t dim)
{
    if (dim > ndim())
    {
        throw std::invalid_argument(
            "Unsqueeze dimension " + std::to_string(dim) + " is out of bounds for tensor with " +
            std::to_string(ndim()) + " dimensions"
        );
    }
    auto out_shape = shape_;
    auto out_stride = stride_;
    out_shape.insert(out_shape.begin() + dim, 1);
    out_stride.insert(out_stride.begin() + dim, dim == ndim() ? 1 : stride_[dim]);

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (requires_grad_)
    {
        auto self = shared_from_this();
        gradfn = [self, dim](std::shared_ptr<Tensor> grad_prev) { self->add_grad(grad_prev->squeeze(dim)); };
        parents = {self};
    };

    return std::make_shared<Tensor>(storage_, out_shape, out_stride, offset_, requires_grad_, gradfn, parents);
}

bool Tensor::is_contiguous() const { return offset_ == 0 && stride_ == tensor::make_stride(shape_); }

std::shared_ptr<Tensor> Tensor::contiguous()
{
    if (is_contiguous())
    {
        return shared_from_this();
    }

    std::vector<float> new_data;
    new_data.resize(numel());

    for (std::size_t out_idx = 0; out_idx < numel(); out_idx++)
    {
        new_data[out_idx] = storage(idx_at_flat(out_idx));
    }

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (requires_grad_)
    {
        auto self = shared_from_this();
        parents = {self};
        gradfn = [self](std::shared_ptr<Tensor> grad_prev) { self->add_grad(grad_prev->contiguous()); };
    }

    return std::make_shared<Tensor>(new_data, shape_, requires_grad_, gradfn, parents);
}

std::shared_ptr<Tensor> Tensor::view(const tensor::Shape &shape)
{
    if (tensor::numel_shape(shape) != numel())
    {
        throw std::invalid_argument(
            "Shape " + utils::to_string(shape) + " is not valid for tensor with " + std::to_string(numel()) +
            " elements"
        );
    }

    if (!is_contiguous())
    {
        throw std::runtime_error("Tensor is not contiguous");
    }

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (requires_grad_)
    {
        auto self = shared_from_this();
        parents = {self};
        gradfn = [self](std::shared_ptr<Tensor> grad_prev) { self->add_grad(grad_prev->view(self->shape())); };
    }

    return std::make_shared<Tensor>(
        storage_, shape, tensor::make_stride(shape), offset_, requires_grad_, gradfn, parents
    );
}

std::shared_ptr<Tensor> Tensor::reshape(const tensor::Shape &shape)
{
    if (tensor::numel_shape(shape) != numel())
    {
        throw std::invalid_argument(
            "Shape " + utils::to_string(shape) + " is not valid for tensor with " + std::to_string(numel()) +
            " elements"
        );
    }

    if (is_contiguous())
    {
        return view(shape);
    }

    return contiguous()->view(shape);
}