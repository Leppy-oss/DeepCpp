#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace tensor
{
    std::vector<std::size_t> make_stride(const std::vector<std::size_t> &shape);
    std::size_t numel_shape(const std::vector<std::size_t> shape);
    std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t> &a, const std::vector<std::size_t> &b);

    std::size_t inv_broadcast_idx(
        std::size_t out_idx,
        const std::vector<std::size_t> &out_shape,
        const std::vector<std::size_t> &in_shape,
        const std::vector<std::size_t> &in_stride,
        std::size_t in_offset
    );
}

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    std::shared_ptr<std::vector<float>> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t offset_ = 0;

    std::shared_ptr<Tensor> grad_ = nullptr;
    std::function<void(std::shared_ptr<Tensor>)> gradfn_;
    std::vector<std::shared_ptr<Tensor>> parents_;
    bool requires_grad_;

    std::size_t flat_idx(const std::vector<std::size_t> &idx) const;
    std::ostream &printf(std::ostream &os, std::size_t dim, std::vector<std::size_t> &idx) const;

public:
    explicit Tensor(
        float data,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );
    explicit Tensor(
        std::vector<float> data,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );
    explicit Tensor(
        std::vector<std::vector<float>> data,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

    explicit Tensor(
        std::vector<float> data,
        std::vector<std::size_t> shape,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

    explicit Tensor(
        std::shared_ptr<std::vector<float>> storage,
        std::vector<std::size_t> shape,
        std::vector<std::size_t> stride,
        std::size_t offset = 0,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

    static std::shared_ptr<Tensor> zeros(
        std::vector<std::size_t> shape,
        bool requires_grad = false,
        std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

    void load_data(std::shared_ptr<std::vector<float>> data);
    std::shared_ptr<Tensor> deep_copy();

    std::shared_ptr<std::vector<float>> storage() const;
    float storage(std::size_t idx) const;
    float &storage(std::size_t idx);

    float item() const;
    float &item();

    float operator()(const std::vector<std::size_t> &idx) const;
    float &operator()(const std::vector<std::size_t> &idx);

    template <typename... Args>
        requires(std::convertible_to<Args, std::size_t> && ...)
    float operator()(Args... idx) const
    {
        return operator()({static_cast<std::size_t>(idx)...});
    }

    template <typename... Args>
        requires(std::convertible_to<Args, std::size_t> && ...)
    float &operator()(Args... idx)
    {
        return operator()({static_cast<std::size_t>(idx)...});
    };

    const std::vector<std::size_t> &shape() const;
    const std::vector<std::size_t> &stride() const;

    std::size_t idx_at_flat(std::size_t flat) const;
    float at(std::size_t idx) const;
    float &at(std::size_t idx);

    std::size_t offset() const;
    std::size_t numel() const;
    std::size_t ndim() const;

    bool requires_grad() const;
    std::shared_ptr<Tensor> grad() const;

    void zero_grad();
    void add_grad(std::shared_ptr<Tensor> grad_update);

    std::shared_ptr<Tensor> broadcast(const std::vector<std::size_t> &target_shape) const;

    friend std::shared_ptr<Tensor> bin_elementwise(
        std::shared_ptr<Tensor> t1,
        std::shared_ptr<Tensor> t2,
        std::function<float(float, float)> fwd_op,
        std::function<float(float, float, float)> grad_t1,
        std::function<float(float, float, float)> grad_t2
    );

    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
    friend std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
    friend std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);

    friend std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
    std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> other);

    std::shared_ptr<Tensor> squeeze(std::size_t dim);
    std::shared_ptr<Tensor> unsqueeze(std::size_t dim);
};