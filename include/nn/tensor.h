#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    std::shared_ptr<std::vector<float>> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t offset_ = 0;

    std::vector<float> grad_;
    std::function<void(const std::vector<float> &)> gradfn_;
    std::vector<std::shared_ptr<Tensor>> parents_;
    bool requires_grad_;

    std::size_t flat_idx(const std::vector<std::size_t> &idx) const;
    std::ostream &printf(std::ostream &os, std::size_t dim, std::vector<std::size_t> &idx) const;

public:
    explicit Tensor(
        float data,
        bool requires_grad = false,
        std::function<void(const std::vector<float> &)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );
    explicit Tensor(
        std::vector<float> data,
        bool requires_grad = false,
        std::function<void(const std::vector<float> &)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );
    explicit Tensor(
        std::vector<std::vector<float>> data,
        bool requires_grad = false,
        std::function<void(const std::vector<float> &)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

    explicit Tensor(
        std::shared_ptr<std::vector<float>> storage,
        std::vector<std::size_t> shape,
        std::vector<std::size_t> stride,
        std::size_t offset = 0,
        bool requires_grad = false,
        std::function<void(const std::vector<float> &)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    );

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

    std::size_t offset() const;
    std::size_t numel() const;
    std::size_t ndim() const;

    bool requires_grad() const;
    const std::vector<float> &grad() const;

    void zero_grad();
    void add_to_grad(const std::vector<float> &grad_update);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
    friend std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
    friend std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
};