#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;
    std::vector<float> _grad;
    std::function<void(const std::vector<float> &)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;

public:
    explicit Tensor(float data, bool requires_grad = false, std::function<void(const std::vector<float> &)> gradfn = nullptr, std::vector<std::shared_ptr<Tensor>> parents = {});
    explicit Tensor(std::vector<float> data, bool requires_grad = false, std::function<void(const std::vector<float> &)> gradfn = nullptr,
                    std::vector<std::shared_ptr<Tensor>> parents = {});
    explicit Tensor(std::vector<std::vector<float>> data, bool requires_grad = false, std::function<void(const std::vector<float> &)> gradfn = nullptr,
                    std::vector<std::shared_ptr<Tensor>> parents = {});
    ~Tensor();

    float item() const;
    float &item();
    float operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    float operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);
    const std::vector<std::size_t> &shape() const;
    const std::vector<std::size_t> &stride() const;
    bool requires_grad() const;
    const std::vector<float> &grad() const;

    void add_to_grad(const std::vector<float> &grad_update);
    void zero_grad();
    std::size_t numel() const;

    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
    friend std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
    friend std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
};