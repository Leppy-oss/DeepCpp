#pragma once
#include "loss.h"
#include "tensor.h"
#include <memory>

class Cel : public Loss
{
private:
    std::size_t dim_;

public:
    Cel(std::size_t dim = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y) override;
};