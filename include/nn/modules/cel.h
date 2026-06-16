#include "loss.h"
#include "nn/tensor.h"
#include <memory>

class Cel : public Loss
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y) override;
};