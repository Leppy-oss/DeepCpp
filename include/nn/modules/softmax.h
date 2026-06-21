#include "module.h"
#include "nn/tensor.h"
#include <memory>

class Softmax : public Module
{
private:
    std::size_t dim_;

public:
    Softmax(std::size_t dim = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
};