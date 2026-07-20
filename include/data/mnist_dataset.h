#pragma once
#include "data/dataset.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

class MNIST : public Dataset<MNIST>
{
private:
    std::vector<std::vector<std::vector<float>>> imgs_;
    std::vector<std::size_t> labels_;
    std::pair<std::shared_ptr<Tensor>, std::size_t> get_item_(std::size_t idx);

public:
    MNIST(std::string path);
};