#pragma once
#include "tensor.h"
#include <memory>
#include <utility>

class Dataset
{
public:
    virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_item(std::size_t idx);
    virtual std::size_t length() = 0;
};