#pragma once
#include "tensor.h"
#include <memory>
#include <utility>

template <typename D> class Dataset
{
public:
    auto get_item(std::size_t idx) { return static_cast<D *>(this)->get_item_(idx); }
    virtual std::size_t length() = 0;
};