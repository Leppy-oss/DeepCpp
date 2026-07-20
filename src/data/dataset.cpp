#include "data/dataset.h"
#include "tensor.h"
#include <memory>
#include <utility>

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> Dataset::get_item(std::size_t idx)
{
    throw std::runtime_error("Cannot call get_item on base Dataset class");
}