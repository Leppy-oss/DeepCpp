#pragma once
#include "data/dataset.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

class MNISTDataset : public Dataset
{
private:
    std::vector<std::vector<std::vector<float>>> imgs_;
    std::vector<std::size_t> labels_;

public:
    MNISTDataset(const std::string &path);

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_item(std::size_t idx) override;
    std::size_t length() override;
};