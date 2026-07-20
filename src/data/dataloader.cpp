#include "data/dataloader.h"
#include "data/dataset.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

DataLoader::DataLoader(Dataset *dataset, int batch_size, bool shuffle) :
    dataset_{dataset},
    batch_size_{batch_size},
    shuffle_{shuffle},
    g_{std::mt19937(std::random_device()())}
{
    idx_map_.resize(length());
    std::iota(idx_map_.begin(), idx_map_.end(), 0);
}

DataLoader::Iterator::Iterator(DataLoader *dataloader, std::size_t idx) : dataloader_{dataloader}, idx_{idx} {}

void DataLoader::Iterator::operator++() { idx_ += dataloader_->batch_size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::Iterator::operator*()
{
    if (idx_ >= dataloader_->length())
    {
        throw std::out_of_range("Attempting to iterate on DataLoader after reaching its end");
    }
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> targets;

    for (std::size_t i = idx_; i < std::min(idx_ + dataloader_->batch_size(), dataloader_->length()); i++)
    {
        auto sample = dataloader_->dataset_->get_item(dataloader_->idx_map_[i]);
        inputs.push_back(sample.first);
        targets.push_back(sample.second);
    }

    return std::make_pair(Tensor::stack(inputs), Tensor::stack(targets));
}

bool DataLoader::Iterator::operator!=(const DataLoader::Iterator &other) { return idx_ != other.idx_; }

DataLoader::Iterator DataLoader::begin()
{
    if (shuffle_)
    {
        std::shuffle(idx_map_.begin(), idx_map_.end(), g_);
    }
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() { return Iterator(this, dataset_->length()); }

std::size_t DataLoader::batch_size() const { return batch_size_; }
std::size_t DataLoader::length() const { return dataset_->length(); }
std::size_t DataLoader::length_batches() const { return (dataset_->length() + batch_size_ - 1) / batch_size_; }