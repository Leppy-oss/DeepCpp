#include "data/dataloader.h"
#include "data/dataset.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

DataLoader::DataLoader(Dataset *dataset, int batch_size, bool shuffle) : dataset_{dataset}, batch_size_{batch_size}
{
    idx_map_.resize(dataset_->length());
    std::iota(idx_map_.begin(), idx_map_.end(), 0);

    if (shuffle)
    {
        std::shuffle(idx_map_.begin(), idx_map_.end(), std::mt19937(std::random_device()()));
    }
}

DataLoader::Iterator::Iterator(DataLoader *dataloader, std::size_t idx) : dataloader_{dataloader}, idx_{idx} {}

void DataLoader::Iterator::operator++() { idx_ += dataloader_->batch_size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::Iterator::operator*() {}