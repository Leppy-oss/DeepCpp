#pragma once
#include "data/dataset.h"
#include "tensor.h"
#include <memory>
#include <random>
#include <utility>

class DataLoader
{
private:
    Dataset *dataset_;
    int batch_size_;
    bool shuffle_;
    std::vector<std::size_t> idx_map_;
    std::mt19937 g_;

public:
    DataLoader(Dataset *dataset, int batch_size, bool shuffle = false);

    class Iterator
    {
    private:
        DataLoader *dataloader_;
        std::size_t idx_;

    public:
        Iterator(DataLoader *dataloader, std::size_t idx);
        void operator++();
        std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator*();
        bool operator!=(const Iterator &other);
    };

    Iterator begin();
    Iterator end();

    std::size_t batch_size() const;
    std::size_t length() const;
    std::size_t length_batches() const;
};