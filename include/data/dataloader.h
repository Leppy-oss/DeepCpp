#pragma once
#include "data/dataset.h"
#include "tensor.h"
#include <memory>
#include <utility>

class DataLoader
{
private:
    Dataset *dataset_;
    int batch_size_;
    std::vector<std::size_t> idx_map_;

public:
    DataLoader(Dataset *dataset, int batch_size, bool shuffle);

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