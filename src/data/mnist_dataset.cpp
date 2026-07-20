#include "data/mnist_dataset.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

uint32_t reverse_endian(uint32_t val)
{
    return ((val << 24) & 0xFF000000) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

std::vector<std::vector<std::vector<float>>> load_images(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file '" + path + "'");
    }

    uint32_t magic_number = 0, len = 0, rows = 0, cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = reverse_endian(magic_number);
    if (magic_number != 2051)
    {
        throw std::runtime_error("File '" + path + "' is not in proper MNIST data format");
    }
    file.read(reinterpret_cast<char *>(&len), 4);
    len = reverse_endian(len);
    file.read(reinterpret_cast<char *>(&rows), 4);
    rows = reverse_endian(rows);
    file.read(reinterpret_cast<char *>(&cols), 4);
    cols = reverse_endian(cols);

    std::cout << magic_number << ", " << len << ", " << rows << ", " << cols << std::endl;
    return {};
}

std::vector<std::size_t> load_labels(const std::string &path) { return {}; }

MNISTDataset::MNISTDataset(const std::string &path)
{
    imgs_ = load_images(path + "images");
    labels_ = load_labels(path + "labels");
}

std::size_t MNISTDataset::length() { return imgs_.size(); }