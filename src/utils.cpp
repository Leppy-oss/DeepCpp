#include "utils.h"
#include "tensor.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

const uint32_t MAGIC_NUMBER = 46637;

void utils::imshow(std::shared_ptr<Tensor> img)
{
    for (std::size_t r = 0; r < img->shape()[0]; r++)
    {
        for (std::size_t c = 0; c < img->shape()[1]; c++)
        {
            float pxl = (*img)(r, c);
            std::cout << (pxl > 0.75 ? '@' : pxl > 0.5 ? '#' : pxl > 0.25 ? '+' : pxl > 0.1 ? '.' : ' ');
        }
        std::cout << "\n";
    }
}

void utils::save_model(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict, const std::string &path
)
{
    std::ofstream file(path, std::ios::binary);

    if (!file || !file.is_open())
    {
        throw std::runtime_error("Could not open file " + path);
    }

    file.write(reinterpret_cast<const char *>(&MAGIC_NUMBER), 4);
    for (const auto &[name, param] : state_dict)
    {
        std::size_t name_len = name.size();
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(std::size_t));
        file.write(name.data(), name_len);

        std::size_t shape_len = param->shape().size();
        file.write(reinterpret_cast<const char *>(&shape_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(param->shape().data()), shape_len * sizeof(std::size_t));

        std::size_t data_len = param->numel();
        file.write(reinterpret_cast<const char *>(&data_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(param->storage()->data()), data_len * sizeof(float));
    }

    file.close();
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> utils::load_model(const std::string &path)
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(path, std::ios::binary);

    if (!file || !file.is_open())
    {
        throw std::runtime_error("Could not open file " + path);
    }

    uint32_t magic_number;
    file.read(reinterpret_cast<char *>(&magic_number), 4);
    if (magic_number != MAGIC_NUMBER)
    {
        throw std::runtime_error("File '" + path + "' is not in proper format (missing magic number)");
    }

    while (file.peek() != EOF)
    {
        std::size_t name_len;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(std::size_t));
        std::string name(name_len, ' ');
        file.read(name.data(), name_len);

        std::size_t shape_len;
        file.read(reinterpret_cast<char *>(&shape_len), sizeof(std::size_t));
        std::vector<std::size_t> shape(shape_len);
        file.read(reinterpret_cast<char *>(shape.data()), shape_len * sizeof(std::size_t));

        std::size_t data_len;
        file.read(reinterpret_cast<char *>(&data_len), sizeof(std::size_t));
        std::vector<float> data(data_len);
        file.read(reinterpret_cast<char *>(data.data()), data_len * sizeof(float));

        state_dict[name] = std::make_shared<Tensor>(data, shape);
    }

    file.close();

    return state_dict;
}