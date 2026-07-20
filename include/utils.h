#pragma once
#include "tensor.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils
{
    template <typename T> std::string to_string(const std::vector<T> &obj)
    {
        std::ostringstream ostr;
        ostr << "(";
        for (std::size_t i = 0; i < obj.size(); i++)
        {
            ostr << obj[i];
            if (i < obj.size() - 1)
            {
                ostr << ", ";
            }
        }
        ostr << ")";
        return ostr.str();
    }

    void imshow(std::shared_ptr<Tensor> img);

    // clang-format off
    void save_model(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict, const std::string &path);
    // clang-format on
    std::unordered_map<std::string, std::shared_ptr<Tensor>> load_model(const std::string &path);
}

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &obj)
{
    os << utils::to_string(obj);
    return os;
}