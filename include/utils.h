#pragma once
#include "tensor.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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
}

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &obj)
{
    os << utils::to_string(obj);
    return os;
}