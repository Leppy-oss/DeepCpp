#include "utils.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <string>

void imshow(std::shared_ptr<Tensor> img)
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