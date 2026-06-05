#include "nn/tensor.h"
#include <iostream>
#include <vector>

int main()
{
    std::vector<float> v{0, 1, 2, 3};
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(v);
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{v, v, v, v});

    std::shared_ptr<Tensor> t2p = std::make_shared<Tensor>(*t2);

    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;
    std::cout << *(t1 + std::make_shared<Tensor>(5.0)) << std::endl;
    std::cout << (*(t2 + t2p))(5) << std::endl;
    std::cout << (*(t1 * t1)).item() << std::endl;

    return 0;
}