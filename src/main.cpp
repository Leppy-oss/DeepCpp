#include <iostream>
#include <memory>
#include <nn/tensor.h>
#include <utils.h>
#include <vector>

int main()
{
    auto t1 = std::make_shared<Tensor>(5.0f, true);
    auto t2 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    auto t3 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}}, true
    );
    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;
    std::cout << *t3 << std::endl;

    auto t4 = t1 * t2;
    auto t5 = t2 * t3;
    auto t6 = t3->mm(t3);
    auto t7 = t2->mm(t6);

    std::cout << *t4 << std::endl;
    std::cout << *t5 << std::endl;
    std::cout << *t6 << std::endl;
    std::cout << *t7 << std::endl;

    auto t8 = Tensor::zeros({100, 1, 50});
    auto t9 = Tensor::zeros({100, 50, 128});
    auto t10 = t8->mm(t9);

    std::cout << t10->shape() << std::endl;

    return 0;
}