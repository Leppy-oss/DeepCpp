#include "data/mnist_dataset.h"
#include "nn/cel.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    // MNISTDataset train_dataset("../data/train/");
    // imshow(train_dataset.get_item(5).first);

    auto t1 = std::make_shared<Tensor>(std::vector<float>{1, 1, 1});
    auto t2 = std::make_shared<Tensor>(std::vector<float>{2, 2, 2});
    auto t3 = std::make_shared<Tensor>(std::vector<float>{3, 3, 3});

    std::cout << *Tensor::stack({t1, t2, t3}) << std::endl;

    return 0;
}