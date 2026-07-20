#include "data/dataloader.h"
#include "data/mnist_dataset.h"
#include "nn/cel.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

int main()
{
    MNISTDataset train_dataset("../data/train/");
    DataLoader train_dataloader(&train_dataset, 1, true);

    utils::imshow((*train_dataloader.begin()).first->squeeze(0));
    std::cout << (*train_dataloader.begin()).second->squeeze(0)->item() << "\n";

    return 0;
}