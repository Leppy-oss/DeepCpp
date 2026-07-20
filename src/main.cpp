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
    imshow(train_dataset.get_item(5).first);

    return 0;
}