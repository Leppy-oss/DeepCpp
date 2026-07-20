#include "data/dataloader.h"
#include "data/mnist_dataset.h"
#include "nn/flatten.h"
#include "nn/linear.h"
#include "nn/sequential.h"
#include "nn/softmax.h"
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

    Sequential model(
        {std::make_shared<Flatten>(),
         std::make_shared<Linear>(784, 256),
         std::make_shared<Linear>(256, 128),
         std::make_shared<Linear>(128, 10)}
    );

    auto out = (model((*train_dataloader.begin()).first));
    std::cout << *out << "\n";
    std::cout << *Softmax()(out) << "\n";

    return 0;
}