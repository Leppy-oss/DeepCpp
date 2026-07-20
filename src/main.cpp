#include "data/dataloader.h"
#include "data/mnist_dataset.h"
#include "nn/cel.h"
#include "nn/flatten.h"
#include "nn/linear.h"
#include "nn/sequential.h"
#include "nn/softmax.h"
#include "optim/adam.h"
#include "tensor.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utils.h>
#include <vector>

void train_epoch(DataLoader &dataloader, Sequential &model, Adam &optim, Cel &loss_fn)
{
    std::size_t log_interval = 100;
    std::size_t batch_idx = 0;

    for (const auto &batch : dataloader)
    {
        std::shared_ptr<Tensor> loss = loss_fn(model(batch.first), batch.second);
        loss->backward();
        optim.step();
        optim.zero_grad();

        if (batch_idx % log_interval == 0)
        {
            std::cout << "loss: " << std::fixed << std::setprecision(5) << loss->item() << " @ ["
                      << std::min(batch_idx * dataloader.batch_size(), dataloader.length()) << " / "
                      << dataloader.length() << "]\n";
        }

        batch_idx++;
    }
}

void test_epoch(DataLoader &dataloader, Sequential &model, Cel &loss_fn)
{
    float correct = 0, total_loss = 0;
    for (const auto &batch : dataloader)
    {
        auto preds = model(batch.first);
        total_loss += loss_fn(preds, batch.second)->item();
        for (std::size_t batch = 0; batch < preds->shape()[0]; batch++)
        {
            std::size_t pred = 0;
            for (std::size_t cls = 0; cls < preds->shape()[1]; cls++)
            {
            }
        }
    }
}

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