#include "data/dataloader.h"
#include "data/mnist_dataset.h"
#include "nn/argmax.h"
#include "nn/cel.h"
#include "nn/flatten.h"
#include "nn/linear.h"
#include "nn/relu.h"
#include "nn/sequential.h"
#include "optim/adam.h"
#include "tensor.h"
#include <algorithm>
#include <filesystem>
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
            std::cout << "Loss: " << std::fixed << std::setprecision(5) << loss->item() << " ["
                      << std::min(batch_idx * dataloader.batch_size(), dataloader.length()) << " / "
                      << dataloader.length() << "]\n";
        }

        batch_idx++;
    }
}

void test_epoch(DataLoader &dataloader, Sequential &model, Cel &loss_fn, Argmax &am)
{
    float correct = 0, total_loss = 0;
    for (const auto &batch : dataloader)
    {
        auto preds = model(batch.first);
        auto gts = batch.second;
        total_loss += loss_fn(preds, gts)->item();
        auto cls = am(preds);
        for (std::size_t batch = 0; batch < dataloader.batch_size(); batch++)
        {
            if (static_cast<size_t>(cls->at(batch)) == static_cast<size_t>(gts->at(batch)))
            {
                correct++;
            }
        }
    }
    // clang-format off
    std::cout << "Avg batch loss: " << std::fixed << std::setprecision(5) << total_loss / dataloader.length_batches() << "\n";
    // clang-format on
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << correct / dataloader.length() << "\n";
}

int main()
{
    std::cout << "Loading training dataset\n";
    MNISTDataset train_dataset(std::string(PROJECT_ROOT) + "/data/train/");
    DataLoader train_dataloader(&train_dataset, 32, true);

    std::cout << "Loading test dataset\n";
    MNISTDataset test_dataset(std::string(PROJECT_ROOT) + "/data/test/");
    DataLoader test_dataloader(&test_dataset, 128, true);

    std::cout << "Instantiating model, optimizer, etc.\n";
    Sequential model(
        {std::make_shared<Flatten>(),
         std::make_shared<Relu>(),
         std::make_shared<Linear>(784, 256),
         std::make_shared<Relu>(),
         std::make_shared<Linear>(256, 128),
         std::make_shared<Relu>(),
         std::make_shared<Linear>(128, 10)}
    );

    Adam optim(model.parameters());
    Cel loss_fn;
    Argmax am;

    std::string model_path = std::string(PROJECT_ROOT) + "/models/mnist.nn";
    if (std::filesystem::exists(model_path))
    {
        std::cout << "Found serialized mnist.nn, loading model state dict\n";
        auto state_dict = utils::load_model(model_path);
        model.load_state_dict(state_dict);
    }

    std::size_t num_epochs = 5;
    for (std::size_t epoch = 0; epoch < num_epochs; epoch++)
    {
        std::cout << "Training epoch " << epoch << " / " << num_epochs << "...\n";
        train_epoch(train_dataloader, model, optim, loss_fn);

        std::cout << "Testing epoch " << epoch << " / " << num_epochs << "...\n";
        test_epoch(test_dataloader, model, loss_fn, am);
    }

    std::cout << "Serializing model to " + model_path + "\n";
    utils::save_model(model.state_dict(), model_path);

    return 0;
}
