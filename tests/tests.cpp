#include "nn/argmax.h"
#include "nn/flatten.h"
#include "nn/linear.h"
#include "nn/module.h"
#include "nn/softmax.h"
#include "tensor.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(TensorTest, Construction)
{
    Tensor t1 = Tensor(0.0f);

    EXPECT_EQ(t1.shape(), std::vector<std::size_t>{});
    EXPECT_THROW(t1(0), std::invalid_argument);
    EXPECT_EQ(t1.item(), 0.0f);

    std::vector<float> v{0.0f, 1.0f, 2.0f, 3.0f};
    Tensor t2 = Tensor(v);

    EXPECT_EQ(t2.shape(), std::vector<std::size_t>{4});
    EXPECT_EQ(t2(0), 0.0f);
    EXPECT_EQ(t2(2), 2.0f);
    EXPECT_THROW(t2(4), std::out_of_range);
    EXPECT_THROW(t2.item(), std::runtime_error);
}

TEST(TensorTest, Multiplication)
{
    auto s1 = std::make_shared<Tensor>(0.5f);
    auto s2 = std::make_shared<Tensor>(5.0f);

    auto v1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto v2 = std::make_shared<Tensor>(std::vector<float>{5.0f, 10.0f});
    auto v3 = s1 * v1;
    auto v4 = v1->mm(v1);

    auto m1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}
    });
    auto m2 = m1->mm(m1);
    auto m3 = v1->mm(m1);

    EXPECT_EQ((s1 * s2)->item(), 2.5f);
    EXPECT_EQ((*v3)(2), 1.5f);
    EXPECT_EQ(v4->item(), 30);
    EXPECT_EQ((*m2)(2, 3), (*m2)(3, 0));
    EXPECT_EQ((*m2)(3, 3), 1.0f);
    EXPECT_EQ((*m3)(3), 4.0f);

    EXPECT_THROW(m1 * v2, std::invalid_argument);
}

TEST(TensorTest, SqueezeUnsqueeze)
{
    auto t = Tensor::zeros({80, 64, 64}, true);
    auto u = t->unsqueeze(3);
    auto s = u->squeeze(3);

    EXPECT_EQ(u->shape(), (tensor::Shape{80, 64, 64, 1}));
    EXPECT_EQ(t->shape(), s->shape());
}

TEST(TensorTest, Reshape)
{
    auto t = Tensor::zeros({50, 4, 16}, true);
    auto r = t->reshape({200, 16});

    EXPECT_EQ(r->shape(), (tensor::Shape{200, 16}));
}

TEST(ModuleTest, Flatten)
{
    auto t = Tensor::zeros({50, 4, 16}, true);
    auto f = Flatten();
    auto ft = f(t);

    EXPECT_EQ(ft->shape(), (tensor::Shape{50, 64}));
}

TEST(ModuleTest, Linear)
{
    std::vector<std::shared_ptr<Module>> layers = {
        std::make_shared<Flatten>(),
        std::make_shared<Linear>(784, 128),
        std::make_shared<Linear>(128, 64),
        std::make_shared<Linear>(64, 10)
    };
    auto x = Tensor::zeros({128, 28, 28});
    auto y = x;
    for (auto &m : layers)
    {
        y = (*m)(y);
    }
    EXPECT_EQ(y->shape(), (tensor::Shape{128, 10}));
}

TEST(ModuleTest, Softmax)
{
    auto x = Tensor::zeros({128, 10});
    auto sm = Softmax();
    EXPECT_EQ(sm(x)->shape(), (tensor::Shape{128, 10}));
}

TEST(ModuleTest, Argmax)
{
    auto x = std::make_shared<Tensor>(std::vector<float>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1})->unsqueeze(0);
    auto am = Argmax();
    EXPECT_EQ(am(x)->at(0), static_cast<float>(5));
}