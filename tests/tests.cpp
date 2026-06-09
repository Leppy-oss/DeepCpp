#include "nn/tensor.h"
#include <gtest/gtest.h>
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
    EXPECT_THROW(t2(4), std::invalid_argument);
    EXPECT_THROW(t2.item(), std::runtime_error);
}

TEST(TensorTest, Multiplication)
{
    std::shared_ptr<Tensor> s1 = std::make_shared<Tensor>(0.5f);
    std::shared_ptr<Tensor> s2 = std::make_shared<Tensor>(5.0f);

    std::shared_ptr<Tensor> v1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    std::shared_ptr<Tensor> v2 = std::make_shared<Tensor>(std::vector<float>{5.0f, 10.0f});
    std::shared_ptr<Tensor> v3 = s1 * v1;
    std::shared_ptr<Tensor> v4 = v1->mm(v1);

    std::shared_ptr<Tensor> m1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}
    });
    std::shared_ptr<Tensor> m2 = m1->mm(m1);
    std::shared_ptr<Tensor> m3 = v1->mm(m1);

    EXPECT_EQ((s1 * s2)->item(), 2.5f);
    EXPECT_EQ((*v3)(2), 1.5f);
    EXPECT_EQ(v4->item(), 30);
    EXPECT_EQ((*m2)(2, 3), (*m2)(3, 0));
    EXPECT_EQ((*m2)(3, 3), 1.0f);
    EXPECT_EQ((*m3)(3), 4.0f);

    EXPECT_THROW(m1 * v2, std::invalid_argument);
}