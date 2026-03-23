#include "nn/tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorTest, Construction)
{
    Tensor t1 = Tensor(0.0);

    EXPECT_EQ(t1.shape(), std::vector<std::size_t>{});
    EXPECT_THROW(t1(0), std::invalid_argument);
    EXPECT_EQ(t1.item(), 0.0);

    std::vector<float> v{0.0, 1.0, 2.0, 3.0};
    Tensor t2 = Tensor(v);

    EXPECT_EQ(t2.shape(), std::vector<std::size_t>{4});
    EXPECT_EQ(t2(0), 0.0);
    EXPECT_EQ(t2(2), 2.0);
    EXPECT_THROW(t2(4), std::invalid_argument);
    EXPECT_THROW(t2.item(), std::runtime_error);
}