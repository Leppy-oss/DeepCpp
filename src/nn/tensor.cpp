#include "nn/tensor.h"
#include <iostream>
#include <vector>

using namespace std;

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {};

Tensor::Tensor(vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {};

Tensor::Tensor(vector<vector<float>> data)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}
{
    size_t n_expected_cols = data[0].size();
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i].size() != n_expected_cols)
            throw invalid_argument(
                "Dimensions are inconsistent. Please check the number of columns in each row.");
    }
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t j = 0; j < data[i].size(); j++)
        {
            _data.push_back(data[i][j]);
        }
    }
}