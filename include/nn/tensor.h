#pragma once
#include <vector>

using namespace std;

class Tensor
{
private:
    vector<float> _data;
    vector<size_t> _shape;
    vector<size_t> _stride;

public:
    Tensor(float data);
    Tensor(vector<float> data);
    Tensor(vector<vector<float>> data);
    const float &item() const;
    float &item();
};