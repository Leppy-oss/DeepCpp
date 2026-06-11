#pragma once
#include "nn/tensor.h"
#include <memory>
#include <string>
#include <unordered_Map>
#include <utility>
#include <vector>

class Parameter
{
private:
    std::string name_;
    std::shared_ptr<Tensor> param_;

public:
    Parameter(std::string name, std::shared_ptr<Tensor> param);
    const std::string &name() const;
    std::string &name();
    std::shared_ptr<Tensor> param() const;
};

class Module
{
private:
    std::string name_;
    std::vector<Parameter> parameters_;
    std::vector<std::shared_ptr<Module>> modules_;

public:
    Module(std::string name = "");
    const std::string &name() const;
    std::string &name();

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> x);

    std::vector<Parameter> parameters() const;
    void add_parameter(Parameter param);
    void add_parameter(std::string name, std::shared_ptr<Tensor> param);

    const std::vector<std::shared_ptr<Module>> &modules() const;
    void add_module(std::shared_ptr<Module> module);

    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict() const;

    void load_state_dict(std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);
};