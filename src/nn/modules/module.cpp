#include "nn/modules/module.h"
#include "nn/tensor.h"
#include "utils.h"
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

Parameter::Parameter(std::string name, std::shared_ptr<Tensor> param) : name_{std::move(name)}, param_(std::move(param)) {} // no-format

const std::string &Parameter::name() const { return name_; }

std::string &Parameter::name() { return name_; }

std::shared_ptr<Tensor> Parameter::param() const { return param_; }

const std::string &Module::name() const { return name_; }

std::string &Module::name() { return name_; }

std::shared_ptr<Tensor> Module::forward(std::shared_ptr<Tensor> x)
{
    throw std::runtime_error("Cannot call forward on based Module class");
}

std::shared_ptr<Tensor> Module::operator()(std::shared_ptr<Tensor> x) { return forward(x); }

std::vector<Parameter> Module::parameters() const
{
    std::vector<Parameter> parameters;
    for (const Parameter &p : parameters_)
    {
        parameters.push_back(p);
    }

    for (const auto &m : modules_)
    {
        for (const Parameter &p : m->parameters())
        {
            std::string full_name = m->name().empty() ? p.name() : m->name() + "." + p.name();
            parameters.push_back(Parameter(full_name, p.param()));
        }
    }

    return parameters;
}

void Module::add_parameter(Parameter param)
{
    for (const Parameter &p : parameters_)
    {
        if (p.name() == param.name())
        {
            throw std::runtime_error("Parameter " + param.name() + " already exists in module");
        }
    }
    parameters_.push_back(param);
}

void Module::add_parameter(std::string name, std::shared_ptr<Tensor> param) { add_parameter(Parameter(name, param)); }

const std::vector<std::shared_ptr<Module>> &Module::modules() const { return modules_; }

void Module::add_module(std::shared_ptr<Module> module)
{
    for (const auto m : modules_)
    {
        if (m->name() == module->name())
        {
            throw std::runtime_error("Submodule " + module->name() + " already exists in module");
        }
    }

    modules_.push_back(module);
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Module::state_dict() const
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;

    std::vector<Parameter> params = parameters();
    for (const Parameter &p : params)
    {
        state_dict[p.name()] = p.param();
    }

    return state_dict;
}

void Module::load_state_dict(std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict)
{
    for (const Parameter &p : parameters())
    {
        auto it = state_dict.find(p.name());
        if (it == state_dict.end())
        {
            std::cerr << "Warning: Parameter " << p.name() << " not found in state_dict" << std::endl;
            continue;
        }
        std::shared_ptr<Tensor> param = it->second;
        if (p.param()->shape() != param->shape())
        {
            throw std::runtime_error(
                "Shape mismatch for parameter '" + p.name() + "' (found " + utils::to_string(param->shape()) +
                ", required " + utils::to_string(p.param()->shape()) + ")"
            );
        }
        p.param()->load_data(param->storage());
    }
}