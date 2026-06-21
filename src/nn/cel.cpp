#include "nn/cel.h"
#include "nn/loss.h"
#include "nn/module.h"
#include "tensor.h"
#include "utils.h"
#include <functional>
#include <limits>
#include <memory>
#include <vector>

Cel::Cel(std::size_t dim) : Loss("Cross Entropy Loss"), dim_{dim} {}

std::shared_ptr<Tensor> Cel::forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y)
{
    if (y_hat->ndim() == 0)
    {
        throw std::invalid_argument("CEL requires tensor to be at least 1d");
    }

    if (dim_ >= y_hat->ndim())
    {
        throw std::runtime_error(
            "Dim " + std::to_string(dim_) + " invalid for tensor with " + std::to_string(y_hat->ndim()) + " dimensions"
        );
    }

    tensor::Shape lshape = y_hat->shape();
    tensor::Shape pshape;
    std::size_t n_cls = lshape[dim_];

    std::size_t outer = 1, inner = 1;
    for (std::size_t dim = 0; dim < y_hat->ndim(); dim++)
    {
        if (dim < dim_)
        {
            outer *= lshape[dim];
            pshape.push_back(lshape[dim]);
        }
        else if (dim > dim_)
        {
            inner *= lshape[dim];
            pshape.push_back(lshape[dim]);
        }
    }

    if (pshape != y->shape())
    {
        throw std::invalid_argument(
            "Shape of logits " + utils::to_string(y_hat->shape()) + " incompatible with shape of gt classes " +
            utils::to_string(y->shape())
        );
    }

    std::vector<float> ps(y_hat->numel(), 0.0f);
    float loss = 0.0f;

    for (std::size_t outer_idx = 0; outer_idx < outer; outer_idx++)
    {
        for (std::size_t inner_idx = 0; inner_idx < inner; inner_idx++)
        {
            std::size_t gt_cls = static_cast<std::size_t>(y->at(outer_idx * inner + inner_idx));

            if (gt_cls >= n_cls)
            {
                throw std::runtime_error(
                    "Found gt class of " + std::to_string(gt_cls) + " (max " + std::to_string(n_cls - 1) + ")"
                );
            }

            std::size_t outer_inner_idx = outer_idx * n_cls * inner + inner_idx;
            float max_val = std::numeric_limits<float>::lowest();

            for (std::size_t cls = 0; cls < n_cls; cls++)
            {
                max_val = std::max(max_val, y_hat->at(cls * inner + outer_inner_idx));
            }

            float sum = 0.0f;

            for (std::size_t cls = 0; cls < n_cls; cls++)
            {
                float ve = std::exp(y_hat->at(cls * inner + outer_inner_idx) - max_val);
                ps[cls * inner + outer_inner_idx] = ve;
                sum += ve;
            }

            for (std::size_t cls = 0; cls < n_cls; cls++)
            {
                ps[cls * inner + outer_inner_idx] /= sum;
            }

            loss -= std::log(ps[gt_cls * inner + outer_inner_idx]);
        }
    }

    loss /= static_cast<float>(outer * inner);

    std::function<void(std::shared_ptr<Tensor>)> gradfn = nullptr;
    std::vector<std::shared_ptr<Tensor>> parents = {};

    if (y_hat->requires_grad())
    {
        gradfn = [y_hat, y, lshape, ps, outer, inner, n_cls](std::shared_ptr<Tensor> grad_prev)
        {
            auto grad_update = Tensor::zeros(lshape);
            for (std::size_t outer_idx = 0; outer_idx < outer; outer_idx++)
            {
                for (std::size_t inner_idx = 0; inner_idx < inner; inner_idx++)
                {
                    std::size_t gt_cls = static_cast<std::size_t>(y->at(outer_idx * inner + inner_idx));
                    std::size_t outer_inner_idx = outer_idx * n_cls * inner + inner_idx;

                    for (std::size_t cls = 0; cls < n_cls; cls++)
                    {
                        std::size_t idx = cls * inner + outer_inner_idx;
                        grad_update->at(idx) = (ps[idx] - (cls == gt_cls)) * grad_prev->at(0) / (outer * inner);
                    }
                }
            }
            y_hat->add_grad(grad_update);
        };
        parents = {y_hat};
    }

    return std::make_shared<Tensor>(loss, y_hat->requires_grad(), gradfn, parents);
}