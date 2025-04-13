#include <iostream>
#include <vector>
#include <memory>

#include <torch/script.h>

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "loading successfully" << std::endl;

    // model input, batch_size x num_inputs
    // auto input = torch::IValue(torch::rand({1, 3}, torch::TensorOptions().requires_grad(true)));
    // auto input = torch::IValue(torch::tensor({{0.0, 0.3, 1.5, 0.0, 0.5000}}, torch::requires_grad()));
    auto tensor1 = torch::tensor({{0.0, 0.3, 1.5, 0.0, 0.5000}}, torch::requires_grad());
    auto tensor2 = torch::tensor({{0.0, 0.3, 1.5, 0.0, 0.6000}}, torch::requires_grad());
    auto input = torch::IValue(torch::cat({tensor1, tensor2}, 0));
    input.toTensor().retain_grad();
    std::cout << input.toTensor() << std::endl;

    // model output, batch_size x num_outputs
    auto output = module.forward({input}).toTensor();
    // std::cout << output.slice(0, 0, 2) << std::endl;
    std::cout << output << std::endl;

    // auto_grad backward
    torch::Tensor grad_tensor = torch::zeros_like(output);
    grad_tensor.slice(1, 0, 1).squeeze().fill_(1.0);
    output.backward(grad_tensor);
    // output.slice(1, 0, 1).squeeze().backward();
    // output.index({0, 0}).backward();
    // output.index({1, 0}).backward();

    std::cout << "backward successfully" << std::endl;


    // grad to inputs
    std::cout << input.toTensor().grad() << std::endl;
}
