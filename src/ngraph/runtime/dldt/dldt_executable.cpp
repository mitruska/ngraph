//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <utility>

#include "ngraph/log.hpp"
#include "ngraph/runtime/dldt/dldt_executable.hpp"
#include "ngraph/runtime/dlt/dldt_tensor.hpp"

ngraph::runtime::dldt::DLDT_Executable::DLDT_Executable(std::shared_ptr<Function> func,
                                                        std::string _device)
{
    auto opset = ngraph::get_opset1();
    bool all_opset1 = true;
    for (auto& node : func->get_ops())
    {
        if (!opset.contains_op_type(node.get()))
        {
            all_opset1 = false;
            break;
        }
    }

    if (!all_opset1)
    {
        std::cout << "UNSUPPORTED OPS DETECTED!" << std::endl;
        THROW_IE_EXCEPTION << "Exit from test";
    }
    std::cout << "Nodes in test: ";
    for (auto& node : func->get_ops())
    {
        std::cout << node->get_type_info().name << " ";
    }
    std::cout << std::endl;
    network = InferenceEngine::CNNNetwork(func);
    device = _device;
}

bool ngraph::runtime::dldt::DLDT_Executable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    try
    {
        InferenceEngine::Core ie;

        //  Loading model to the plugin (BACKEND_NAME)
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
        //  Create infer request
        InferenceEngine::InferRequest inferRequest = exeNetwork.CreateInferRequest();
        //  Prepare input and output blobs
        InferenceEngine::InputsDataMap inputInfo = network.getInputsInfo();

        if (inputInfo.size() != inputs.size())
        {
            THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
        }

        size_t i = 0;
        for (auto& it : inputInfo)
        {
            size_t size = inputs[i]->data.size() / sizeof(float);
            float* orig_data = (float*)inputs[i]->data.data();
            std::vector<float> data(orig_data, orig_data + size);
            inferRequest.SetBlob(it.first, fill_blob(it.second->getTensorDesc().getDims(), data));
            i++;
        }

        //  Prepare output blobs
        std::string output_name = network.getOutputsInfo().begin()->first;

        inferRequest.Infer();
        InferenceEngine::Blob::Ptr output = inferRequest.GetBlob(output_name);

        InferenceEngine::MemoryBlob::Ptr moutput =
            InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
        if (!moutput)
        {
            THROW_IE_EXCEPTION << "Cannot get output MemoryBlob in call_with_validate()";
        }

        auto lm = moutput->rmap();
        float* output_ptr = lm.as<float*>();
        // TODO: how to get size without explicit calculation?
        size_t size = 1;
        for (const auto& dim : output->getTensorDesc().getDims())
        {
            size *= dim;
        }
        //  Vector initialization from pointer
        std::vector<float> result(output_ptr, output_ptr + size);
        outputs[0]->write(result.data(), result.size() * sizeof(float));
        return true;
    }
    catch (...)
    {
        THROW_IE_EXCEPTION << "FAILED";
    }
}
}
