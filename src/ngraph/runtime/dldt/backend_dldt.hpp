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

#pragma once

#include <ie_core.hpp>
#include <string>
#include "ngraph/opsets/opset.hpp"
#include "ngraph/runtime/dldt/dldt_executable.hpp"
#include "ngraph/runtime/dldt/dldt_tensor_view.hpp"
#include "ngraph/runtime/tensor.hpp"
//#include "ngraph/ngraph.hpp"

InferenceEngine::Blob::Ptr fill_blob(InferenceEngine::SizeVector shape, std::vector<float> data);

class Handle;

namespace ngraph
{
    namespace runtime
    {
        namespace dldt
        {
            class DLDT_Backend : public runtime::Backend
            {
            public:
                DLDT_Backend(const std::string& configuration_string);
                ~DLDT_Backend() {}
                std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                                    bool enable_performance_data = false);
                bool is_supported(const Node& node) const;

                bool is_supported_property(const Property prop) const;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_dynamic_tensor(ngraph::element::Type type, ngraph::PartialShape shape)
                {
                    return std::make_shared<DLDTTensorView>(type, shape);
                }

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type, const Shape& shape);

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer)
                {
                    return std::make_shared<DLDTTensorView>(element_type, shape);
                }

                template <class T>
                std::shared_ptr<ngraph::runtime::Tensor> create_tensor(ngraph::Shape shape)
                {
                    return std::make_shared<DLDTTensorView>(ngraph::element::from<T>(), shape);
                }

                template <typename T>
                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(ngraph::element::Type type, ngraph::Shape shape, T* data)
                {
                    auto tensor = std::make_shared<DLDTTensorView>(type, shape);
                    size_t size = 1;
                    for (auto x : shape)
                    {
                        size *= x;
                    }
                    std::vector<T> v(data, data + size);
                    tensor->write(data, size * sizeof(T));
                    return tensor;
                }

            private:
                std::string device;
            };
        }
    }
}
