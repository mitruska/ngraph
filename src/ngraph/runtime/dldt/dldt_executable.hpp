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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <plaidml/plaidml++.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace dldt
        {
            // A PlaidML executable object produced by compiling an nGraph function.
            class DLDT_Executable final : public Executable
            {
            public:
                DLDT_Executable(std::shared_ptr<Function> func, std::string device);
                virtual ~DLDT_Executable() {}
                bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) final;

                // std::vector<PerformanceCounter> get_performance_data() const final;

                // void save_as_format(const std::string& filename, plaidml_file_format format)
                // const;

                // const std::shared_ptr<Function>& src_func() const { return m_src_func; }
            private:
                InferenceEngine::CNNNetwork network;
                std::string device;
            };
        }
    }
}
