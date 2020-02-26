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

#include <string>

#include "ngraph/runtime/dldt/dldt_backend_visibility.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"

// This define is a workaround for gcc on centos and is required for aligned()
#define NGRAPH_CPU_ALIGNMENT 64

namespace ngraph
{
    namespace runtime
    {
        namespace dldt
        {
            class DLDTTensorView : public ngraph::runtime::Tensor
            {
            public:
                DLDT_BACKEND_API DLDTTensorView(const ngraph::element::Type& element_type,
                                                const Shape& shape);
                DLDT_BACKEND_API DLDTTensorView(const ngraph::element::Type& element_type,
                                                const PartialShape& shape);

                /// \brief Write bytes directly into the tensor
                /// \param p Pointer to source of data
                /// \param n Number of bytes to write, must be integral number of elements.
                void write(const void* p, size_t n) override;

                /// \brief Read bytes directly from the tensor
                /// \param p Pointer to destination for data
                /// \param n Number of bytes to read, must be integral number of elements.
                void read(void* p, size_t n) const override;

                /// \brief copy bytes directly from source to this tensor
                /// \param source The source tensor

            private:
                DLDTTensorView(const DLDTTensorView&) = delete;
                DLDTTensorView(DLDTTensorView&&) = delete;
                DLDTTensorView& operator=(const DLDTTensorView&) = delete;
           public:
                std::vector<int8_t> data;
            };
        }
    }
}
