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

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            /// Experimental for support of recurrent matches.
            ///
            /// Capture adds the pattern value map to a list of pattern value maps and resets
            /// matches for pattern nodes not in the static node list. The match always succeeds.
            class NGRAPH_API Capture : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternCapture", 0};
                const NodeTypeInfo& get_type_info() const override;
                Capture(const Output<Node>& arg)
                    : Pattern({arg})
                {
                    set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
                }

                /// \brief static nodes are retained after a capture. All other nodes are dropped
                std::set<Node*> get_static_nodes() { return m_static_nodes; }
                void set_static_nodes(const std::set<Node*>& static_nodes)
                {
                    m_static_nodes = static_nodes;
                }

                virtual bool match_value(pattern::Matcher* matcher,
                                         const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value) override;

            protected:
                std::set<Node*> m_static_nodes;
            };
        }
    }
}
