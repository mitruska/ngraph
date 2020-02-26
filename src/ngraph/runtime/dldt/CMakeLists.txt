# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(SRC
	backend_dldt.cpp
	dldt_tensor_view.cpp
	dldt_executable.hpp 
)
#add_library(libdldt INTERFACE)
#target_link_libraries(libdldt INTERFACE ${DLDT_LIBRARIES})
add_library(libdldt_backend SHARED ${SRC})
#target_include_directories(libdldt_backend SYSTEM PUBLIC ${DLDT_INCLUDE_DIRS})
target_include_directories(libdldt_backend PUBLIC ${IE_MAIN_SOURCE_DIR}/include)
target_link_libraries(libdldt_backend PRIVATE ngraph inference_engine)
install(TARGETS libdldt_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})

target_compile_definitions(libdldt_backend PRIVATE DLDT_BACKEND_EXPORTS)

#set(CMAKE_MACOSX_RPATH 1)
#if(APPLE)
#    set_property(TARGET libdldt_backend PROPERTY INSTALL_RPATH "@loader_path/;@loader_path/../../..")
#elseif(DEFINED NGRAPH_RPATH)
#    set_property(TARGET libdldt_backend PROPERTY INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..;${NGRAPH_RPATH}")
#else()
#    set_property(TARGET libdldt_backend PROPERTY INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..")
#endif()
