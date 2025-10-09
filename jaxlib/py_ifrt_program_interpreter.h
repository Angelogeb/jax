/* Copyright 2025 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_PY_IFRT_PROGRAM_INTERPRETER_H_
#define JAXLIB_PY_IFRT_PROGRAM_INTERPRETER_H_

#include <memory>
#include <string>
#include <vector>

#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "xla/python/ifrt/ir/program_interpreter.h"

namespace jax {

// Python wrapper for xla::ifrt::ProgramInterpreter that allows execution of
// IFRT-IR programs directly from Python.
class PyIfrtProgramInterpreter {
 public:
  // Creates a PyIfrtProgramInterpreter from an MLIR IFRT-IR module.
  // Takes the MLIR module as a string, the client to use, and the devices.
  static absl::StatusOr<std::unique_ptr<PyIfrtProgramInterpreter>>
  CreateFromMlirModule(std::string mlir_module_str,
                       nb_class_ptr<PyClient> client,
                       nanobind::sequence devices);

  // Returns the input specifications (shapes, dtypes, shardings) of the program.
  nanobind::object GetInputSpecs() const;

  // Returns the output specifications (shapes, dtypes, shardings) of the program.
  nanobind::object GetOutputSpecs() const;

  // Returns the program name.
  std::string GetProgramName() const;

  // Accessors for internal use by bindings
  xla::ifrt::ProgramInterpreter* interpreter() { return interpreter_.get(); }
  nb_class_ptr<PyClient> client() { return client_; }

 private:
  PyIfrtProgramInterpreter(
      std::unique_ptr<xla::ifrt::ProgramInterpreter> interpreter,
      std::shared_ptr<xla::ifrt::CompiledIfrtIrProgram> compiled_program,
      nb_class_ptr<PyClient> client);

  std::unique_ptr<xla::ifrt::ProgramInterpreter> interpreter_;
  std::shared_ptr<xla::ifrt::CompiledIfrtIrProgram> compiled_program_;
  nb_class_ptr<PyClient> client_;
};

void BuildIfrtProgramInterpreterSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_PY_IFRT_PROGRAM_INTERPRETER_H_

