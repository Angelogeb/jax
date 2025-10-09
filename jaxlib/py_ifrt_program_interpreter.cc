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

#include "jaxlib/py_ifrt_program_interpreter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/sharding.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/basic_atom_program_compiler.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/program_interpreter.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace nb = nanobind;

namespace jax {

PyIfrtProgramInterpreter::PyIfrtProgramInterpreter(
    std::unique_ptr<xla::ifrt::ProgramInterpreter> interpreter,
    std::shared_ptr<xla::ifrt::CompiledIfrtIrProgram> compiled_program,
    nb_class_ptr<PyClient> client)
    : interpreter_(std::move(interpreter)),
      compiled_program_(std::move(compiled_program)),
      client_(std::move(client)) {}

/* static */
absl::StatusOr<std::unique_ptr<PyIfrtProgramInterpreter>>
PyIfrtProgramInterpreter::CreateFromMlirModule(
    std::string mlir_module_str, nb_class_ptr<PyClient> client,
    nb::sequence devices) {
  // Parse the MLIR module.
  auto context = std::make_unique<mlir::MLIRContext>();
  // Load the IFRT dialect before parsing
  context->loadDialect<xla::ifrt::IfrtDialect>();
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, *context));

  // Get the device list.
  xla::ifrt::DeviceListRef ifrt_device_list;
  if (devices.type().is(PyDeviceList::type())) {
    TF_ASSIGN_OR_RETURN(ifrt_device_list,
                        nb::cast<const PyDeviceList*>(devices)->ifrt_device_list());
  } else {
    auto py_devices = nb::cast<std::vector<nb_class_ptr<PyDevice>>>(devices);
    if (py_devices.empty()) {
      return absl::InvalidArgumentError(
          "IFRT program interpreter requires at least one device");
    }
    absl::InlinedVector<xla::ifrt::Device*, 1> ifrt_devices;
    ifrt_devices.reserve(py_devices.size());
    for (const nb_class_ptr<PyDevice>& py_device : py_devices) {
      ifrt_devices.push_back(py_device->device());
    }
    TF_ASSIGN_OR_RETURN(ifrt_device_list,
                        client->ifrt_client()->MakeDeviceList(ifrt_devices));
  }

  // Create the IfrtIRProgram.
  auto ifrt_ir_program = std::make_unique<xla::ifrt::IfrtIRProgram>(
      std::move(context), std::move(module));

  // Create the compile options.
  auto compile_options =
      std::make_unique<xla::ifrt::IfrtIRCompileOptions>();
  
  // Extract device assignments from device list.
  std::vector<xla::ifrt::DeviceId> device_assignments;
  device_assignments.reserve(ifrt_device_list->devices().size());
  for (const auto* device : ifrt_device_list->devices()) {
    device_assignments.push_back(device->Id());
  }
  compile_options->device_assignments = std::move(device_assignments);

  // Create the atom program compiler.
  TF_ASSIGN_OR_RETURN(
      auto atom_program_compiler,
      xla::ifrt::BasicAtomProgramCompiler::Create(
          client->ifrt_client(), compile_options->device_assignments));

  // Compile the IFRT IR program.
  TF_ASSIGN_OR_RETURN(
      auto compiled_program,
      xla::ifrt::CompiledIfrtIrProgram::Create(
          std::move(ifrt_ir_program), std::move(compile_options),
          client->ifrt_client(), std::move(atom_program_compiler)));

  // Wrap in shared_ptr for sharing between interpreter and PyIfrtProgramInterpreter.
  auto shared_compiled_program =
      std::make_shared<xla::ifrt::CompiledIfrtIrProgram>(
          std::move(compiled_program));

  // Create the program interpreter.
  TF_ASSIGN_OR_RETURN(
      auto interpreter,
      xla::ifrt::ProgramInterpreter::Create(
          client->ifrt_client(), shared_compiled_program, ifrt_device_list));

  return std::unique_ptr<PyIfrtProgramInterpreter>(
      new PyIfrtProgramInterpreter(std::move(interpreter),
                                   std::move(shared_compiled_program),
                                   std::move(client)));
}

nb::object PyIfrtProgramInterpreter::GetInputSpecs() const {
  // TODO: Implement conversion of ArraySpec to Python objects.
  // For now, return None. A full implementation would convert the
  // compiled_program_->in_specs to Python dictionaries or objects.
  return nb::none();
}

nb::object PyIfrtProgramInterpreter::GetOutputSpecs() const {
  // TODO: Implement conversion of ArraySpec to Python objects.
  // For now, return None. A full implementation would convert the
  // compiled_program_->out_specs to Python dictionaries or objects.
  return nb::none();
}

std::string PyIfrtProgramInterpreter::GetProgramName() const {
  return compiled_program_->program_name;
}

void BuildIfrtProgramInterpreterSubmodule(nb::module_& m) {
  auto sub_module = m.def_submodule("ifrt_interpreter");
  sub_module.attr("_Client") = m.attr("Client");
  sub_module.attr("_Device") = m.attr("Device");
  sub_module.attr("_DeviceList") = m.attr("DeviceList");
  sub_module.attr("_ArrayImpl") = m.attr("ArrayImpl");

  nb::class_<PyIfrtProgramInterpreter>(sub_module, "ProgramInterpreter")
      .def_static(
          "create",
          xla::ValueOrThrowWrapper(
              PyIfrtProgramInterpreter::CreateFromMlirModule),
          nb::arg("mlir_module"), nb::arg("client"), nb::arg("devices"),
          nb::sig(
              // clang-format off
              "def create("
              "mlir_module: str, "
              "client: _Client, "
              "devices: typing.Sequence[_Device] | _DeviceList"
              ") -> ProgramInterpreter"
              // clang-format on
          ),
          "Creates a ProgramInterpreter from an MLIR IFRT-IR module string.")
      .def("execute",
           [](PyIfrtProgramInterpreter& self, nb::list inputs) {
             std::cout << "[TRACE] Execute called with " << nb::len(inputs) << " inputs" << std::endl;
             
             // Convert Python inputs to IFRT arrays
             std::vector<xla::ifrt::ArrayRef> ifrt_arrays;
             ifrt_arrays.reserve(nb::len(inputs));
             std::vector<nb::object> keep_alive;  // Keep Python objects alive
             keep_alive.reserve(nb::len(inputs));
             
             for (size_t i = 0; i < nb::len(inputs); ++i) {
               PyArray py_array = nb::cast<PyArray>(inputs[i]);
               ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
               keep_alive.push_back(inputs[i]);
             }
             
             std::cout << "[TRACE] Calling interpreter->Execute" << std::endl;
             
             // Execute with GIL released (following py_executable pattern)
             xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
             execute_options.fill_status = false;
             std::vector<xla::ifrt::ArrayRef> output_arrays;
             {
               nb::gil_scoped_release gil_release;
               auto result = xla::ValueOrThrow(
                   self.interpreter()->Execute(absl::MakeSpan(ifrt_arrays), 
                                             execute_options,
                                             /*devices=*/std::nullopt));
               output_arrays = std::move(result.outputs);
             }
             
             std::cout << "[TRACE] Execute returned with " << output_arrays.size() << " outputs" << std::endl;
             
             // Convert outputs to PyArrays (following py_executable pattern)
             std::vector<PyArray> outputs;
             outputs.reserve(output_arrays.size());
             for (auto& ifrt_array : output_arrays) {
               // Disassemble into single-device arrays (works for all sharding types)
               auto exploded_arrays = xla::ValueOrThrow(
                   ifrt_array->DisassembleIntoSingleDeviceArrays(
                       xla::ifrt::ArrayCopySemantics::kReuseInput,
                       xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));
               
               // Take the first exploded array (for single-device outputs)
               // TODO: Handle multi-device outputs properly (return all shards)
               outputs.push_back(PyArray::MakeFromSingleDeviceArray(
                   self.client(), std::move(exploded_arrays[0]),
                   /*weak_type=*/false, /*committed=*/true));
             }
             
             std::cout << "[TRACE] Returning " << outputs.size() << " outputs" << std::endl;
             return outputs;
           },
           nb::arg("inputs"),
           nb::sig(
               // clang-format off
               "def execute("
               "inputs: typing.Sequence[_ArrayImpl]"
               ") -> typing.Sequence[_ArrayImpl]"
               // clang-format on
           ),
           "Executes the IFRT-IR program with the given input arrays.")
      .def("get_input_specs", &PyIfrtProgramInterpreter::GetInputSpecs,
           "Returns the input specifications of the program.")
      .def("get_output_specs", &PyIfrtProgramInterpreter::GetOutputSpecs,
           "Returns the output specifications of the program.")
      .def("get_program_name", &PyIfrtProgramInterpreter::GetProgramName,
           "Returns the program name.");
}

}  // namespace jax

