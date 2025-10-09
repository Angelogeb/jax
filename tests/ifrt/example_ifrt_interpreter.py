#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax._src.lib import _jax


def create_simple_call_program():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])

    mlir_module_str0 = """
!array_t0 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @add_one_program {
  func.func @main(%arg0: !array_t0) -> !array_t0 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0] : (!array_t0) -> !array_t0
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0] : (!array_t0) -> !array_t0
    return %1 : !array_t0
  }

  func.func private @add_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}"""
    args0 = [jax.device_put(x, jax.devices()[0])]
    expected_output0 = [x + 1.0 + 1.0]
    ex0 = (mlir_module_str0, args0, expected_output0)

    mlir_module_str1 = """
!array_t0 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
!array_t1 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @add_one_program {
  func.func @main(%arg0: !array_t0, %arg1: !array_t1) -> (!array_t0, !array_t1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0] : (!array_t0) -> !array_t0
    %1, %ctrl_1 = ifrt.Call @mul_one(%arg1) on devices [1] : (!array_t1) -> !array_t1
    return %0, %1 : !array_t0, !array_t1
  }

  func.func private @add_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  func.func private @mul_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}"""
    args1 = [jax.device_put(x, jax.devices()[0]), jax.device_put(x, jax.devices()[1])]
    expected_output1 = [x + 1.0, x * 1.0]
    ex1 = (mlir_module_str1, args1, expected_output1)

    mlir_module_str2 = """
!array_t0 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
!array_t1 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @add_one_program {
  func.func @main(%arg0: !array_t0) -> (!array_t0, !array_t1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0] : (!array_t0) -> !array_t0
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0] : (!array_t0) -> !array_t0
    %3, %ctrl_3 = ifrt.CopyArrays(%1) : (!array_t0) -> !array_t1
    %2, %ctrl_2 = ifrt.Call @add_one(%3) on devices [1] : (!array_t1) -> !array_t1
    return %1, %2 : !array_t0, !array_t1
  }

  func.func private @add_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}"""
    args2 = [jax.device_put(x, jax.devices()[0])]
    expected_output2 = [x + 2.0, x + 3.0]
    ex2 = (mlir_module_str2, args2, expected_output2)
    return ex2


def test_interpreter_basic():
    devices = jax.devices()
    print(devices)

    client = devices[0].client
    mlir_module, args, expected_output = create_simple_call_program()

    print("\n" + "-" * 80)
    print("Creating ProgramInterpreter")
    print("-" * 80)

    interpreter = _jax.ifrt_interpreter.ProgramInterpreter.create(
        mlir_module=mlir_module, client=client, devices=devices
    )
    print("Created ProgramInterpreter")

    program_name = interpreter.get_program_name()
    print(f"Program name: {program_name}")

    print("\n" + "-" * 80)
    print("Executing Program")
    print("-" * 80)

    outputs = interpreter.execute(args)

    if all(jnp.allclose(output, expected) for output, expected in zip(outputs, expected_output, strict=True)):
        print()
        print("Output matches expected result!")
    else:
        print()
        print("Output does NOT match expected result")

    print("Expected output:")
    print(expected_output)
    print("Output:")
    print(outputs)


def main():
    test_interpreter_basic()


if __name__ == "__main__":
    main()
