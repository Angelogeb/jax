#!/usr/bin/env python3
"""Comprehensive test for IFRT ArrayType Python bindings."""

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func, ifrt

import jax
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P


def test_basic_ifrt_bindings():
    """Test basic IFRT dialect bindings."""
    print("=" * 70)
    print("Test 1: Basic IFRT Dialect Bindings")
    print("=" * 70)

    with ir.Context() as ctx:
        ifrt.register_dialect(ctx)

        with ir.Location.unknown(ctx):
            # Create a RankedTensorType for shape (8, 16, f32)
            shape = ir.RankedTensorType.get([8, 16], ir.F32Type.get())
            print(f"✓ Created shape: {shape}")

            # Create DevicesAttr
            devices_attr = ifrt.DevicesAttr.get([0, 1, 2, 3], ctx)
            print(f"✓ Created DevicesAttr: {devices_attr}")
            print(f"  Device IDs: {devices_attr.ids}")

            # Create UnspecifiedShardingAttr
            sharding_attr = ifrt.UnspecifiedShardingAttr.get(ctx)
            print(f"✓ Created UnspecifiedShardingAttr: {sharding_attr}")

            # Create memory kind attribute
            memory_kind_attr = ir.StringAttr.get("device", ctx)
            print(f"✓ Created memory kind: {memory_kind_attr}")

            # Create IFRT ArrayType
            array_type = ifrt.ArrayType.get(
                shape=shape,
                sharding_attr=sharding_attr,
                devices_attr=devices_attr,
                memory_kind_attr=memory_kind_attr,
                layout_attr=None,
            )

            print("\n=== IFRT ArrayType Created Successfully! ===")
            print(f"Array Type: {array_type}")
            print(f"  Shape: {array_type.shape}")
            print(f"  Sharding: {array_type.sharding_attr}")
            print(f"  Devices: {array_type.devices_attr}")
            print(f"  Memory Kind: {array_type.memory_kind_attr}")
            print(f"  Layout: {array_type.layout_attr}")

            # Create ControlType
            control_type = ifrt.ControlType.get(ctx)
            print(f"\n✓ Created ControlType: {control_type}")


def test_sharding_param_attr():
    """Test ShardingParamAttr bindings."""
    print("\n" + "=" * 70)
    print("Test 2: ShardingParamAttr Bindings")
    print("=" * 70)

    with ir.Context() as ctx:
        ifrt.register_dialect(ctx)

        with ir.Location.unknown(ctx):
            # Create a ShardingParamAttr
            # Example: 2x1x3 to [1,0] on 3x2
            # This means shard a rank-3 tensor into 2 slices in dim-0 and 3 slices in dim-2
            # The 6 slices will be distributed to 6 logical devices
            dim_shards = [2, 1, 3]
            permutation = [1, 0]  # minor to major
            axis_sizes = [3, 2]  # mesh dimensions

            sharding_param_attr = ifrt.ShardingParamAttr.get(
                dim_shards, permutation, axis_sizes, ctx
            )
            print(f"✓ Created ShardingParamAttr: {sharding_param_attr}")
            print(f"  dim_shards: {sharding_param_attr.dim_shards}")
            print(f"  permutation: {sharding_param_attr.permutation}")
            print(f"  axis_sizes: {sharding_param_attr.axis_sizes}")

            # Create ArrayType with ShardingParamAttr
            shape = ir.RankedTensorType.get([8, 16, 24], ir.F32Type.get())
            devices_attr = ifrt.DevicesAttr.get([0, 1, 2, 3, 4, 5], ctx)
            memory_kind_attr = ir.StringAttr.get("device", ctx)

            array_type = ifrt.ArrayType.get(
                shape=shape,
                sharding_attr=sharding_param_attr,
                devices_attr=devices_attr,
                memory_kind_attr=memory_kind_attr,
                layout_attr=None,
            )

            print("\n=== IFRT ArrayType with ShardingParamAttr Created! ===")
            print(f"Array Type: {array_type}")
            print(f"  Shape: {array_type.shape}")
            print(f"  Sharding: {array_type.sharding_attr}")
            print(f"  Devices: {array_type.devices_attr}")


def test_ifrt_call_op():
    """Test building IFRT Call operation (from verify_call.mlir main)."""
    print("\n" + "=" * 70)
    print("Test 3: IFRT Call Operation")
    print("=" * 70)

    # Use JAX's make_ir_context() which properly registers all dialects including func
    ctx = jax.interpreters.mlir.make_ir_context()
    ifrt.register_dialect(ctx)

    with ctx, ir.Location.unknown(ctx):
        # Create a module to hold the functions
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):
            # Create the IFRT ArrayType for 2x2xi32 tensor
            tensor_type = ir.RankedTensorType.get(
                [2, 2], ir.IntegerType.get_signless(32)
            )
            sharding_param = ifrt.ShardingParamAttr.get(
                [1, 1],  # dim_shards: 1x1
                [0],  # permutation: [0]
                [2],  # axis_sizes: on 2 devices
                ctx,
            )
            devices_attr = ifrt.DevicesAttr.get([0, 1], ctx)
            memory_kind = ir.StringAttr.get("device", ctx)

            ifrt_array_type = ifrt.ArrayType.get(
                shape=tensor_type,
                sharding_attr=sharding_param,
                devices_attr=devices_attr,
                memory_kind_attr=memory_kind,
                layout_attr=None,
            )

            print(f"✓ Created IFRT ArrayType (devices [0,1]): {ifrt_array_type}")

            # Create a second IFRT ArrayType for devices [2, 3]
            devices_attr_23 = ifrt.DevicesAttr.get([2, 3], ctx)
            ifrt_array_type_23 = ifrt.ArrayType.get(
                shape=tensor_type,
                sharding_attr=sharding_param,
                devices_attr=devices_attr_23,
                memory_kind_attr=memory_kind,
                layout_attr=None,
            )
            print(f"✓ Created IFRT ArrayType (devices [2,3]): {ifrt_array_type_23}")

            # Create the callee function: tensor<2x2xi32> -> tensor<2x2xi32>
            callee_type = ir.FunctionType.get(
                inputs=[tensor_type], results=[tensor_type]
            )
            callee_func = func.FuncOp(
                name="callee", type=callee_type, visibility="private"
            )

            # Add body to callee - simple identity return
            callee_block = callee_func.add_entry_block()
            with ir.InsertionPoint(callee_block):
                func.ReturnOp([callee_block.arguments[0]])

            print("✓ Created callee function")

            # Create the main function with ifrt.function attribute
            # Takes array on [0,1], returns array on [2,3]
            ifrt_function_type = ir.FunctionType.get(
                inputs=[ifrt_array_type], results=[ifrt_array_type_23]
            )
            main_func = func.FuncOp(
                name="main", type=ifrt_function_type, visibility="public"
            )

            # Add the ifrt.function attribute
            main_func.attributes["ifrt.function"] = ir.UnitAttr.get()

            print("✓ Created main function with ifrt.function attribute")

            # Add body to main with multiple operations
            call_block = main_func.add_entry_block()
            with ir.InsertionPoint(call_block):
                control_type = ifrt.ControlType.get(ctx)

                # First ifrt.Call operation on devices [0, 1]
                call_op_01 = ifrt.CallOp(
                    [ifrt_array_type],  # outputs (list of result types)
                    control_type,  # control_output (single type)
                    [call_block.arguments[0]],  # inputs
                    [],  # control_inputs
                    ir.FlatSymbolRefAttr.get("callee"),  # callee
                    devices_attr,  # devices
                )
                result_01 = call_op_01.results[0]
                ctrl_01 = call_op_01.results[1]

                print(f"✓ Created first ifrt.Call operation on devices [0,1]")

                # ifrt.CopyArrays to copy from [0,1] to [2,3]
                copy_op = ifrt.CopyArraysOp(
                    [ifrt_array_type_23],  # outputs
                    control_type,  # control_output
                    [result_01],  # inputs
                    [],  # control_inputs
                )
                copied_array = copy_op.results[0]
                ctrl_copy = copy_op.results[1]

                print(f"✓ Created ifrt.CopyArrays operation")

                # Second ifrt.Call operation on devices [2, 3]
                call_op_23 = ifrt.CallOp(
                    [ifrt_array_type_23],  # outputs (list of result types)
                    control_type,  # control_output (single type)
                    [copied_array],  # inputs
                    [],  # control_inputs
                    ir.FlatSymbolRefAttr.get("callee"),  # callee
                    devices_attr_23,  # devices
                )
                result_23 = call_op_23.results[0]

                print(f"✓ Created second ifrt.Call operation on devices [2,3]")

                func.ReturnOp([result_23])

            print("\n=== IFRT Call IR Module Created Successfully! ===")

            # Verify the module - should pass now with proper dialect registration
            try:
                module.operation.verify()
                print("✓ Module verification passed!")
            except Exception as e:
                print(f"✗ Module verification failed: {e}")

            print("\nModule IR:")
            module.operation.print(
                enable_debug_info=False,
                print_generic_op_form=False,
            )


def named_sharding_to_sharding_param(ns: jax.sharding.NamedSharding, ndim: int):
    return ifrt.to_sharding_param(ns._to_xla_hlo_sharding(ndim), ndim, ns.num_devices)


def sharding_param_to_attr(sp: ifrt.ShardingParam):
    return ifrt.ShardingParamAttr.get(
        sp.dim_shards, sp.minor_to_major.permutation, sp.minor_to_major.axis_sizes
    )


def test_sharding_param_python_bindings():
    """Test ShardingParam Python bindings from ir_py module."""
    mesh = AbstractMesh((2, 2, 2), ("x", "y", "z"))
    sharding = NamedSharding(mesh, P(("x", "z"), None))
    sharding_param = named_sharding_to_sharding_param(sharding, 3)
    print(sharding_param)
    return

    print("\n" + "=" * 70)
    print("Test 4: ShardingParam Python Bindings")
    print("=" * 70)

    # Create a MinorToMajor object
    minor_to_major = ifrt.MinorToMajor()
    minor_to_major.permutation = [1, 0]
    minor_to_major.axis_sizes = [3, 2]
    print(f"✓ Created MinorToMajor: {minor_to_major}")

    # Create a ShardingParam object
    dim_shards = [2, 1, 3]
    sharding_param = ifrt.ShardingParam(dim_shards, minor_to_major)
    print(f"✓ Created ShardingParam: {sharding_param}")
    print(f"  dim_shards: {sharding_param.dim_shards}")
    print(f"  minor_to_major: {sharding_param.minor_to_major}")
    print(f"  num_devices: {sharding_param.num_devices()}")
    print(f"  debug_string: {sharding_param.debug_string()}")

    # Test equality
    minor_to_major2 = ifrt.MinorToMajor()
    minor_to_major2.permutation = [1, 0]
    minor_to_major2.axis_sizes = [3, 2]
    sharding_param2 = ifrt.ShardingParam(dim_shards, minor_to_major2)
    print(f"\n✓ Equality test: {sharding_param == sharding_param2}")
    assert sharding_param == sharding_param2, "ShardingParams should be equal"

    # Create a simple replicated ShardingParam
    simple_minor_to_major = ifrt.MinorToMajor()
    simple_minor_to_major.permutation = [0]
    simple_minor_to_major.axis_sizes = [4]
    simple_sharding_param = ifrt.ShardingParam([1, 1], simple_minor_to_major)
    print(f"\n✓ Created simple replicated ShardingParam: {simple_sharding_param}")
    print(f"  num_devices: {simple_sharding_param.num_devices()}")


test_basic_ifrt_bindings()
test_sharding_param_attr()
test_ifrt_call_op()
test_sharding_param_python_bindings()
