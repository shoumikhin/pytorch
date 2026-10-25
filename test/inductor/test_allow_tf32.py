# Owner(s): ["module: inductor"]

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    device_interfaces,
    DeviceInterface,
    get_interface_for_device,
    register_interface_for_device,
    XpuInterface,
)
from torch._inductor import config
from torch._inductor.codecache import compiled_fx_graph_hash, FxGraphHashDetails
from torch._inductor.graph import GraphLowering
from torch._inductor.heuristics.template.triton import MMTemplateConfigMixin
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel.conv import conv_template_allow_tf32
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import recover_orig_fp32_precision


class _TensorNode:
    def __init__(self, device: torch.device, size: tuple[int, ...] = (2, 2)) -> None:
        self._device = device
        self._size = size

    def get_device(self) -> torch.device:
        return self._device

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_size(self) -> tuple[int, ...]:
        return self._size

    def get_stride(self) -> tuple[int, ...]:
        return (self._size[-1], 1)


class _MutableAllowTf32Interface(DeviceInterface):
    enabled = False
    conv_enabled = False

    @staticmethod
    def allow_tf32() -> bool:
        return _MutableAllowTf32Interface.enabled

    @staticmethod
    def conv_allow_tf32() -> bool:
        return _MutableAllowTf32Interface.conv_enabled

    @staticmethod
    def is_available() -> bool:
        return True


class _AllowTf32Interface(DeviceInterface):
    @staticmethod
    def allow_tf32() -> bool:
        return True

    @staticmethod
    def conv_allow_tf32() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class TestAllowTf32Interface(TestCase):
    def test_device_interface_default_false(self):
        self.assertFalse(DeviceInterface.allow_tf32())

    def test_cuda_allow_tf32_follows_fp32_precision(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            self.assertTrue(CudaInterface.allow_tf32())
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            self.assertFalse(CudaInterface.allow_tf32())
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    @recover_orig_fp32_precision
    def test_device_interface_conv_default_follows_cudnn_precision(self):
        # The cuDNN-wide precision overrides the generic one, which it inherits
        # when unset.
        with torch.backends.flags(fp32_precision="ieee"):
            self.assertFalse(DeviceInterface.conv_allow_tf32())
            torch._C._set_fp32_precision_setter("cuda", "all", "tf32")
            try:
                self.assertTrue(DeviceInterface.conv_allow_tf32())
            finally:
                torch._C._set_fp32_precision_setter("cuda", "all", "none")
        with torch.backends.flags(fp32_precision="tf32"):
            self.assertTrue(DeviceInterface.conv_allow_tf32())
        # The conv precision does not count.
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        with torch.backends.flags(fp32_precision="none"):
            self.assertFalse(DeviceInterface.conv_allow_tf32())

    @recover_orig_fp32_precision
    def test_cuda_conv_allow_tf32_follows_conv_precision(self):
        # Eager cuDNN convolutions use TF32 by default while the cuDNN-wide
        # precision stays "none". ROCm keeps following the cuDNN-wide one, which
        # inherits the generic one.
        cuda = torch.version.hip is None
        with mock.patch.object(torch.backends.cudnn, "is_available", return_value=True):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            with torch.backends.flags(fp32_precision="ieee"):
                self.assertEqual(CudaInterface.conv_allow_tf32(), cuda)
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            with torch.backends.flags(fp32_precision="tf32"):
                self.assertEqual(CudaInterface.conv_allow_tf32(), not cuda)

    @recover_orig_fp32_precision
    def test_cuda_conv_allow_tf32_is_off_where_eager_skips_cudnn_tf32(self):
        # Without cuDNN, eager would run most convolutions as a matmul, even
        # under a "tf32" matmul precision. ROCm follows the cuDNN-wide
        # precision, "none" here.
        cuda = torch.version.hip is None
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with mock.patch.object(torch.backends.cudnn, "is_available", return_value=True):
            self.assertEqual(CudaInterface.conv_allow_tf32(), cuda)
            with torch.backends.cudnn.flags(enabled=False):
                self.assertFalse(CudaInterface.conv_allow_tf32())
        with mock.patch.object(
            torch.backends.cudnn, "is_available", return_value=False
        ):
            self.assertFalse(CudaInterface.conv_allow_tf32())

    def test_nvidia_tf32_override_disables_conv_template_tf32(self):
        # Inductor reads it once, when its config loads.
        code = "import torch._inductor.config as c; print(c.disable_conv_template_tf32)"
        # Only on CUDA builds.
        nvidia = torch.version.cuda is not None
        for value, expected in (("0", nvidia), ("1", False)):
            env = {**os.environ, "NVIDIA_TF32_OVERRIDE": value}
            out = subprocess.check_output([sys.executable, "-c", code], env=env)
            self.assertEqual(out.decode().strip(), str(expected))

    def test_xpu_allow_tf32_matches_mkldnn(self):
        # Without USE_XPU the raw flag is None. The interface still returns a bool.
        self.assertEqual(
            XpuInterface.allow_tf32(), bool(torch.backends.mkldnn.allow_tf32)
        )

    @unittest.skipUnless(
        torch.xpu._is_compiled(), "oneDNN TF32 flag is a no-op without USE_XPU"
    )
    def test_xpu_allow_tf32_follows_mkldnn_flag(self):
        with torch.backends.mkldnn.flags(allow_tf32=True):
            self.assertTrue(XpuInterface.allow_tf32())
        with torch.backends.mkldnn.flags(allow_tf32=False):
            self.assertFalse(XpuInterface.allow_tf32())


class TestAllowTf32ExtraKwargs(TestCase):
    mixin = MMTemplateConfigMixin()

    def setUp(self):
        super().setUp()
        self._prev_pua = device_interfaces.get("privateuseone")
        gm = make_fx(lambda: torch.zeros(2, 2))()
        self.graph = GraphLowering(gm)
        self._graph_ctx = V.set_graph_handler(self.graph)
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)

    def tearDown(self):
        if self._prev_pua is None:
            device_interfaces.pop("privateuseone", None)
        else:
            device_interfaces["privateuseone"] = self._prev_pua
        super().tearDown()

    def _mm_kernel_inputs(
        self, device_type: str, m: int = 2, n: int = 2, k: int = 2
    ) -> MMKernelInputs:
        device = torch.device(device_type)
        mat1 = Buffer(
            name="mat1",
            layout=FixedLayout(device, torch.float32, (m, k)),
        )
        mat2 = Buffer(
            name="mat2",
            layout=FixedLayout(device, torch.float32, (k, n)),
        )
        return MMKernelInputs([mat1, mat2])

    def test_registered_privateuse1_allow_tf32_true(self):
        register_interface_for_device("privateuseone", _AllowTf32Interface)
        kernel_inputs = self._mm_kernel_inputs("privateuseone")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertTrue(extra["ALLOW_TF32"])

    def test_unregistered_device_allow_tf32_false(self):
        # meta is a real device type with no DeviceInterface. A name torch.device
        # rejects never reaches get_interface_for_device.
        kernel_inputs = self._mm_kernel_inputs("meta")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertFalse(extra["ALLOW_TF32"])

    def test_default_device_interface_allow_tf32_false(self):
        register_interface_for_device("privateuseone", DeviceInterface)
        kernel_inputs = self._mm_kernel_inputs("privateuseone")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertFalse(extra["ALLOW_TF32"])

    @config.patch(disable_conv_template_tf32=False)
    def test_conv_template_allow_tf32_follows_device_interface(self):
        register_interface_for_device("privateuseone", _AllowTf32Interface)
        self.assertTrue(conv_template_allow_tf32("privateuseone"))
        # No device, or a device type with no DeviceInterface.
        self.assertFalse(conv_template_allow_tf32(None))
        self.assertFalse(conv_template_allow_tf32("meta"))

    @recover_orig_fp32_precision
    @config.patch(disable_conv_template_tf32=False)
    def test_conv_template_keeps_cuda_depthwise_convs_in_full_precision(self):
        # Other devices follow their interface for depthwise convolutions.
        register_interface_for_device("privateuseone", _AllowTf32Interface)
        self.assertTrue(conv_template_allow_tf32("privateuseone", depthwise=True))
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        rocm = torch.version.hip is not None
        with (
            torch.backends.flags(fp32_precision="tf32"),
            mock.patch.object(torch.backends.cudnn, "is_available", return_value=True),
            config.patch(disable_conv_template_tf32=False),
        ):
            self.assertTrue(conv_template_allow_tf32("cuda"))
            self.assertEqual(conv_template_allow_tf32("cuda", depthwise=True), rocm)
            self.assertEqual(
                conv_template_allow_tf32("cuda", cudnn_enabled=False), rocm
            )
            self.assertTrue(
                conv_template_allow_tf32("privateuseone", cudnn_enabled=False)
            )
            with config.patch(disable_conv_template_tf32=True):
                self.assertFalse(conv_template_allow_tf32("privateuseone"))
                self.assertFalse(conv_template_allow_tf32("cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_small_shapes_disable_allow_tf32(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            kernel_inputs = self._mm_kernel_inputs("cuda", m=2, n=2, k=2)
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertFalse(extra["ALLOW_TF32"])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_large_shapes_enable_allow_tf32(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            kernel_inputs = self._mm_kernel_inputs("cuda", m=64, n=1024, k=1024)
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertTrue(extra["ALLOW_TF32"])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    def test_xpu_unaffected_by_cuda_shape_threshold(self):
        # Platform check. On a non-XPU build both sides are False, so this does
        # not prove the shape gate was skipped. CPU CI coverage for that is
        # test_registered_privateuse1_allow_tf32_true, which uses a 2x2 shape.
        # CUDA would force ALLOW_TF32 off for these shapes. XPU follows the flag.
        kernel_inputs = self._mm_kernel_inputs("xpu", m=2, n=2, k=2)
        if torch.xpu._is_compiled():
            with torch.backends.mkldnn.flags(allow_tf32=True):
                extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertTrue(extra["ALLOW_TF32"])
            return
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertEqual(extra["ALLOW_TF32"], XpuInterface.allow_tf32())

    def test_cache_key_records_registered_allow_tf32(self):
        details = FxGraphHashDetails(None, [torch.zeros(2, 2)], {}, [])
        self.assertIn(("cpu", False), details.mm_template_allow_tf32)

    def test_cache_key_skips_unregistered_device(self):
        details = FxGraphHashDetails(None, [torch.zeros(2, 2, device="meta")], {}, [])
        recorded = [device for device, _flag in details.mm_template_allow_tf32]
        self.assertNotIn("meta", recorded)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cache_key_tracks_cuda_allow_tf32(self):
        tensor = torch.zeros(2, 2, device="cuda")
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            enabled = FxGraphHashDetails(None, [tensor], {}, [])
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            disabled = FxGraphHashDetails(None, [tensor], {}, [])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig
        self.assertIn(("cuda", True), enabled.mm_template_allow_tf32)
        self.assertIn(("cuda", False), disabled.mm_template_allow_tf32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @recover_orig_fp32_precision
    def test_cache_key_tracks_cuda_conv_allow_tf32(self):
        tensor = torch.zeros(2, 2, device="cuda")
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        enabled = FxGraphHashDetails(None, [tensor], {}, [])
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        disabled = FxGraphHashDetails(None, [tensor], {}, [])
        expected = torch.version.hip is None
        self.assertIn(("cuda", expected), enabled.conv_template_allow_tf32)
        self.assertIn(("cuda", False), disabled.conv_template_allow_tf32)

    def test_non_cuda_flag_changes_fx_graph_cache_key(self):
        # cuda_matmul_settings already hashes CUDA fp32_precision. This swaps
        # the CPU interface so a flag outside that tuple changes the cache key.
        self._check_cpu_flag_changes_cache_key("enabled", "mm_template_allow_tf32")

    def test_conv_flag_changes_fx_graph_cache_key(self):
        self._check_cpu_flag_changes_cache_key(
            "conv_enabled", "conv_template_allow_tf32"
        )

    def _check_cpu_flag_changes_cache_key(self, flag, field):
        get_interface_for_device("cpu")
        previous = device_interfaces["cpu"]
        gm = make_fx(lambda x: x * 2)(torch.zeros(2, 2))
        example = [torch.zeros(2, 2)]
        try:
            register_interface_for_device("cpu", _MutableAllowTf32Interface)
            setattr(_MutableAllowTf32Interface, flag, True)
            enabled = FxGraphHashDetails(gm, example, {}, [])
            enabled_key, _ = compiled_fx_graph_hash(gm, example, {}, [])
            setattr(_MutableAllowTf32Interface, flag, False)
            disabled = FxGraphHashDetails(gm, example, {}, [])
            disabled_key, _ = compiled_fx_graph_hash(gm, example, {}, [])
            self.assertEqual(getattr(enabled, field), (("cpu", True),))
            self.assertEqual(getattr(disabled, field), (("cpu", False),))
            self.assertNotEqual(enabled_key, disabled_key)
        finally:
            setattr(_MutableAllowTf32Interface, flag, False)
            register_interface_for_device("cpu", previous)


if __name__ == "__main__":
    run_tests()
