"""Test configuration shared across the suite.

PyCUDA is an optional GPU dependency. These tests must run on machines without
a GPU (including CI), so we install lightweight stub ``pycuda`` modules into
``sys.modules`` before any project module imports them. Individual tests then
patch ``rightnow_cli.bench.cuda`` / ``rightnow_cli.bench.SourceModule`` as
usual, keeping the GPU code paths exercised instead of skipped.
"""

import sys
import types
from unittest.mock import MagicMock

if "pycuda" not in sys.modules:
    pycuda_module = types.ModuleType("pycuda")
    driver_module = types.ModuleType("pycuda.driver")
    autoinit_module = types.ModuleType("pycuda.autoinit")
    compiler_module = types.ModuleType("pycuda.compiler")

    driver_module.init = MagicMock()
    driver_module.Device = MagicMock()
    driver_module.Context = MagicMock()
    driver_module.mem_alloc = MagicMock()
    driver_module.memcpy_htod = MagicMock()
    driver_module.memcpy_dtoh = MagicMock()
    driver_module.module_from_buffer = MagicMock()
    driver_module.Event = MagicMock()

    compiler_module.SourceModule = MagicMock()

    pycuda_module.driver = driver_module
    pycuda_module.autoinit = autoinit_module
    pycuda_module.compiler = compiler_module

    sys.modules.setdefault("pycuda", pycuda_module)
    sys.modules.setdefault("pycuda.driver", driver_module)
    sys.modules.setdefault("pycuda.autoinit", autoinit_module)
    sys.modules.setdefault("pycuda.compiler", compiler_module)
