"""Test suite for the CLI module.

These tests exercise the current command API defined in
``rightnow_cli.cli_enhanced``: the ``optimize`` command operates on a CUDA
kernel *file* (not a model name) and drives CacheManager, OpenRouterClient,
KernelAnalyzer, CUDACompiler, Benchmarker and CUDAProfiler.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from rightnow_cli.cli_enhanced import app, main
from rightnow_cli.openrouter import KernelCandidate


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def kernel_file(tmp_path):
    """Create a real CUDA kernel file for the optimize command."""
    path = tmp_path / "kernel.cu"
    path.write_text("__global__ void test_kernel() {}\n")
    return path


@pytest.fixture
def mock_cache_manager():
    """Create a mock CacheManager."""
    with patch("rightnow_cli.cli_enhanced.CacheManager") as mock_cache:
        cache_instance = MagicMock()
        cache_instance.get_cached_kernel.return_value = None
        cache_instance.has_api_key.return_value = True
        cache_instance.get_api_key.return_value = "test-api-key"
        mock_cache.return_value = cache_instance
        yield cache_instance


@pytest.fixture
def mock_openrouter_client():
    """Create a mock OpenRouterClient."""
    with patch("rightnow_cli.cli_enhanced.OpenRouterClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_kernel_analyzer():
    """Create a mock KernelAnalyzer."""
    with patch("rightnow_cli.cli_enhanced.KernelAnalyzer") as mock_analyzer:
        analyzer_instance = MagicMock()
        mock_analyzer.return_value = analyzer_instance
        analyzer_instance.analyze_kernel.return_value = {
            "kernel_name": "test_kernel",
            "parameters": ["a", "b"],
            "launch_bounds": None,
            "shared_memory_usage": "None detected",
            "global_accesses": {"reads": 2, "writes": 1, "likely_coalesced": True},
            "patterns": ["matrix multiplication"],
            "optimization_opportunities": [],
            "performance_hints": [],
            "complexity": "Low",
            "synchronization": {"syncthreads": 0, "atomic_ops": 0},
            "arithmetic_intensity": 1.5,
        }
        yield analyzer_instance


@pytest.fixture
def mock_cuda_compiler():
    """Create a mock CUDACompiler."""
    with patch("rightnow_cli.cli_enhanced.CUDACompiler") as mock_compiler:
        compiler_instance = MagicMock()
        mock_compiler.return_value = compiler_instance
        compiler_instance.compile_kernel.return_value = {
            "code": "__global__ void test_kernel() {}",
            "operation": "test_kernel",
            "ptx_code": "mock_ptx_code",
            "full_code": "mock_cuda_code",
        }
        yield compiler_instance


@pytest.fixture
def mock_benchmarker():
    """Create a mock Benchmarker."""
    with patch("rightnow_cli.cli_enhanced.Benchmarker") as mock_bench:
        bench_instance = MagicMock()
        mock_bench.return_value = bench_instance
        bench_instance.benchmark_kernel_standalone.return_value = {
            "avg_time_ms": 10.5,
            "min_time_ms": 10.0,
            "max_time_ms": 11.0,
            "std_dev_ms": 0.5,
            "success": True,
        }
        bench_instance.benchmarker_class = mock_bench
        yield bench_instance


@pytest.fixture
def mock_cuda_profiler():
    """Create a mock CUDAProfiler."""
    with patch("rightnow_cli.cli_enhanced.CUDAProfiler") as mock_profiler:
        profiler_instance = MagicMock()
        mock_profiler.return_value = profiler_instance
        profiler_instance.profile_kernel.return_value = {
            "memory_usage_mb": 256.0,
            "occupancy": 0.85,
            "register_usage": 48,
        }
        yield profiler_instance


def _candidate(operation="test_kernel", code="__global__ void test_kernel() {}"):
    """Build a KernelCandidate matching the current OpenRouter API."""
    return KernelCandidate(
        code=code,
        operation=operation,
        constraints={"max_registers": 255},
        metadata={"variant": 1},
    )


class TestOptimizeCommand:
    """Test the optimize command."""

    def test_optimize_success(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test successful optimization."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cache_manager.get_cached_kernel.return_value = None
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 0
        assert "Optimization Results" in result.stdout
        mock_openrouter_client.generate_kernel_optimizations.assert_called()
        mock_cuda_compiler.compile_kernel.assert_called()
        mock_benchmarker.benchmark_kernel_standalone.assert_called()
        mock_cuda_profiler.profile_kernel.assert_called()
        mock_cache_manager.cache_kernel.assert_called()

    def test_optimize_with_missing_api_key(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test optimization when API key is missing."""
        mock_cache_manager.has_api_key.return_value = False
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        # Simulate user input for API key
        result = runner.invoke(app, ["optimize", str(kernel_file)], input="test-api-key\n")

        assert result.exit_code == 0
        assert "OpenRouter API key not found" in result.stdout
        assert "API key saved successfully" in result.stdout
        mock_cache_manager.save_api_key.assert_called_once_with("test-api-key")

    def test_optimize_passes_model_name(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test that the selected model name reaches the OpenRouter client."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(
            app, ["optimize", str(kernel_file), "--model_name", "anthropic/claude-3"]
        )

        assert result.exit_code == 0
        calls = mock_openrouter_client.generate_kernel_optimizations.call_args_list
        assert all(c.kwargs["model_name"] == "anthropic/claude-3" for c in calls)

    def test_optimize_with_cached_kernel(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
    ):
        """Test optimization short-circuits when a cached kernel exists."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cache_manager.get_cached_kernel.return_value = {
            "code": "__global__ void cached_kernel() {}"
        }

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 0
        assert "Using cached optimized kernel" in result.stdout
        mock_openrouter_client.generate_kernel_optimizations.assert_not_called()

    def test_optimize_force_regenerate(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test optimization with force regenerate flag."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cache_manager.get_cached_kernel.return_value = {"code": "cached"}
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file), "--force"])

        assert result.exit_code == 0
        mock_cache_manager.get_cached_kernel.assert_not_called()
        assert mock_openrouter_client.generate_kernel_optimizations.called

    def test_optimize_compilation_failure(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test handling of compilation failures (no valid variants)."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cuda_compiler.compile_kernel.side_effect = Exception("Compilation error")
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 1
        assert "No valid optimization variants generated" in result.stdout

    def test_optimize_writes_output_file(
        self,
        runner,
        kernel_file,
        tmp_path,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test that the optimized kernel is written to --output."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]
        output_file = tmp_path / "optimized.cu"

        result = runner.invoke(app, ["optimize", str(kernel_file), "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.read_text() == "__global__ void test_kernel() {}"

    def test_optimize_keyboard_interrupt(
        self, runner, kernel_file, mock_cache_manager, mock_kernel_analyzer
    ):
        """Test handling of keyboard interrupt."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_kernel_analyzer.analyze_kernel.side_effect = KeyboardInterrupt()

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 0
        assert "Optimization cancelled by user" in result.stdout

    def test_optimize_general_error(
        self, runner, kernel_file, mock_cache_manager, mock_kernel_analyzer
    ):
        """Test handling of general errors."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_kernel_analyzer.analyze_kernel.side_effect = Exception("Test error")

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 1
        assert "Error: Test error" in result.stdout


class TestAnalyzeCommand:
    """Test the analyze command."""

    def test_analyze_success(self, runner, kernel_file, mock_kernel_analyzer):
        """Test analyzing a kernel."""
        result = runner.invoke(app, ["analyze", str(kernel_file)])

        assert result.exit_code == 0
        assert "test_kernel" in result.stdout
        mock_kernel_analyzer.analyze_kernel.assert_called_once()

    def test_analyze_verbose(self, runner, kernel_file, mock_kernel_analyzer):
        """Test verbose analysis output."""
        result = runner.invoke(app, ["analyze", str(kernel_file), "--verbose"])

        assert result.exit_code == 0
        assert "Detailed Metrics" in result.stdout


class TestConfigCommand:
    """Test the config command."""

    def test_config_show(self, runner, mock_cache_manager):
        """Test showing configuration."""
        mock_cache_manager.get_config.return_value = {"openrouter_api_key": "xxx"}
        mock_cache_manager.config_dir = Path("/home/user/.ultrathink-cli")
        mock_cache_manager.cache_dir = Path("/home/user/.ultrathink-cli/cache")
        mock_cache_manager.count_cached_kernels.return_value = 5

        result = runner.invoke(app, ["config", "--show"])

        assert result.exit_code == 0
        assert "RightNow CLI Configuration" in result.stdout
        assert "Config:" in result.stdout
        assert "Cache:" in result.stdout
        assert "Cached kernels: 5" in result.stdout

    def test_config_reset_api_key(self, runner, mock_cache_manager):
        """Test resetting API key."""
        result = runner.invoke(app, ["config", "--reset-api-key"], input="new-api-key\n")

        assert result.exit_code == 0
        assert "API key updated successfully" in result.stdout
        mock_cache_manager.save_api_key.assert_called_once_with("new-api-key")

    def test_config_clear_cache_confirmed(self, runner, mock_cache_manager):
        """Test clearing cache with confirmation."""
        result = runner.invoke(app, ["config", "--clear-cache"], input="y\n")

        assert result.exit_code == 0
        assert "Cache cleared successfully" in result.stdout
        mock_cache_manager.clear_cache.assert_called_once()

    def test_config_clear_cache_cancelled(self, runner, mock_cache_manager):
        """Test clearing cache cancelled."""
        result = runner.invoke(app, ["config", "--clear-cache"], input="n\n")

        assert result.exit_code == 0
        assert "Cache cleared successfully" not in result.stdout
        mock_cache_manager.clear_cache.assert_not_called()

    def test_config_multiple_options(self, runner, mock_cache_manager):
        """Test multiple config options."""
        mock_cache_manager.get_config.return_value = {}
        mock_cache_manager.count_cached_kernels.return_value = 0

        result = runner.invoke(app, ["config", "--show", "--reset-api-key"], input="new-key\n")

        assert result.exit_code == 0
        assert "RightNow CLI Configuration" in result.stdout
        assert "API key updated successfully" in result.stdout


class TestBenchmarkCommand:
    """Test the benchmark command."""

    def test_benchmark_command_placeholder(self, runner, kernel_file):
        """Test that the benchmark command reports the coming-soon notice."""
        result = runner.invoke(app, ["benchmark", str(kernel_file)])

        assert result.exit_code == 0
        assert "coming soon" in result.stdout


class TestVersionCommand:
    """Test the version command."""

    def test_version_success(self, runner):
        """Test version command success."""
        with patch("importlib.metadata.version", return_value="1.0.0"):
            result = runner.invoke(app, ["version"])

            assert result.exit_code == 0
            assert "RightNow CLI" in result.stdout
            assert "1.0.0" in result.stdout

    def test_version_development(self, runner):
        """Test version command in development mode."""
        with patch("importlib.metadata.version", side_effect=Exception("Not installed")):
            result = runner.invoke(app, ["version"])

            assert result.exit_code == 0
            assert "RightNow CLI" in result.stdout
            assert "development" in result.stdout


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test main function calls app."""
        with patch("rightnow_cli.cli_enhanced.app") as mock_app:
            main()
            mock_app.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_optimize_custom_iterations(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test optimization with custom iterations."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file), "--iterations", "50"])

        assert result.exit_code == 0
        mock_benchmarker.benchmarker_class.assert_called_with(iterations=50)

    def test_optimize_custom_variants(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test optimization with a custom number of variants."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file), "--variants", "4"])

        assert result.exit_code == 0
        assert mock_openrouter_client.generate_kernel_optimizations.call_count == 4

    def test_optimize_no_valid_kernels(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test optimization when no generated kernels compile."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cuda_compiler.compile_kernel.side_effect = Exception("Compilation error")
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(app, ["optimize", str(kernel_file)])

        assert result.exit_code == 1
        assert "No valid optimization variants generated" in result.stdout

    def test_optimize_constraints_passed(
        self,
        runner,
        kernel_file,
        mock_cache_manager,
        mock_openrouter_client,
        mock_kernel_analyzer,
        mock_cuda_compiler,
        mock_benchmarker,
        mock_cuda_profiler,
    ):
        """Test that register/shared-memory constraints reach the client."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_openrouter_client.generate_kernel_optimizations.return_value = [_candidate()]

        result = runner.invoke(
            app,
            [
                "optimize",
                str(kernel_file),
                "--max-registers",
                "64",
                "--shared-memory",
                "32",
                "--gpu",
                "sm_86",
            ],
        )

        assert result.exit_code == 0
        constraints = mock_openrouter_client.generate_kernel_optimizations.call_args.kwargs[
            "constraints"
        ]
        assert constraints["max_registers"] == 64
        assert constraints["shared_memory_kb"] == 32
        assert constraints["target_gpu"] == "sm_86"
