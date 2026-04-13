"""
Hardware Metrics Collection for Model Experiments

Measures:
- GPU VRAM usage (peak allocated, peak reserved, per-GPU for multi-GPU)
- Model disk size (HuggingFace cache)
- GPU power draw and energy consumption (Watt-hours)
- CPU RSS peak memory
- System memory usage

Energy measurement priority:
1. NVML total_energy_consumption counter (most accurate, if GPU supports it)
2. nvidia-smi power.draw sampling with trapezoidal integration (fallback)

Falls back gracefully when GPU is not available.
"""

import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Optional torch import (for VRAM measurement)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional pynvml import (for NVML energy counters)
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# Optional psutil import (for CPU RSS monitoring)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HardwareMetrics:
    """Hardware resource metrics for an experiment run."""

    # GPU memory (bytes)
    gpu_vram_allocated_bytes: int = 0       # Peak allocated by PyTorch
    gpu_vram_reserved_bytes: int = 0        # Peak reserved by PyTorch (includes fragmentation)
    gpu_vram_total_bytes: int = 0           # Total VRAM on device

    # Friendly versions (GB)
    gpu_vram_allocated_gb: float = 0.0
    gpu_vram_reserved_gb: float = 0.0
    gpu_vram_total_gb: float = 0.0

    # Model disk size
    model_disk_size_bytes: int = 0
    model_disk_size_gb: float = 0.0
    model_cache_path: str = ""

    # Energy / power
    gpu_energy_wh: float = 0.0              # Total energy consumed during experiment
    gpu_avg_power_watts: float = 0.0        # Average power draw
    gpu_peak_power_watts: float = 0.0       # Peak power draw observed
    gpu_power_samples: int = 0              # Number of power readings taken

    # Timing
    model_load_time_seconds: float = 0.0    # Total time to load LLM + embedder
    llm_load_time_seconds: float = 0.0      # Time to load LLM into VRAM
    embedder_load_time_seconds: float = 0.0 # Time to load embedding model

    # CPU memory
    cpu_rss_peak_bytes: int = 0
    cpu_rss_peak_gb: float = 0.0

    # Energy method used
    gpu_energy_method: str = ""          # "nvml", "power_sampling", or "none"

    # System
    gpu_name: str = ""
    gpu_device_id: int = 0
    gpu_count: int = 0                   # Number of GPUs visible
    cuda_version: str = ""

    # Machine identification
    hostname: str = ""
    cpu_model: str = ""
    os_platform: str = ""                # e.g. "Linux-4.4.0-aarch64" or "Linux-5.15.0-x86_64"


def _bytes_to_gb(b: int) -> float:
    return round(b / (1024 ** 3), 3)


def get_machine_info() -> dict:
    """Collect machine identification: hostname, CPU model, OS platform, GPU count."""
    info: dict = {
        "hostname": socket.gethostname(),
        "os_platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "cpu_model": "",
        "gpu_count": 0,
    }

    # CPU model from /proc/cpuinfo (Linux) or platform fallback
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        # ARM chips may use "Hardware" or "Model" instead
        if not info["cpu_model"]:
            for line in cpuinfo.splitlines():
                if line.lower().startswith(("hardware", "model\t")):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"

    # GPU count
    if HAS_TORCH and torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()

    return info


# =============================================================================
# GPU VRAM Measurement
# =============================================================================

def get_gpu_vram_snapshot(device_id: int = 0) -> dict:
    """Take a snapshot of current GPU VRAM usage via PyTorch."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    device = torch.device(f"cuda:{device_id}")
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
        "total": torch.cuda.get_device_properties(device).total_memory,
    }


def reset_gpu_peak_stats(device_id: int = 0) -> None:
    """Reset peak memory tracking. Call BEFORE the experiment starts."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        # CUDA context may not be initialized yet (no kernels launched).
        # Peak stats start at zero anyway, so safe to skip.
        pass


def get_gpu_info(device_id: int = 0) -> dict:
    """Get GPU name, CUDA version, total VRAM."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"name": "N/A", "cuda_version": "N/A", "total_gb": 0}

    device = torch.device(f"cuda:{device_id}")
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "cuda_version": torch.version.cuda or "unknown",
        "total_bytes": props.total_memory,
        "total_gb": _bytes_to_gb(props.total_memory),
    }


# =============================================================================
# Model Disk Size
# =============================================================================

def get_model_disk_size(model_id: str) -> dict:
    """Measure total disk size of a HuggingFace model in the local cache.

    Checks both the HF_HOME/hub cache and the default ~/.cache/huggingface/hub.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    # HF cache structure: models--org--name
    safe_name = f"models--{model_id.replace('/', '--')}"
    model_cache = hub_dir / safe_name

    if not model_cache.exists():
        # Try alternate cache locations
        for alt in [
            hub_dir / model_id.replace("/", "--"),
            hub_dir / model_id.split("/")[-1],
        ]:
            if alt.exists():
                model_cache = alt
                break

    if not model_cache.exists():
        return {"path": str(model_cache), "size_bytes": 0, "size_gb": 0.0, "found": False}

    total = 0
    for f in model_cache.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass

    return {
        "path": str(model_cache),
        "size_bytes": total,
        "size_gb": _bytes_to_gb(total),
        "found": True,
    }


# =============================================================================
# GPU Power Monitoring (nvidia-smi based)
# =============================================================================

class GPUPowerMonitor:
    """Background thread that polls nvidia-smi for GPU power draw.

    Usage:
        monitor = GPUPowerMonitor(device_id=0, interval=1.0)
        monitor.start()
        # ... run experiment ...
        monitor.stop()
        print(f"Energy: {monitor.energy_wh:.3f} Wh")
    """

    def __init__(self, device_id: int = 0, interval: float = 1.0):
        self.device_id = device_id
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._power_readings: list[float] = []  # watts
        self._timestamps: list[float] = []
        self._available = shutil.which("nvidia-smi") is not None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Start background power monitoring."""
        if not self._available:
            return
        self._stop_event.clear()
        self._power_readings = []
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and compute totals."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _poll_loop(self) -> None:
        """Poll nvidia-smi in a loop."""
        while not self._stop_event.is_set():
            power = self._read_power()
            if power is not None:
                self._power_readings.append(power)
                self._timestamps.append(time.time())
            self._stop_event.wait(self.interval)

    def _read_power(self) -> float | None:
        """Read current GPU power draw in watts from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val and val != "[N/A]":
                    return float(val)
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    @property
    def energy_wh(self) -> float:
        """Total energy consumed in Watt-hours (trapezoidal integration)."""
        if len(self._timestamps) < 2:
            return 0.0

        total_joules = 0.0
        for i in range(1, len(self._timestamps)):
            dt = self._timestamps[i] - self._timestamps[i - 1]
            avg_power = (self._power_readings[i] + self._power_readings[i - 1]) / 2
            total_joules += avg_power * dt

        return total_joules / 3600  # joules -> Wh

    @property
    def avg_power_watts(self) -> float:
        if not self._power_readings:
            return 0.0
        return sum(self._power_readings) / len(self._power_readings)

    @property
    def peak_power_watts(self) -> float:
        if not self._power_readings:
            return 0.0
        return max(self._power_readings)

    @property
    def num_samples(self) -> int:
        return len(self._power_readings)


# =============================================================================
# NVML Energy Counter (preferred over nvidia-smi polling when available)
# =============================================================================

class NVMLEnergyCounter:
    """Uses NVML total_energy_consumption counter for precise energy measurement.

    This is more accurate than nvidia-smi power sampling because it uses the
    GPU's built-in hardware energy counter. Falls back gracefully if NVML is
    not available or the GPU doesn't support energy counters.
    """

    def __init__(self):
        self.available = False
        self._handles: list = []
        self._start_mj: dict[str, float] = {}

        if not HAS_PYNVML:
            return

        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            # Probe whether energy counter is supported
            for h in self._handles:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            self.available = True
        except Exception:
            self.available = False

    def start(self) -> None:
        """Record starting energy counter values."""
        if not self.available:
            return
        self._start_mj = {}
        for i, h in enumerate(self._handles):
            try:
                self._start_mj[str(i)] = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            except Exception:
                pass

    def stop(self) -> dict[str, float]:
        """Return energy consumed per GPU in Watt-hours since start()."""
        if not self.available:
            return {}
        result: dict[str, float] = {}
        for i, h in enumerate(self._handles):
            try:
                end_mj = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
                start_mj = self._start_mj.get(str(i), end_mj)
                delta_mj = max(0.0, end_mj - start_mj)
                result[str(i)] = (delta_mj / 1000.0) / 3600.0  # mJ -> J -> Wh
            except Exception:
                pass
        return result


# =============================================================================
# CPU RSS Monitor
# =============================================================================

class CPURSSMonitor:
    """Monitors peak CPU RSS (Resident Set Size) in a background thread.

    Uses psutil to sample RSS at regular intervals and tracks the peak.
    Falls back gracefully if psutil is not installed.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss: int = 0
        self._available = HAS_PSUTIL

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self.peak_rss = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop monitoring and return peak RSS in bytes."""
        if self._thread is None:
            return self.peak_rss
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        return self.peak_rss

    def _loop(self) -> None:
        proc = psutil.Process() if HAS_PSUTIL else None
        while not self._stop_event.is_set():
            try:
                if proc is not None:
                    rss = proc.memory_info().rss
                    if rss > self.peak_rss:
                        self.peak_rss = rss
            except Exception:
                pass
            self._stop_event.wait(self.interval)


# =============================================================================
# Convenience: Collect All Metrics
# =============================================================================

def collect_pre_experiment_metrics(
    model_id: str,
    device_id: int = 0,
) -> dict:
    """Collect metrics BEFORE the experiment (model disk size, GPU info).

    Call this after model is loaded but before running questions.
    """
    gpu_info = get_gpu_info(device_id)
    disk_info = get_model_disk_size(model_id)
    vram = get_gpu_vram_snapshot(device_id)

    return {
        "gpu_info": gpu_info,
        "disk_info": disk_info,
        "vram_after_load": vram,
    }


def collect_post_experiment_metrics(
    model_id: str,
    device_id: int = 0,
    power_monitor: GPUPowerMonitor | None = None,
    nvml_energy: NVMLEnergyCounter | None = None,
    cpu_monitor: CPURSSMonitor | None = None,
    model_load_time: float = 0.0,
    llm_load_time: float = 0.0,
    embedder_load_time: float = 0.0,
) -> HardwareMetrics:
    """Collect all metrics AFTER the experiment completes.

    Energy measurement priority:
    1. NVML counter (nvml_energy) - most accurate
    2. nvidia-smi power sampling (power_monitor) - fallback
    """
    gpu_info = get_gpu_info(device_id)
    disk_info = get_model_disk_size(model_id)
    vram = get_gpu_vram_snapshot(device_id)

    # Determine energy values: prefer NVML if available
    energy_wh = 0.0
    energy_method = "none"
    avg_power = 0.0
    peak_power = 0.0
    power_samples = 0

    if nvml_energy is not None and nvml_energy.available:
        nvml_results = nvml_energy.stop()
        energy_wh = nvml_results.get(str(device_id), 0.0)
        energy_method = "nvml"
        # Still use power_monitor for avg/peak power stats
        if power_monitor:
            avg_power = power_monitor.avg_power_watts
            peak_power = power_monitor.peak_power_watts
            power_samples = power_monitor.num_samples
    elif power_monitor:
        energy_wh = power_monitor.energy_wh
        avg_power = power_monitor.avg_power_watts
        peak_power = power_monitor.peak_power_watts
        power_samples = power_monitor.num_samples
        energy_method = "power_sampling"

    # CPU RSS
    cpu_rss = 0
    if cpu_monitor is not None:
        cpu_rss = cpu_monitor.stop()

    machine = get_machine_info()

    metrics = HardwareMetrics(
        # VRAM
        gpu_vram_allocated_bytes=vram.get("max_allocated", 0),
        gpu_vram_reserved_bytes=vram.get("max_reserved", 0),
        gpu_vram_total_bytes=vram.get("total", 0),
        gpu_vram_allocated_gb=_bytes_to_gb(vram.get("max_allocated", 0)),
        gpu_vram_reserved_gb=_bytes_to_gb(vram.get("max_reserved", 0)),
        gpu_vram_total_gb=_bytes_to_gb(vram.get("total", 0)),
        # Disk
        model_disk_size_bytes=disk_info.get("size_bytes", 0),
        model_disk_size_gb=disk_info.get("size_gb", 0.0),
        model_cache_path=disk_info.get("path", ""),
        # Power / Energy
        gpu_energy_wh=energy_wh,
        gpu_avg_power_watts=avg_power,
        gpu_peak_power_watts=peak_power,
        gpu_power_samples=power_samples,
        # Timing
        model_load_time_seconds=model_load_time,
        llm_load_time_seconds=llm_load_time,
        embedder_load_time_seconds=embedder_load_time,
        # CPU
        cpu_rss_peak_bytes=cpu_rss,
        cpu_rss_peak_gb=_bytes_to_gb(cpu_rss),
        # Energy method
        gpu_energy_method=energy_method,
        # System
        gpu_name=gpu_info.get("name", ""),
        gpu_device_id=device_id,
        gpu_count=machine["gpu_count"],
        cuda_version=gpu_info.get("cuda_version", ""),
        # Machine identification
        hostname=machine["hostname"],
        cpu_model=machine["cpu_model"],
        os_platform=machine["os_platform"],
    )

    return metrics
