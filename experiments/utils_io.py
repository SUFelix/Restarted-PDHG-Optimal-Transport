from __future__ import annotations
import csv
import dataclasses
import datetime
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import psutil
import scipy
from algorithms.models import LPProblem, EnvironmentInfo, CPUInfo, GPUInfo, RAMInfo

def gather_environment_info() -> EnvironmentInfo:
    """
    Collect hardware and software environment information.

    GPU detection uses nvidia-smi; sets gpu=None silently when not available.
    Returns a fully populated EnvironmentInfo dataclass (no plain dicts).
    """
    try:
        git_commit: Optional[str] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = None

    cpu = CPUInfo(
        processor=platform.processor(),
        cores_physical=psutil.cpu_count(logical=False) or 0,
        cores_logical=psutil.cpu_count(logical=True) or 0,
        frequency_mhz=psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
    )

    mem = psutil.virtual_memory()
    ram = RAMInfo(
        total_gb=round(mem.total / 1024**3, 2),
        available_at_start_gb=round(mem.available / 1024**3, 2),
    )

    gpu: Optional[List[GPUInfo]] = None
    try:
        nvidia_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        rows = [line for line in nvidia_out.splitlines() if line.strip()]
        if rows:
            gpu = []
            for line in rows:
                name, total_mem, avail_mem, driver = line.split(",")

                gpu.append(GPUInfo(
                    name=name.strip(),
                    vram_total_mb=int(total_mem.strip()),
                    vram_available_at_start_mb=int(avail_mem.strip()),  # Mapping des neuen Werts
                    driver_version=driver.strip(),
                ))
    except Exception:
        gpu = None

    gurobi_version: Optional[str] = None
    try:
        import gurobipy
        gurobi_version = ".".join(str(v) for v in gurobipy.gurobi.version())
    except ImportError:
        pass

    return EnvironmentInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        git_commit=git_commit,
        cpu=cpu,
        ram=ram,
        gpu=gpu,
        gurobi_version=gurobi_version,
    )

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types, dataclasses, and None-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_serializable(dataclasses.asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

def write_experiment_run(
    exp_dir: Union[str, Path],
    experiment_name: str,
    solver_name: str,
    solver_params: Any,
    problem: LPProblem,
    environment: Union[EnvironmentInfo, Dict[str, Any]],
    metrics: Dict[str, Any],
    sequences: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    csv_export: bool = False,
) -> Path:

    exp_dir = Path(exp_dir)

    safe_name = experiment_name.replace(" ", "_")
    if filename is None:
        filename = f"{safe_name}.json"
    out_path = exp_dir / filename

    out_dict = {
        "experiment_name": experiment_name,
        "solver": {
            "name": solver_name,
            **_to_serializable(solver_params),
        },
        "problem":{
            "name": problem.name,
            "shape":problem.A_shape,
            "cost_type": problem.cost_type,
            "dotmark_type": problem.dotmark_type

        },
        "environment": _to_serializable(environment),
        "metrics": _to_serializable(metrics),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    if sequences is not None:
        out_dict["sequences"] = _to_serializable(sequences)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    if csv_export:
        csv_path = exp_dir / f"{safe_name}.csv"
        # Only scalar values in CSV rows
        flat_data: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "solver_name": solver_name,
            "timestamp": out_dict["timestamp"],
            **{
                f"solver_{k}": v
                for k, v in _to_serializable(solver_params).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            **{
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(flat_data)

    return out_path


if __name__ == "__main__":
    info = gather_environment_info()
    print(json.dumps(dataclasses.asdict(info), indent=4))
