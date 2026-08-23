from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


_EXCLUDED_DIRS = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "agentic_tmp",
    "agentic-artifacts",
)


def log(level: str, msg: str, *args: object) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}")
    except Exception:
        print(f"{prefix} {level}: {msg}")


def _safe_patch_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Patch path escapes sandbox: {relative_path}")
    return candidate


def _copy_project(project_path: Path, sandbox_root: Path) -> None:
    for item in project_path.iterdir():
        if item.name in _EXCLUDED_DIRS:
            continue
        destination = sandbox_root / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=shutil.ignore_patterns(*_EXCLUDED_DIRS),
                copy_function=shutil.copy2,
            )
        else:
            shutil.copy2(item, destination)


def _inject_patches(sandbox_root: Path, patches: Mapping[str, str]) -> None:
    for relative_path, content in patches.items():
        patched_path = _safe_patch_path(sandbox_root, relative_path)
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        patched_path.write_text(content, encoding="utf-8")
        log("INFO", "Injected patch for %s into sandbox", relative_path)


def _container_logs(container: Any) -> str:
    output = container.logs(stream=False)
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def run_tests(
    project_path: str | Path = ".",
    test_command: str = "python -m pytest tests/ -v",
    patches_dict: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run accumulated patches in a Docker sandbox.

    Dependencies are installed with network access in a temporary image. The
    test container uses that image with networking disabled.
    """
    source_root = Path(project_path).resolve()
    patches = patches_dict or {}

    try:
        import docker
    except ImportError as error:
        return {
            "success": False,
            "returncode": 1,
            "logs": f"Docker SDK is not installed: {error}",
        }

    try:
        client = docker.from_env()
    except Exception as error:
        return {
            "success": False,
            "returncode": 1,
            "logs": f"Could not connect to Docker: {error}",
        }

    base_image = os.getenv("SANDBOX_IMAGE", "python:3.11-slim")
    sandbox_image = f"agentic-sandbox:{os.getpid()}-{int(time.time())}"
    install_container = None
    test_container = None
    image_created = False

    with tempfile.TemporaryDirectory(prefix="agentic-sandbox-", ignore_cleanup_errors=True) as temp_dir:
        sandbox_root = Path(temp_dir)
        log("INFO", "Spinning up secure sandbox at %s", sandbox_root)
        try:
            _copy_project(source_root, sandbox_root)
            _inject_patches(sandbox_root, patches)
        except (OSError, ValueError) as error:
            return {
                "success": False,
                "returncode": 1,
                "logs": f"Could not prepare sandbox: {error}",
            }

        try:
            requirements = sandbox_root / "requirements.txt"
            if requirements.exists():
                log("INFO", "Phase 1: Installing dependencies with network enabled")
                install_container = client.containers.run(
                    base_image,
                    command=["sh", "-c", "pip install --no-cache-dir -r /app/requirements.txt"],
                    volumes={str(sandbox_root): {"bind": "/app", "mode": "ro"}},
                    working_dir="/app",
                    detach=True,
                    network_disabled=False,
                )
                install_container.wait(timeout=300)
                install_container.reload()
                install_logs = _container_logs(install_container)
                install_exit = install_container.attrs["State"]["ExitCode"]
                if install_exit != 0:
                    return {
                        "success": False,
                        "returncode": install_exit,
                        "logs": f"Dependency install failed:\n{install_logs[-12000:]}",
                    }
                install_container.commit(repository="agentic-sandbox", tag=sandbox_image.split(":", 1)[1])
                image_created = True
                log("INFO", "Phase 1 complete: dependencies installed")
            else:
                sandbox_image = base_image

            log("INFO", "Phase 2: Running tests with network disabled")
            test_container = client.containers.run(
                sandbox_image,
                command=["sh", "-c", test_command],
                volumes={str(sandbox_root): {"bind": "/app", "mode": "rw"}},
                working_dir="/app",
                detach=True,
                network_disabled=True,
            )
            test_container.wait(timeout=300)
            test_container.reload()
            logs = _container_logs(test_container)
            exit_code = test_container.attrs["State"]["ExitCode"]
            return {
                "success": exit_code == 0,
                "returncode": exit_code,
                "logs": logs[-12000:],
            }
        except Exception as error:
            log("ERROR", "Sandbox execution failed: %s", error)
            return {
                "success": False,
                "returncode": 1,
                "logs": f"Sandbox execution failed: {error}",
            }
        finally:
            for container in (install_container, test_container):
                if container is not None:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
            if image_created:
                try:
                    client.images.remove(sandbox_image, force=True)
                except Exception:
                    pass
