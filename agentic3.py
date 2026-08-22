from __future__ import annotations

import os
import json
import time
import difflib
import io
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional

import requests as req
from github import Github, InputGitTreeElement
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Direct Google GenAI SDK
from google import genai
from google.genai import types

load_dotenv()


def log(level: str, msg: str, *args) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}", flush=True)
    except Exception:
        print(f"{prefix} {level}: {msg}", flush=True)


# ==========================================================
# Configuration
# ==========================================================

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMP = float(os.getenv("GEMINI_TEMP", "0.2"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

ALLOWED_ACTIONS = set(
    a.strip()
    for a in os.getenv("ALLOWED_ACTIONS", "create_pr").split(",")
    if a.strip()
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")
WORKFLOW_RUN_ID = os.getenv("workflow_run_id", "")
GITHUB_BASE_BRANCH = os.getenv("GITHUB_BASE_BRANCH", "main")

HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() == "true"

TEST_COMMAND = os.getenv("TEST_COMMAND", "python -m pytest tests/ -v")

AGENTIC_TMP_DIR = Path("agentic_tmp")
AGENTIC_TMP_DIR.mkdir(exist_ok=True)


# ==========================================================
# Gemini - direct SDK
# ==========================================================

def gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required")
    return genai.Client(api_key=GEMINI_API_KEY)


def invoke_structured(model_class, prompt: str):
    """Call Gemini directly and validate the response with Pydantic."""
    client = gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMP,
            response_mime_type="application/json",
            response_schema=model_class,
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")

    return model_class.model_validate_json(text)


# ==========================================================
# Pydantic structured outputs - KEEP THESE
# ==========================================================

class AnalyzeOutput(BaseModel):
    failure_summary: str = Field(description="Short description of the error")
    root_cause: str = Field(description="Detailed explanation of why it failed")
    target_file: str = Field(description="The EXACT file path from AVAILABLE FILES to fix")
    repair_strategy: str = Field(description="Step-by-step plan to fix it")
    confidence: float = Field(description="0.0 to 1.0 confidence score")


class EditChunk(BaseModel):
    start_line: int = Field(description="1-indexed start line")
    end_line: int = Field(description="1-indexed inclusive end line")
    replacement: str = Field(description="New code for these lines")


class FixOutput(BaseModel):
    edits: List[EditChunk] = Field(description="List of surgical edits")


# ==========================================================
# Audit / utility functions
# ==========================================================

def save_audit(iteration: int, name: str, prompt: str, response_text: str) -> Path:
    path = AGENTIC_TMP_DIR / f"iter_{iteration}_{name}.json"
    data = {
        "iteration": iteration,
        "name": name,
        "prompt": prompt,
        "response": response_text,
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def apply_edits(original_code: str, edits: List[dict]) -> str:
    lines = original_code.split("\n")
    edits = sorted(edits, key=lambda x: x.get("start_line", 0), reverse=True)

    for edit in edits:
        start = edit["start_line"] - 1
        end = edit["end_line"]
        replacement = edit["replacement"].split("\n")
        if start < 0 or end < start + 1 or end > len(lines):
            raise ValueError(
                f"Invalid edit range {edit['start_line']}-{edit['end_line']} "
                f"for file containing {len(lines)} lines"
            )
        lines[start:end] = replacement

    return "\n".join(lines)


def find_callers(target_file: str) -> dict:
    target_module = Path(target_file).stem
    callers = {}

    for py_file in Path(".").rglob("*.py"):
        if any(
            part.startswith(".") or part in ("__pycache__", ".venv", "venv", "agentic_tmp")
            for part in py_file.parts
        ):
            continue

        if str(py_file) == target_file:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            if f"from {target_module} import" in source or f"import {target_module}" in source:
                callers[str(py_file)] = source
        except Exception:
            pass

    return callers


def get_python_files() -> List[str]:
    files = []
    for p in Path(".").rglob("*.py"):
        if any(
            part.startswith(".") or part in ("__pycache__", ".venv", "venv", "agentic_tmp")
            for part in p.parts
        ):
            continue
        files.append(str(p))
    return sorted(files)


# ==========================================================
# GitHub Helpers
# ==========================================================

def github_client():
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub operations")
    from github import Auth
    return Github(auth=Auth.Token(GITHUB_TOKEN))


def create_branch_and_commit_multiple(
    repo_full_name: str,
    branch_name: str,
    patches_dict: dict,
    commit_message: str,
):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = GITHUB_BASE_BRANCH

    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    base_commit_sha = base_ref.object.sha
    base_commit = repo.get_git_commit(base_commit_sha)

    tree_items = []
    for file_path, file_content in patches_dict.items():
        blob = repo.create_git_blob(file_content, "utf-8")
        tree_items.append(
            InputGitTreeElement(
                path=file_path,
                mode="100644",
                type="blob",
                sha=blob.sha,
            )
        )

    new_tree = repo.create_git_tree(
        tree_items,
        base_tree=repo.get_git_tree(base_commit.tree.sha),
    )

    new_commit = repo.create_git_commit(
        commit_message,
        new_tree,
        [base_commit],
    )

    try:
        repo.get_git_ref(f"heads/{branch_name}").delete()
    except Exception:
        pass

    repo.create_git_ref(
        ref=f"refs/heads/{branch_name}",
        sha=new_commit.sha,
    )

    log("INFO", "Created branch %s -> commit %s", branch_name, new_commit.sha)


def open_pr_with_rca(
    repo_full_name: str,
    branch_name: str,
    pr_title: str,
    pr_body: str,
    draft: bool = False,
):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = GITHUB_BASE_BRANCH

    pr = repo.create_pull(
        title=pr_title,
        body=pr_body,
        head=branch_name,
        base=base_branch,
        draft=draft,
    )

    return pr.html_url, pr.number


# ==========================================================
# State - manually managed across the repair loop
# ==========================================================

class AgenticState(TypedDict, total=False):
    logs: str
    iteration_count: int
    success: bool
    repair_memory: Dict[str, Any]
    current_file: Optional[str]
    repair_strategy: Optional[str]
    rca_html_path: Optional[str]
    pr_url: Optional[str]
    approved: Optional[bool]
    lint_failed: Optional[bool]


# ==========================================================
# Steps - kept as normal Python functions
# ==========================================================

def fetch_logs_step(state: AgenticState) -> dict:
    log("INFO", "STEP[fetch_logs_step]: Fetching logs...")

    repair_memory = state.get("repair_memory")
    if not repair_memory:
        repair_memory = {
            "iterations": [],
            "repo_state": {},
            "context": {
                "original_logs": "",
                "latest_logs": "",
                "files_attempted": [],
            },
        }

    if state.get("iteration_count", 0) > 0:
        repair_memory["context"]["latest_logs"] = state.get("logs", "")
        return {
            "logs": state.get("logs", ""),
            "repair_memory": repair_memory,
        }

    repo_full_name = GITHUB_REPO
    workflow_run_id = WORKFLOW_RUN_ID

    if not repo_full_name:
        return {
            "logs": "No GITHUB_REPO configured.",
            "repair_memory": repair_memory,
        }

    try:
        gh = github_client()
        repo = gh.get_repo(repo_full_name)

        run = None
        if workflow_run_id:
            run = repo.get_workflow_run(int(workflow_run_id))
        else:
            failed_runs = repo.get_workflow_runs(status="failure")
            for candidate in failed_runs:
                if candidate.name and "doctor" in candidate.name.lower():
                    continue
                run = candidate
                break

        if run is None:
            return {
                "logs": "No failed workflow runs found.",
                "repair_memory": repair_memory,
            }

        logs_parts = []

        # Collect information about failed jobs/steps.
        for job in run.jobs():
            if job.conclusion == "failure":
                for step in job.steps:
                    if step.conclusion == "failure":
                        logs_parts.append(
                            f"=== Job: {job.name} | Step: {step.name} ===\n"
                            f"Status: {step.conclusion}\n"
                        )

        # Download the workflow log archive.
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        logs_url = (
            f"https://api.github.com/repos/{repo_full_name}"
            f"/actions/runs/{run.id}/logs"
        )

        resp = req.get(
            logs_url,
            headers=headers,
            allow_redirects=True,
            timeout=60,
        )

        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                for name in z.namelist():
                    content = z.read(name).decode("utf-8", errors="replace")

                    if any(
                        kw in name.lower()
                        for kw in ("run tests", "test", "build", "run")
                    ):
                        logs_parts.append(
                            f"=== {name} ===\n{content[-4000:]}\n"
                        )
                    elif not logs_parts:
                        logs_parts.append(
                            f"=== {name} ===\n{content[-2000:]}\n"
                        )
        else:
            logs_parts.append(
                f"Could not download workflow logs: HTTP {resp.status_code}\n"
                f"{resp.text[:1000]}"
            )

        real_logs = "\n".join(logs_parts)
        if len(real_logs) > 12000:
            real_logs = real_logs[-12000:]

        if not real_logs:
            real_logs = "Workflow run found but no failure logs could be extracted."

        repair_memory["context"]["original_logs"] = real_logs
        repair_memory["context"]["latest_logs"] = real_logs

        return {
            "logs": real_logs,
            "repair_memory": repair_memory,
        }

    except Exception as e:
        return {
            "logs": f"Failed to fetch logs from GitHub: {str(e)}",
            "repair_memory": repair_memory,
        }


def analyze_code_step(state: AgenticState) -> dict:
    log("INFO", "STEP[analyze_code_step]: Planning repair strategy...")

    repair_memory = state.get("repair_memory", {})
    iterations_history = json.dumps(
        repair_memory.get("iterations", []),
        indent=2,
    )

    repo_files = get_python_files()
    file_listing = ", ".join(repo_files) if repo_files else "unknown"

    prompt = f"""
You are a Senior Python Developer diagnosing a CI/CD failure.

ERROR LOGS:
{state.get('logs', '')}

AVAILABLE FILES IN REPO:
{file_listing}

PAST REPAIR ATTEMPTS (Do not repeat failed strategies):
{iterations_history}

Analyze the logs and determine the root cause.

Return:
- a short failure summary
- the detailed root cause
- the EXACT target file path from AVAILABLE FILES
- a step-by-step repair strategy
- confidence from 0.0 to 1.0

Do not invent a file path.
"""

    response = invoke_structured(AnalyzeOutput, prompt)

    save_audit(
        state.get("iteration_count", 0),
        "plan_response",
        prompt,
        response.model_dump_json(indent=2),
    )

    target_file = response.target_file
    repair_strategy = response.repair_strategy

    attempted = repair_memory.setdefault("context", {}).setdefault(
        "files_attempted", []
    )

    if target_file and target_file not in attempted:
        attempted.append(target_file)

    log(
        "INFO",
        "-> Plan: Fix %s | Strategy: %s | Confidence: %.2f",
        target_file,
        repair_strategy,
        response.confidence,
    )

    return {
        "current_file": target_file,
        "repair_strategy": repair_strategy,
        "repair_memory": repair_memory,
    }


def fix_code_step(state: AgenticState) -> dict:
    current_file = state.get("current_file")
    iteration = state.get("iteration_count", 0)
    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.setdefault("repo_state", {})

    log(
        "INFO",
        "STEP[fix_code_step]: Generating surgical patch for %s",
        current_file,
    )

    if not current_file:
        return {
            "lint_failed": True,
            "logs": "AI did not identify a target file.",
        }

    if current_file in repo_state:
        broken_code = repo_state[current_file]
        log("INFO", "-> Reading previously patched version from memory")
    else:
        try:
            broken_code = Path(current_file).read_text(encoding="utf-8")
        except Exception as e:
            return {
                "lint_failed": True,
                "logs": f"Could not read target file {current_file}: {e}",
            }

    callers = find_callers(current_file)
    callers_context = ""

    if callers:
        callers_context = "CALLERS OF THIS FILE (for context):\n"
        for file_path, code in callers.items():
            callers_context += (
                f"--- {file_path} ---\n"
                f"{code[-1500:]}\n\n"
            )

    prompt = f"""
You are a Senior Python Developer implementing a CI/CD fix.

TARGET FILE: {current_file}
REPAIR STRATEGY: {state.get('repair_strategy')}
ERROR LOGS: {state.get('logs', '')}

{callers_context}

CURRENT CODE ({current_file}):
{broken_code}

Generate ONLY surgical edits.
Do NOT rewrite the whole file unless absolutely necessary.
Use 1-indexed inclusive line numbers.
Preserve unrelated code.
"""

    response = invoke_structured(FixOutput, prompt)

    save_audit(
        iteration,
        "fix_response",
        prompt,
        response.model_dump_json(indent=2),
    )

    try:
        edits = [e.model_dump() for e in response.edits]
        patched_code = apply_edits(broken_code, edits)
        repo_state[current_file] = patched_code

        repair_memory.setdefault("iterations", []).append(
            {
                "iteration": iteration + 1,
                "target_file": current_file,
                "strategy": state.get("repair_strategy"),
                "edits_applied": edits,
                "result": "pending",
                "reason": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

        log("INFO", "-> Applied %d edits to %s", len(edits), current_file)

    except Exception as e:
        log("ERROR", "Failed to apply surgical edits: %s", str(e))
        return {
            "repair_memory": repair_memory,
            "lint_failed": True,
            "logs": f"Failed to apply AI edits: {e}",
        }

    return {
        "repair_memory": repair_memory,
        "iteration_count": iteration + 1,
        "lint_failed": False,
    }


def lint_check_step(state: AgenticState) -> dict:
    log("INFO", "STEP[lint_check_step]: Running syntax checks on patched files...")

    repo_state = state.get("repair_memory", {}).get("repo_state", {})

    for filepath, code in repo_state.items():
        tmp = AGENTIC_TMP_DIR / f"lint_{Path(filepath).name}"
        tmp.write_text(code, encoding="utf-8")

        result = subprocess.run(
            ["python3", "-m", "py_compile", str(tmp)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log(
                "WARNING",
                "Syntax error found in %s:\n%s",
                filepath,
                result.stderr,
            )

            iterations = state.get("repair_memory", {}).get("iterations", [])
            if iterations:
                iterations[-1]["result"] = "failed"
                iterations[-1]["reason"] = f"Syntax error: {result.stderr}"

            return {
                "lint_failed": True,
                "logs": f"Syntax error in {filepath}:\n{result.stderr}",
            }

    log("INFO", "-> All patched files passed syntax checks.")
    return {"lint_failed": False}


def test_code_step(state: AgenticState) -> dict:
    """Run the real project tests against the accumulated in-memory patches."""
    log("INFO", "STEP[test_code_step]: Testing ALL accumulated patches...")

    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.get("repo_state", {})

    if not repo_state:
        return {
            "success": False,
            "logs": "No patches available for testing.",
            "repair_memory": repair_memory,
        }

    backups: Dict[str, Optional[str]] = {}

    try:
        # Materialize patches temporarily so normal tests can execute them.
        for file_path, patched_code in repo_state.items():
            path = Path(file_path)

            if path.exists():
                backups[file_path] = path.read_text(encoding="utf-8")
            else:
                backups[file_path] = None

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(patched_code, encoding="utf-8")

        log("INFO", "-> Running: %s", TEST_COMMAND)

        result = subprocess.run(
            TEST_COMMAND,
            shell=True,
            capture_output=True,
            text=True,
        )

        logs = (
            f"Exit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout[-7000:]}\n\n"
            f"STDERR:\n{result.stderr[-7000:]}"
        )

        success = result.returncode == 0

        if success:
            log("INFO", "-> Tests: SUCCESS")
        else:
            log("WARNING", "-> Tests: FAILED. Gathering new logs...")

        iterations = repair_memory.get("iterations", [])
        if iterations:
            iterations[-1]["result"] = "passed" if success else "failed"
            iterations[-1]["reason"] = "Tests passed" if success else "Tests failed"

        return {
            "success": success,
            "logs": logs,
            "repair_memory": repair_memory,
        }

    finally:
        # Restore the original working tree. Patches remain in repo_state.
        for file_path, original in backups.items():
            path = Path(file_path)
            if original is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_text(original, encoding="utf-8")


def generate_rca_step(state: AgenticState) -> dict:
    log("INFO", "STEP[generate_rca_step]: Generating RCA and HTML Report...")

    repair_memory = state.get("repair_memory", {})
    iterations = repair_memory.get("iterations", [])
    repo_state = repair_memory.get("repo_state", {}) or {}

    rca_obj = {
        "summary": "Automated pipeline repair",
        "iterations_taken": len(iterations),
        "files_modified": list(repo_state.keys()),
        "final_status": "Success" if state.get("success") else "Max Iterations Reached",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    rca_path = AGENTIC_TMP_DIR / "final_rca.json"
    rca_path.write_text(json.dumps(rca_obj, indent=2), encoding="utf-8")

    diff_parts = []

    for file_path, patched_code in repo_state.items():
        try:
            orig_text = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            orig_text = ""

        diff = difflib.unified_diff(
            orig_text.splitlines(keepends=True),
            patched_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        diff_parts.append("".join(diff))

    patch_path = AGENTIC_TMP_DIR / "patch.diff"
    patch_path.write_text("\n\n".join(diff_parts) or "", encoding="utf-8")

    patched_zip_path = AGENTIC_TMP_DIR / "patched_files.zip"
    with zipfile.ZipFile(patched_zip_path, "w") as zf:
        for file_path, patched_code in repo_state.items():
            zf.writestr(file_path, patched_code)

    files_modified_html = "".join(
        f"<li>{f}</li>" for f in rca_obj["files_modified"]
    ) or "<li>None</li>"

    timeline_html = "".join(
        f"""
        <div style="margin-bottom:10px;">
          <strong>Iteration {it.get('iteration')} — {it.get('target_file')}</strong>
          <div class="muted">Strategy: {it.get('strategy')}</div>
          <pre>{json.dumps(it.get('edits_applied', []), indent=2)}</pre>
          <div class="muted">Result: {it.get('result')} {it.get('reason') or ''}</div>
        </div>
        """
        for it in iterations
    ) or "<p>No iterations recorded.</p>"

    html_content = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Pipeline Doctor Report</title>
<style>
body {{ font-family: Arial, sans-serif; color:#000; background:#fff; margin:24px; }}
.card {{ border:1px solid #ddd; padding:14px; margin-bottom:12px; }}
pre {{ background:#f8f8f8; padding:10px; overflow:auto; border:1px solid #e6e6e6; }}
.muted {{ color:#666; font-size:90%; }}
</style>
</head>
<body>
<h1>Pipeline Doctor Report</h1>
<div class="muted">Generated: {rca_obj['timestamp']}</div>

<div class="card">
<h2>Summary</h2>
<p><strong>Status:</strong> {rca_obj['final_status']}</p>
<p><strong>Iterations:</strong> {rca_obj['iterations_taken']}</p>
<p><strong>Files Modified:</strong></p>
<ul>{files_modified_html}</ul>
</div>

<div class="card">
<h2>Iteration Timeline</h2>
{timeline_html}
</div>

<div class="card">
<h2>Artifacts</h2>
<ul>
<li>final_rca.json</li>
<li>report.html</li>
<li>patch.diff</li>
<li>patched_files.zip</li>
</ul>
</div>

<div class="card">
<h2>Reproduction</h2>
<pre>{TEST_COMMAND}</pre>
</div>
</body>
</html>
"""

    html_path = AGENTIC_TMP_DIR / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    log("INFO", "-> RCA, patch.diff and patched_files.zip generated.")

    return {
        "rca_path": str(rca_path),
        "rca_html_path": str(html_path),
        "patch_path": str(patch_path),
        "patched_zip": str(patched_zip_path),
    }


def create_pr_step(state: AgenticState) -> dict:
    log("INFO", "STEP[create_pr_step]: Creating PR with ALL patches...")

    if "create_pr" not in ALLOWED_ACTIONS:
        log("INFO", "create_pr is not allowed by ALLOWED_ACTIONS")
        return {"pr_created": False, "reason": "action-not-allowed"}

    repo_state = state.get("repair_memory", {}).get("repo_state", {})
    if not repo_state:
        return {"pr_created": False, "reason": "no-patches"}

    repo_full_name = GITHUB_REPO
    if not repo_full_name:
        return {"pr_created": False, "reason": "missing-repo"}

    branch_name = f"agentic/auto-fix/run_{int(time.time())}"
    commit_msg = f"agentic: auto-fix ({len(repo_state)} files)"

    try:
        create_branch_and_commit_multiple(
            repo_full_name,
            branch_name,
            repo_state,
            commit_msg,
        )

        iterations = state.get("repair_memory", {}).get("iterations", [])
        last_iter = iterations[-1] if iterations else {}
        target_file = last_iter.get("target_file", "multiple files")
        strategy = last_iter.get("strategy", "Automated code repair")

        pr_title = f"Pipeline Doctor — Fix {target_file}"
        pr_body = (
            f"Summary: Automated fix for failing tests in {target_file}.\n\n"
            f"Root cause / strategy: {strategy}\n\n"
            f"Change: Applied surgical edits to {len(repo_state)} file(s).\n\n"
            "Verification: Tests passed. Full artifacts are available in the "
            "Pipeline Doctor workflow artifacts.\n"
        )

        pr_url, pr_number = open_pr_with_rca(
            repo_full_name,
            branch_name,
            pr_title,
            pr_body,
            draft=False,
        )

        log("INFO", "-> PR created: %s", pr_url)

        return {
            "pr_url": pr_url,
            "pr_number": pr_number,
            "pr_created": True,
        }

    except Exception as e:
        log("ERROR", "-> Failed to create PR: %s", str(e))
        return {
            "pr_created": False,
            "reason": str(e),
        }


# ==========================================================
# Main flow
# ==========================================================

def run_agent() -> AgenticState:
    log("INFO", "🚀 Starting Pipeline Doctor Agent (Gemini + Python)")
    log("INFO", "Pydantic: ENABLED")

    state: AgenticState = {
        "iteration_count": 0,
        "success": False,
        "logs": "",
        "repair_memory": {},
        "current_file": None,
        "repair_strategy": None,
        "approved": None,
        "lint_failed": False,
    }

    # Initial log fetch.
    state.update(fetch_logs_step(state))

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(
            "INFO",
            "================ ITERATION %d/%d ================",
            iteration,
            MAX_ITERATIONS,
        )

        # Analyze failure.
        try:
            state.update(analyze_code_step(state))
        except Exception as e:
            log("ERROR", "Analysis failed: %s", e)
            state["logs"] = f"AI analysis failed: {e}"
            break

        # Generate and apply patch.
        state["iteration_count"] = iteration - 1
        fix_result = fix_code_step(state)
        state.update(fix_result)

        if state.get("lint_failed"):
            log("WARNING", "Patch generation/application failed; retrying.")
            continue

        # Syntax validation.
        lint_result = lint_check_step(state)
        state.update(lint_result)

        if state.get("lint_failed"):
            # The syntax error becomes the next AI input.
            continue

        # Real tests.
        test_result = test_code_step(state)
        state.update(test_result)

        if state.get("success"):
            log("INFO", "-> Tests passed. Repair successful.")
            break

        # Failed test logs become the next iteration's context.
        log("WARNING", "-> Tests failed. Returning to analysis...")

    # Generate RCA regardless of success/failure.
    rca_result = generate_rca_step(state)
    state.update(rca_result)

    # Create PR only after successful tests.
    if state.get("success"):
        pr_result = create_pr_step(state)
        state.update(pr_result)
    else:
        log(
            "WARNING",
            "Repair did not succeed within %d iteration(s). No PR will be created.",
            MAX_ITERATIONS,
        )

    return state


if __name__ == "__main__":
    final_state = run_agent()

    log("INFO", "================ FINAL RESULT ================")
    log("INFO", "Success: %s", final_state.get("success"))
    log("INFO", "Iterations: %s", final_state.get("iteration_count"))
    log("INFO", "RCA: %s", final_state.get("rca_path"))
    log("INFO", "Report: %s", final_state.get("rca_html_path"))
    log("INFO", "PR: %s", final_state.get("pr_url"))

    # Return a non-zero code when the repair was not successful.
    raise SystemExit(0 if final_state.get("success") else 1)
