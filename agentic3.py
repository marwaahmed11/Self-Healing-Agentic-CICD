from __future__ import annotations
import os
import json
import time
from pathlib import Path
import difflib
import io
import zipfile
import requests as req
from typing import List, Dict, Any, TypedDict, Optional
from github import Github, InputGitTreeElement
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sandbox import run_tests


load_dotenv()

def log(level: str, msg: str, *args) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}")
    except Exception:
        print(f"{prefix} {level}: {msg}")

# ------------------
# Centralized configuration (environment-driven)
# Read all environment flags here so it's easy to find and modify defaults
# ------------------
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
AZURE_OPENAI_ENDPOINT = (
    os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    or "https://monahussein-5428-resource.services.ai.azure.com/openai/v1"
).rstrip("/") + "/"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or "gpt-4.1-mini"
AZURE_OPENAI_TEMP = float(os.getenv("AZURE_OPENAI_TEMP", "0.2"))

if AZURE_OPENAI_API_KEY:
    client = OpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )
else:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )
    client = OpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=token_provider,
    )

ALLOWED_ACTIONS = set(a.strip() for a in os.getenv("ALLOWED_ACTIONS", "create_pr").split(",") if a.strip())

# GitHub / workflow identifiers
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")
WORKFLOW_RUN_ID = os.getenv("workflow_run_id", "")
GITHUB_BASE_BRANCH = os.getenv("GITHUB_BASE_BRANCH", "main")

# Human-in-the-loop / approval
HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() == "true"
AUTO_MERGE = os.getenv("AUTO_MERGE", "false").lower() == "true"
TEST_COMMAND = os.getenv("TEST_COMMAND", "python -m pytest tests/ -v")

AGENTIC_TMP_DIR = Path('agentic_tmp')
AGENTIC_TMP_DIR.mkdir(exist_ok=True)

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


def invoke_structured(prompt: str, schema: type[BaseModel]) -> BaseModel:
    last_error = None
    for _ in range(3):
        try:
            response = client.responses.parse(
                model=AZURE_OPENAI_DEPLOYMENT,
                input=prompt,
                temperature=AZURE_OPENAI_TEMP,
                text_format=schema,
            )
            if response.output_parsed is None:
                raise RuntimeError("Azure OpenAI returned no structured output")
            return response.output_parsed
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Azure OpenAI structured request failed after 3 attempts "
        f"for deployment '{AZURE_OPENAI_DEPLOYMENT}'. "
        "AZURE_OPENAI_DEPLOYMENT must exactly match the deployed model name."
    ) from last_error

def save_audit(iteration: int, name: str, prompt: str, response_text: str) -> Path:
    path = AGENTIC_TMP_DIR / f"iter_{iteration}_{name}.json"
    data = {
        'iteration': iteration,
        'name': name,
        'prompt': prompt,
        'response': response_text,
        'timestamp': time.time()
    }
    path.write_text(json.dumps(data, indent=2))
    return path

def apply_edits(original_code: str, edits: List[dict]) -> str:
    lines = original_code.split('\n')
    # Sort edits in reverse order so line numbers don't shift for earlier edits
    edits = sorted(edits, key=lambda x: x.get('start_line', 0), reverse=True)
    for edit in edits:
        start = edit['start_line'] - 1
        end = edit['end_line']
        replacement = edit['replacement'].split('\n')
        lines[start:end] = replacement
    return '\n'.join(lines)

def find_callers(target_file: str) -> dict:
    target_module = Path(target_file).stem
    callers = {}
    for py_file in Path('.').rglob('*.py'):
        if any(part.startswith('.') or part in ('__pycache__', '.venv') for part in py_file.parts):
            continue
        if str(py_file) == target_file:
            continue
        try:
            source = py_file.read_text()
            if f"from {target_module} import" in source or f"import {target_module}" in source:
                callers[str(py_file)] = source
        except Exception:
            pass
    return callers

# ------------------
# GitHub Helpers
# ------------------
def github_client():
    token = GITHUB_TOKEN
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub operations")
    from github import Auth
    return Github(auth=Auth.Token(token))

def create_branch_and_commit_multiple(repo_full_name: str, branch_name: str, patches_dict: dict, commit_message: str):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = GITHUB_BASE_BRANCH
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    base_commit_sha = base_ref.object.sha
    base_commit = repo.get_git_commit(base_commit_sha)
    tree_items = []
    for file_path, file_content in patches_dict.items():
        blob = repo.create_git_blob(file_content, "utf-8")
        tree_items.append(InputGitTreeElement(path=file_path, mode='100644', type='blob', sha=blob.sha))
    new_tree = repo.create_git_tree(tree_items, base_tree=repo.get_git_tree(base_commit.tree.sha))
    new_commit = repo.create_git_commit(commit_message, new_tree, [base_commit])
    try:
        repo.get_git_ref(f"heads/{branch_name}").delete()
    except Exception:
        pass
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=new_commit.sha)
    log('INFO', "Created branch %s -> commit %s.", branch_name, new_commit.sha)

def open_pr_with_rca(repo_full_name: str, branch_name: str, pr_title: str, pr_body: str, draft: bool = False):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = GITHUB_BASE_BRANCH
    pr = repo.create_pull(title=pr_title, body=pr_body, head=branch_name, base=base_branch, draft=draft)
    return pr.html_url, pr.number, pr.node_id


def enable_auto_merge(pull_request_node_id: str, merge_method: str = "MERGE") -> bool:
    if not GITHUB_TOKEN:
        log('WARNING', "No GITHUB_TOKEN; cannot enable auto-merge")
        return False

    try:
        response = req.post(
            "https://api.github.com/graphql",
            json={
                "query": """
                mutation($input: EnablePullRequestAutoMergeInput!) {
                  enablePullRequestAutoMerge(input: $input) {
                    pullRequest { number merged }
                  }
                }
                """,
                "variables": {
                    "input": {
                        "pullRequestId": pull_request_node_id,
                        "mergeMethod": merge_method,
                    }
                },
            },
            headers={
                "Authorization": f"bearer {GITHUB_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if response.status_code != 200:
            log('WARNING', "Auto-merge GraphQL failed: %s %s", response.status_code, response.text)
            return False
        errors = response.json().get("errors")
        if errors:
            log('WARNING', "Enable auto-merge errors: %s", errors)
            return False
        return True
    except Exception as error:
        log('ERROR', "Exception enabling auto-merge: %s", error)
        return False

# ==========================================
# THE STATE 
# ==========================================
class AgenticState(TypedDict):
    logs: str                  
    iteration_count: int       
    success: bool                
    
    # Memory Bank
    repair_memory: Dict[str, Any]
    
    current_file: Optional[str]
    repair_strategy: Optional[str]
    
    rca_html_path: Optional[str]
    pr_url: Optional[str]
    pr_number: Optional[int]
    pr_node_id: Optional[str]
    pr_created: Optional[bool]
    auto_merge_requested: Optional[bool]
    approved: Optional[bool]
    lint_failed: Optional[bool]


# ==========================================
# THE STEPS
# ==========================================
def fetch_logs_step(state: AgenticState) -> dict:    
    log('INFO', "STEP[fetch_logs_step]: Fetching logs...")
    
    # Initialize repair memory if it doesn't exist
    repair_memory = state.get("repair_memory")
    if not repair_memory:
        repair_memory = {
            "iterations": [],
            "repo_state": {},
            "context": {
                "original_logs": "",
                "latest_logs": "",
                "files_attempted": []
            }
        }

    if state.get("iteration_count", 0) > 0:
        log('INFO', "-> Fetching latest test logs from the secure Sandbox.")
        repair_memory["context"]["latest_logs"] = state["logs"]
        return {"logs": state["logs"], "repair_memory": repair_memory}

    # Fetch real logs from GitHub Actions
    workflow_run_id = WORKFLOW_RUN_ID
    repo_full_name = GITHUB_REPO

    if repo_full_name:
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
                return {"logs": "No failed workflow runs found.", "repair_memory": repair_memory}

            logs_parts = []
            for job in run.jobs():
                if job.conclusion == "failure":
                    for step in job.steps:
                        if step.conclusion == "failure":
                            logs_parts.append(f"=== Job: {job.name} | Step: {step.name} ===\nStatus: {step.conclusion}\n")
            
            
            token = GITHUB_TOKEN
            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
            logs_url = f"https://api.github.com/repos/{repo_full_name}/actions/runs/{run.id}/logs"
            resp = req.get(logs_url, headers=headers, allow_redirects=True)
            if resp.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in z.namelist():
                    content = z.read(name).decode("utf-8", errors="replace")
                    if any(kw in name.lower() for kw in ["run tests", "test", "build", "run"]):
                        logs_parts.append(f"=== {name} ===\n{content[-3000:]}\n")
                    elif not logs_parts:
                        logs_parts.append(f"=== {name} ===\n{content[-2000:]}\n")
            
            if logs_parts:
                real_logs = "\n".join(logs_parts)
                if len(real_logs) > 8000:
                    real_logs = real_logs[-8000:]
                repair_memory["context"]["original_logs"] = real_logs
                repair_memory["context"]["latest_logs"] = real_logs
                return {"logs": real_logs, "repair_memory": repair_memory}
            else:
                return {"logs": "Workflow run found but no failure logs could be extracted.", "repair_memory": repair_memory}
        except Exception as e:
            return {"logs": f"Failed to fetch logs from GitHub: {str(e)}", "repair_memory": repair_memory}

    return {"logs": "No GITHUB_REPO configured.", "repair_memory": repair_memory}

def analyze_code_step(state: AgenticState) -> dict:
    log('INFO', "STEP[analyze_code_step]: Planning repair strategy...")
    
    repair_memory = state.get("repair_memory", {})
    iterations_history = json.dumps(repair_memory.get("iterations", []), indent=2)

    repo_files = []
    try:
        for p in Path('.').rglob('*.py'):
            if not any(part.startswith('.') or part in ('__pycache__', '.venv') for part in p.parts):
                repo_files.append(str(p))
    except Exception:
        pass
    file_listing = ', '.join(repo_files) if repo_files else 'unknown'

    prompt = f"""
    You are a Senior Python Developer diagnosing a CI/CD failure.
    
    ERROR LOGS:
    {state['logs']}

    AVAILABLE FILES IN REPO:
    {file_listing}

    PAST REPAIR ATTEMPTS (Do not repeat failed strategies):
    {iterations_history}

    Analyze the logs and determine the root cause.
    """
    
    response = invoke_structured(prompt, AnalyzeOutput)
    
    save_audit(state.get('iteration_count', 0), 'plan_response', prompt, str(response.model_dump()))

    try:
        target_file = response.target_file
        repair_strategy = response.repair_strategy
        
        # Track attempted files
        if target_file and target_file not in repair_memory["context"]["files_attempted"]:
            repair_memory["context"]["files_attempted"].append(target_file)
            
        log('INFO', "-> Plan: Fix %s | Strategy: %s", target_file, repair_strategy)
        return {
            "current_file": target_file,
            "repair_strategy": repair_strategy,
            "repair_memory": repair_memory
        }
    except Exception as e:
        log('ERROR', "Failed to parse analysis JSON: %s", str(e))
        return {"current_file": "", "repair_strategy": "Failed to parse plan."}

def fix_code_step(state: AgenticState) -> dict:
    current_file = state.get("current_file")
    iteration = state.get("iteration_count", 0)
    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.get("repo_state", {})
    
    log('INFO', "STEP[fix_code_step]: Generating surgical patch for %s", current_file)

    if not current_file:
        return {"lint_failed": False}

    # 1. Read from repo_state memory OR disk
    if current_file in repo_state:
        broken_code = repo_state[current_file]
        log('INFO', "-> Reading previously patched version of %s from memory", current_file)
    else:
        try:
            broken_code = Path(current_file).read_text()
        except Exception:
            broken_code = "# File missing or empty"

    # 2. Context-aware reading (callers)
    callers = find_callers(current_file)
    callers_context = ""
    if callers:
        callers_context = "CALLERS OF THIS FILE (for context):\n"
        for f, code in callers.items():
            callers_context += f"--- {f} ---\n{code[-1000:]}\n\n"

    # 3. Prompt for surgical diff
    prompt = f"""
    You are a Senior Python Developer implementing a fix.

    TARGET FILE: {current_file}
    REPAIR STRATEGY: {state.get('repair_strategy')}
    ERROR LOGS: {state.get('logs')}

    {callers_context}

    CURRENT CODE ({current_file}):
    {broken_code}

    Do NOT rewrite the whole file unless necessary.
    """

    response = invoke_structured(prompt, FixOutput)

    save_audit(iteration, 'fix_response', prompt, str(response.model_dump()))

    try:
        edits = [e.model_dump() for e in response.edits]
        
        # Apply the edits programmatically
        patched_code = apply_edits(broken_code, edits)
        repo_state[current_file] = patched_code
        
        # Log to repair memory
        repair_memory["iterations"].append({
            "iteration": iteration + 1,
            "target_file": current_file,
            "strategy": state.get("repair_strategy"),
            "edits_applied": edits,
            "result": "pending",
            "reason": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        
        log('INFO', "-> Applied %d edits to %s", len(edits), current_file)
        
    except Exception as e:
        log('ERROR', "Failed to apply surgical edits: %s", str(e))

    repair_memory["repo_state"] = repo_state
    
    return {
        "repair_memory": repair_memory,
        "iteration_count": iteration + 1,
        "lint_failed": False # Reset lint status
    }

def lint_check_step(state: AgenticState) -> dict:
    log('INFO', "STEP[lint_check_step]: Running syntax checks on patched files...")
    import subprocess
    repo_state = state.get("repair_memory", {}).get("repo_state", {})
    
    for filepath, code in repo_state.items():
        tmp = Path(f"/tmp/lint_agentic.py")
        tmp.write_text(code)
        result = subprocess.run(["python3", "-m", "py_compile", str(tmp)], capture_output=True, text=True)
        if result.returncode != 0:
            log('WARNING', "Syntax error found in %s:\n%s", filepath, result.stderr)
            # Record failure in memory
            iterations = state.get("repair_memory", {}).get("iterations", [])
            if iterations:
                iterations[-1]["result"] = "failed"
                iterations[-1]["reason"] = f"Syntax error: {result.stderr}"
            return {"lint_failed": True, "logs": f"Syntax error in {filepath}:\n{result.stderr}"}
            
    log('INFO', "-> All patched files passed syntax checks.")
    return {"lint_failed": False}


def test_code_step(state: AgenticState) -> dict:
    log('INFO', "STEP[test_code_step]: Injecting ALL accumulated patches into Sandbox...")
    
    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.get("repo_state", {})
    
    sandbox_result = run_tests(
        project_path=".",
        test_command=TEST_COMMAND,
        patches_dict=repo_state,
    )
    success = sandbox_result.get("success", False)
    logs = sandbox_result.get("logs", "")
    
    if success:
        log('INFO', "-> Sandbox Execution: SUCCESS")
    else:
        log('WARNING', "-> Sandbox Execution: FAILED. Gathering new logs...")

    # Update iteration history
    iterations = repair_memory.get("iterations", [])
    if iterations:
        iterations[-1]["result"] = "passed" if success else "failed"
        iterations[-1]["reason"] = "Tests failed" if not success else "Tests passed"

    return {
        "success": success,
        "logs": logs,
        "repair_memory": repair_memory
    }


def route_after_lint(state: AgenticState) -> str:
    if state.get("lint_failed"):
        return "analyze_code"
    return "test_code"


def route_after_test(state: AgenticState) -> str:
    if state.get("success") or state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "generate_rca"
    return "analyze_code"


def generate_rca_step(state: AgenticState) -> dict:
    log('INFO', "STEP[generate_rca_step]: Generating RCA and HTML Report...")
    repair_memory = state.get("repair_memory", {})
    iterations = repair_memory.get("iterations", [])
    # 1. JSON RCA for programmatic consumption
    rca_obj = {
        "summary": "Automated pipeline repair",
        "iterations_taken": len(iterations),
        "files_modified": list(repair_memory.get("repo_state", {}).keys()),
        "final_status": "Success" if state.get("success") else "Max Iterations Reached",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rca_path = AGENTIC_TMP_DIR / "final_rca.json"
    rca_path.write_text(json.dumps(rca_obj, indent=2))

    # 2. Create patch.diff (unified diff) and patched_files.zip
    repo_state = repair_memory.get("repo_state", {}) or {}
    diff_parts = []
    patched_zip_path = AGENTIC_TMP_DIR / "patched_files.zip"
    with zipfile.ZipFile(patched_zip_path, 'w') as zf:
        for file_path, patched_code in repo_state.items():
            try:
                orig_text = Path(file_path).read_text()
            except Exception:
                orig_text = ""
            # Generate unified diff
            diff = difflib.unified_diff(
                orig_text.splitlines(keepends=True),
                patched_code.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=""
            )
            diff_text = "".join(list(diff))
            if diff_text:
                diff_parts.append(diff_text)
            # Add patched file to zip
            zf.writestr(file_path, patched_code)

        patch_path = AGENTIC_TMP_DIR / "patch.diff"
        patch_path.write_text("\n\n".join(diff_parts) or "")

        # 3. Simple black & white HTML report (no AI generation)
        files_modified_html = "".join([f"<li>{f}</li>" for f in rca_obj["files_modified"]]) if rca_obj["files_modified"] else "<li>None</li>"
        # Build iteration timeline HTML separately to avoid backslashes inside f-string expressions
        timeline_html = ''.join([
            f"<div style=\"margin-bottom:10px;\"><strong>Iteration {it.get('iteration')} — {it.get('target_file')}</strong>"
            f"<div class=\"muted\">Strategy: {it.get('strategy')}</div>"
            f"<pre>{json.dumps(it.get('edits_applied', []), indent=2)}</pre>"
            f"<div class=\"muted\">Result: {it.get('result')} {it.get('reason') or ''}</div></div>"
            for it in iterations
        ]) or '<p>No iterations recorded.</p>'

        html_content = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width,initial-scale=1" />
            <title>Pipeline Doctor Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color: #000; background: #fff; margin: 24px; }}
                h1 {{ margin-bottom: 4px; }}
                .meta {{ color: #333; font-size: 90%; margin-bottom: 12px; }}
                .card {{ border: 1px solid #ddd; padding: 14px; margin-bottom: 12px; background: #fff; }}
                pre {{ background: #f8f8f8; padding: 10px; overflow: auto; border: 1px solid #e6e6e6; }}
                .muted {{ color: #666; font-size: 90%; }}
                a {{ color: #000; text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Pipeline Doctor Report</h1>
            <div class="meta">Generated: {rca_obj['timestamp']}</div>

            <div class="card">
                <h2>Summary</h2>
                <p><strong>Status:</strong> {rca_obj['final_status']}</p>
                <p><strong>Iterations:</strong> {rca_obj['iterations_taken']}</p>
                <p><strong>Files Modified:</strong></p>
                <ul>
                    {files_modified_html}
                </ul>
            </div>

            <div class="card">
                <h2>Iteration Timeline</h2>
                {timeline_html}
            </div>

            <div class="card">
                <h2>Artifacts</h2>
                <ul>
                    <li><a href="final_rca.json">final_rca.json</a></li>
                    <li><a href="report.html">report.html</a> (this file)</li>
                    <li><a href="patch.diff">patch.diff</a></li>
                    <li><a href="patched_files.zip">patched_files.zip</a></li>
                </ul>
            </div>

            <div class="card">
                <h2>Reproduction</h2>
                <pre>python -m pytest tests/ -q</pre>
            </div>

        </body>
        </html>
        """

        html_path = AGENTIC_TMP_DIR / "report.html"
        html_path.write_text(html_content)

        log('INFO', "-> RCA, patch.diff and patched_files.zip generated.")
        return {"rca_path": str(rca_path), "rca_html_path": str(html_path), "patch_path": str(patch_path), "patched_zip": str(patched_zip_path)}

def create_pr_step(state: AgenticState) -> dict:
    log('INFO', "STEP[create_pr_step]: Creating PR with ALL patches...")
    if 'create_pr' not in ALLOWED_ACTIONS:
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
        create_branch_and_commit_multiple(repo_full_name, branch_name, repo_state, commit_msg)
        
        iterations = state.get("repair_memory", {}).get("iterations", [])
        last_iter = iterations[-1] if iterations else {}
        target_file = last_iter.get("target_file", "multiple files")
        strategy = last_iter.get("strategy", "Automated code repair")
        pr_title = f"Pipeline Doctor — Fix {target_file}"
        pr_body = (
            f"Summary: Automated fix for failing tests in {target_file}.\n\n"
            f"Root cause: {strategy}\n\n"
            f"Change: Applied edits to {len(repo_state)} file(s): {', '.join(repo_state)}.\n\n"
            f"Verification: Docker sandbox tests passed using `{TEST_COMMAND}`.\n\n"
            f"Workflow run: {WORKFLOW_RUN_ID or 'not provided'}\n"
            "Artifacts: final_rca.json, report.html, patch.diff, and patched_files.zip"
        )

        pr_url, pr_number, pr_node_id = open_pr_with_rca(
            repo_full_name, branch_name, pr_title, pr_body, draft=False
        )
        log('INFO', "-> PR created: %s", pr_url)

        auto_merge_requested = False
        if not HITL_ENABLED and AUTO_MERGE:
            auto_merge_requested = enable_auto_merge(pr_node_id)
            log(
                'INFO' if auto_merge_requested else 'WARNING',
                "-> Auto-merge %s for PR #%s",
                "requested" if auto_merge_requested else "could not be enabled",
                pr_number,
            )

        return {
            "pr_url": pr_url,
            "pr_number": pr_number,
            "pr_node_id": pr_node_id,
            "pr_created": True,
            "auto_merge_requested": auto_merge_requested,
        }
    except Exception as e:
        log('ERROR', "-> Failed to create PR: %s", str(e))
        return {"pr_created": False, "reason": str(e)}

# ==========================================
# MAIN FLOW
# ==========================================
if __name__ == "__main__":
    log('INFO', "Starting Pipeline Doctor Agent (Azure OpenAI + Python)...")

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

    state.update(fetch_logs_step(state))

    for iteration in range(1, MAX_ITERATIONS + 1):
        log('INFO', "================ ITERATION %d/%d ================", iteration, MAX_ITERATIONS)

        try:
            state.update(analyze_code_step(state))
        except Exception as error:
            log('ERROR', "Analysis failed: %s", error)
            state["logs"] = f"AI analysis failed: {error}"
            break

        state["iteration_count"] = iteration - 1
        state.update(fix_code_step(state))

        if state.get("lint_failed"):
            continue

        state.update(lint_check_step(state))
        if state.get("lint_failed"):
            continue

        state.update(test_code_step(state))
        if state.get("success"):
            log('INFO', "-> Tests passed. Repair successful.")
            break

    state.update(generate_rca_step(state))

    if state.get("success"):
        state.update(create_pr_step(state))
    else:
        log('WARNING', "Repair did not succeed within %d iteration(s). No PR will be created.", MAX_ITERATIONS)