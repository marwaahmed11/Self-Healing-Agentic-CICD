# Self-Healing CI/CD Pipeline

This repository demonstrates an AI-assisted self-healing pipeline that detects failed CI jobs, analyzes the failure context, proposes code fixes with Gemini, validates the changes in an isolated sandbox, and opens a pull request for review.

## Project Overview

The pipeline is built around an agentic workflow that:
- Reads failure logs from CI or local sandbox runs
- Identifies the most likely root cause
- Generates focused code edits
- Re-runs checks to confirm the fix
- Creates a pull request when the repair is valid

The included FastAPI service and test suite provide a realistic target for the agent to repair, validate, and iterate against.

## Key Parts

- `agentic.py`: LangGraph-based self-healing orchestrator
- `main.py`: FastAPI application used as the demo service
- `sandbox.py`: Safe isolated test runner, backed by Docker
- `tests/test_todos.py`: Test suite used to trigger and verify repairs

## How It Works

1. A workflow failure is detected or test output is collected from the sandbox.
2. The agent inspects logs and source files to determine the root cause.
3. Gemini generates a surgical fix.
4. The sandbox re-runs tests to confirm the change.
5. If the fix passes, the agent can open a pull request for human review.

## Setup In A New Project

Use these steps if you want to recreate this project from scratch in a fresh folder or repository.

### 1. Create the project folder
```bash
mkdir self-healing-cicd
cd self-healing-cicd
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the project files
Copy or create the same core files in the new project:
- `main.py`
- `agentic.py`
- `sandbox.py`
- `tests/test_todos.py`
- `requirements.txt`

### 5. Configure environment variables
Set the values used by the agent before running it:
- `GEMINI_API_KEY`
- `GITHUB_TOKEN`
- `GITHUB_REPO`
- `GITHUB_BASE_BRANCH` if your default branch is not `main`
- `HITL_ENABLED` if you want manual approval in the loop
- `AUTO_MERGE` if you want the workflow to attempt automatic merge

Example:
```bash
export GEMINI_API_KEY="your-key"
export GITHUB_TOKEN="your-github-token"
export GITHUB_REPO="owner/repo"
```

### 6. Run the demo API
```bash
uvicorn main:app --reload --port 8000
```

The service will be available at `http://localhost:8000`.

### 7. Run the tests locally
```bash
pytest tests/test_todos.py -v
```

### 8. Run the self-healing agent
```bash
python agentic.py
```

## Recommended Runtime Requirements

- Python 3.10 or newer
- Docker installed and running for sandbox execution
- Access to a Gemini API key
- Access to a GitHub repository and token with write permissions if you want PR creation

## Notes

- The demo application uses in-memory state, so test runs are fast and isolated.
- The agent is designed for iterative repair workflows, not one-shot patching.
- Human review can remain enabled even after automated validation succeeds.