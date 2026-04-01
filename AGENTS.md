# Sakha Development Guide

OpenEnv-compatible hospital ward assistant environment for Meta x PyTorch x HuggingFace x Scaler hackathon.

## Build, Lint, Test Commands

### Setup
```bash
uv sync --dev
```

### Running Tests
```bash
uv run sakha test
uv run python -m pytest tests/test_easy_grader.py -v
uv run python -m pytest tests/test_easy_grader.py::test_easy_grader_returns_0_to_1 -v
```

### Linting & Formatting
```bash
uv run sakha fix
uv run sakha check
uv run sakha ci
uv run sakha submit-check
```

### Server & Inference
```bash
uv run python server/app.py
docker build -t sakha-env . && docker run --rm -p 7860:7860 sakha-env
uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline
uv run openenv validate
```

### Adding Dependencies
**Never edit `pyproject.toml` directly.** Always use uv to ensure both `pyproject.toml` and `uv.lock` stay in sync:

```bash
uv add package_name           # runtime dependency
uv add --dev package_name    # dev dependency
```

---

## Code Style

### General
- Python 3.12 only
- Line length: 100 chars
- Indentation: 4 spaces
- Quotes: Double quotes (`"`)
- No trailing commas

### Imports (order: stdlib → third-party → local)
```python
import logging
from openenv.core.env_server.interfaces import Environment
from sakha.models import SakhaAction, SakhaObservation
```

### Types (Python 3.12 syntax)
```python
def process(actions: list[tuple[str, int | None]]) -> dict[str, float]:
class PatientState(BaseModel):
    medications_due: list[str] = Field(default_factory=list)
```

### Naming
| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `SakhaEnvironment` |
| Functions | snake_case | `reset()` |
| Constants | UPPER_SNAKE | `EIGHT_HOURS = 480` |
| Private | _snake_case | `_validate()` |
| Enums | PascalCase | `ActionType.CHECK_VITALS = "check_vitals"` |

### Error Handling
```python
# Bad
except Exception:
    pass

# Good
except ValueError as e:
    logger.warning(f"Invalid: {e}")
    return None
```

### Pydantic Models
```python
class SakhaAction(Action):
    """Action space for the Sakha environment."""
    action_type: ActionType
    patient_id: int | None = None

    @field_validator("action_type", mode="before")
    @classmethod
    def _validate(cls, v):
        if isinstance(v, str):
            return ActionType(v)
        return v
```

### Docstrings
```python
def reset(self, seed: int | None = None) -> SakhaObservation:
    """Reset the environment to initial state.

    Args:
        seed: Optional random seed.

    Returns:
        Initial observation.
    """
```

---

## Project Structure
```
sakha/
├── src/sakha/
│   ├── __init__.py    # Version
│   ├── env.py         # Environment
│   ├── graders.py     # Task graders
│   └── models.py      # Pydantic models
├── server/app.py      # FastAPI
├── tests/             # Test suite
├── inference.py       # Baseline
├── openenv.yaml       # OpenEnv spec
└── Dockerfile         # Container (port 7860)
```

## Important
- Env vars: `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL`
- OpenEnv: implement `reset()`, `step()`, `state()`
- Graders: return 0.0-1.0 deterministically

---

## Git Remotes
This repo has two remotes:
- `origin` → GitHub (https://github.com/atharva-again/sakha)
- `hf` → Hugging Face Spaces (https://huggingface.co/spaces/atharva-again/sakha)

```bash
git push origin main    # GitHub
git push hf main       # Hugging Face
```
