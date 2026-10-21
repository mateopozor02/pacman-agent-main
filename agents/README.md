# Candidate teams

Each member develops a full team in their own folder. Only touch your own folder;
root `my_team.py` is the submission and is only changed when we merge the winner.

| Folder    | Owner  | Approach |
|-----------|--------|----------|
| `mateo/`  | Mateo  | _tbd_    |
| `nikita/` | Nikita | _tbd_    |
| `caspar/` | Caspar | _tbd_    |

All three start as a copy of the baseline reflex agents.

## Rules for every candidate
- Single self-contained `my_team.py`, importing only from `contest.*` and the standard library
  (the submission is one file).
- Under 1 s per move, under 15 s in `register_initial_state`. No threads, no printing.

## Evaluate
From the repo root with the venv active:

```bash
python scripts/eval.py mateo baseline                  # head-to-head, 10 fixed + 5 random layouts
python scripts/eval.py mateo nikita caspar baseline    # round robin
python scripts/eval.py mateo baseline --layouts defaultCapture -r 0   # quick check
```

Watch a single game with graphics:

```bash
cd pacman-contest/src/contest
python capture.py -r ../../../agents/mateo/my_team.py -b baseline_team -l RANDOM7
```

## Git
Work on `agent/<name>` branches and merge into `main` often; since everyone edits only their own
folder, merges should not conflict.
