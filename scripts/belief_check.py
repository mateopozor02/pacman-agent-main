"""
Measure how accurately a team's EnemyBeliefs tracker locates hidden enemies.

Plays a few games against an opponent (baseline by default) and, every time
the tracker updates, compares its most likely cell for each hidden enemy with the enemy's true cell.
The true state is read by hooking the engine from this script only; the agent
itself never sees it.

Usage (from the repo root, venv active):
    python scripts/belief_check.py                 # agents/mateo/my_team.py
    python scripts/belief_check.py mateo
    python scripts/belief_check.py path/to/team.py --layouts defaultCapture RANDOM3
    python scripts/belief_check.py mateo --opponent caspar
"""

import argparse
import contextlib
import importlib.machinery
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEST_DIR = os.path.join(ROOT, 'pacman-contest', 'src', 'contest')

DEFAULT_LAYOUTS = ['defaultCapture', 'RANDOM3', 'jumboCapture', 'officeCapture', 'RANDOM8']


def resolve_team(team):
    if team == 'baseline':
        return 'baseline_team'
    if team.endswith('.py'):
        return os.path.abspath(team)
    path = os.path.join(ROOT, 'agents', team, 'my_team.py')
    if not os.path.exists(path):
        sys.exit(f'Unknown team "{team}": {path} does not exist')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('team', nargs='?', default='mateo', help='team name or path to my_team.py')
    parser.add_argument('--opponent', default='baseline', help='team to play against (name or path)')
    parser.add_argument('--layouts', nargs='*', default=DEFAULT_LAYOUTS, help='layouts to play')
    args = parser.parse_args()
    team_path = resolve_team(args.team)
    opponent_path = resolve_team(args.opponent)

    import contest.capture as capture
    from contest.util import manhattan_distance

    true_state = {}
    make_observation = capture.GameState.make_observation

    def spy(self, index):
        true_state['state'] = self
        return make_observation(self, index)
    capture.GameState.make_observation = spy

    errors, times = [], []
    original = {}

    def update(self, agent, game_state):
        start = time.time()
        original['update'](self, agent, game_state)
        times.append(time.time() - start)
        for i in agent.get_opponents(game_state):
            if game_state.get_agent_position(i) is None:
                real = true_state['state'].get_agent_position(i)
                errors.append(manhattan_distance(real, self.most_likely(i)))

    # The engine loads the team file afresh for every game, so patch EnemyBeliefs each time it is executed
    exec_module = importlib.machinery.SourceFileLoader.exec_module

    def patched_exec_module(self, module):
        exec_module(self, module)
        if getattr(module, '__file__', None) == team_path:
            original['update'] = module.EnemyBeliefs.update
            module.EnemyBeliefs.update = update
    importlib.machinery.SourceFileLoader.exec_module = patched_exec_module

    os.chdir(CONTEST_DIR)  # layouts are looked up relative to the working directory
    for layout in args.layouts:
        argv = ['-r', team_path, '-b', opponent_path, '-l', layout, '-Q', '-c']
        with contextlib.redirect_stdout(io.StringIO()):
            capture.run_games(**capture.read_command(argv))

    if 'update' not in original:
        sys.exit(f'{team_path} has no EnemyBeliefs class to measure')
    if not errors:
        sys.exit('No hidden-enemy samples collected')

    errors.sort()
    n = len(errors)
    print(f'hidden-enemy samples: {n}')
    print(f'error (cells): mean {sum(errors) / n:.2f}, median {errors[n // 2]}, 90th pct {errors[int(n * .9)]}')
    print(f'within 3 cells: {100 * sum(e <= 3 for e in errors) / n:.0f}%')
    print(f'update time: mean {1000 * sum(times) / len(times):.2f} ms, max {1000 * max(times):.1f} ms')


if __name__ == '__main__':
    main()
