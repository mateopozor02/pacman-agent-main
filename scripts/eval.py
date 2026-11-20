"""
Evaluate Pacman capture-the-flag teams against each other.

Each pairing is played on every layout with both colour assignments, so
neither team gets a side advantage. Games run in parallel worker processes,
one fresh process per game so agent modules never leak state.

Usage (from the repo root, venv active):
    python scripts/eval.py mateo baseline              # head-to-head
    python scripts/eval.py mateo nikita caspar baseline # round robin
    python scripts/eval.py mateo baseline -r 10 -j 4    # 10 random mazes, 4 workers
    python scripts/eval.py agents/mateo/my_team.py main --layouts defaultCapture

Opponent pool: play each listed team against a fixed set of opponents only
(no games between the opponents) and compare score margins side by side:
    python scripts/eval.py mateo --pool                 # vs the default pool
    python scripts/eval.py mateo mateo@HEAD~1 --pool    # new version vs previous commit
    python scripts/eval.py mateo --pool rush turtle     # vs a custom pool

Team shortcuts:
    baseline      -> pacman-contest/src/contest/baseline_team.py
    <opponent>    -> scripts/opponents/<opponent>.py:
                     rush (two baseline attackers), turtle (two baseline defenders),
                     mateo_raid (Mateo's agent as merged in PR #2)
    main          -> my_team.py (the submission)
    <name>        -> agents/<name>/my_team.py
    <name>/<file> -> agents/<name>/<file>.py
    <team>@<rev>  -> that team's file as of git revision <rev> (e.g. mateo@HEAD~1)
    path/to/x.py  -> used as-is
"""

import argparse
import atexit
import contextlib
import io
import itertools
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEST_DIR = os.path.join(ROOT, 'pacman-contest', 'src', 'contest')

DEFAULT_LAYOUTS = ['defaultCapture', 'alleyCapture', 'bloxCapture', 'crowdedCapture',
                   'distantCapture', 'fastCapture', 'jumboCapture', 'mediumCapture',
                   'officeCapture', 'strategicCapture']


OPPONENTS_DIR = os.path.join(ROOT, 'scripts', 'opponents')
DEFAULT_POOL = ['baseline', 'rush', 'turtle', 'mateo_raid', 'caspar/hybrid', 'caspar/strategic_v3']
_snapshot_dir = None  # temporary folder for files checked out by <team>@<rev>, created on first use


def team_path(team):
    """Path of a team's file from its shortcut, without checking that it exists."""
    if team == 'baseline':
        return os.path.join(CONTEST_DIR, 'baseline_team.py')
    if os.path.exists(os.path.join(OPPONENTS_DIR, team + '.py')):
        return os.path.join(OPPONENTS_DIR, team + '.py')
    if team == 'main':
        return os.path.join(ROOT, 'my_team.py')
    if team.endswith('.py'):
        return os.path.abspath(team)
    if '/' in team:
        name, file = team.split('/', 1)
        return os.path.join(ROOT, 'agents', name, file + '.py')
    return os.path.join(ROOT, 'agents', team, 'my_team.py')


def snapshot(team, rev):
    """Writes the team's file as of git revision rev to a temporary file and returns its path."""
    rel = os.path.relpath(team_path(team), ROOT)
    try:
        source = subprocess.run(['git', 'show', f'{rev}:{rel}'], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f'Cannot read {rel} at {rev}: {e.stderr.strip()}')
    global _snapshot_dir
    if _snapshot_dir is None:
        _snapshot_dir = tempfile.mkdtemp(prefix='eval_snapshots_')
        atexit.register(shutil.rmtree, _snapshot_dir, ignore_errors=True)
    safe = ''.join(c if c.isalnum() else '_' for c in f'{team}_{rev}')
    path = os.path.join(_snapshot_dir, safe + '.py')
    with open(path, 'w') as f:
        f.write(source)
    return path


def resolve_team(team):
    if '@' in team:
        name, rev = team.rsplit('@', 1)
        return snapshot(name, rev)
    path = team_path(team)
    if not os.path.exists(path):
        sys.exit(f'Unknown team "{team}": {path} does not exist')
    return path


def play_game(job):
    """Runs one game in this worker process. Returns a result dict."""
    red, blue, red_path, blue_path, layout, length = job
    import contest.capture as capture

    os.chdir(CONTEST_DIR)  # layouts are looked up relative to the working directory
    argv = ['-r', red_path, '-b', blue_path, '-l', layout, '-Q', '-c', '-i', str(length)]
    sink = io.StringIO()
    start = time.time()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            options = capture.read_command(argv)
            game = capture.run_games(**options)[0]
    except Exception as e:  # team failed to load, etc.
        return {'red': red, 'blue': blue, 'layout': layout, 'error': repr(e), 'log': sink.getvalue()}

    warnings = game.total_agent_time_warnings
    return {
        'red': red, 'blue': blue, 'layout': layout,
        'score': game.state.data.score,
        'crashed': game.agent_crashed,
        'timeout': game.agent_timeout,
        'red_warnings': warnings[0] + warnings[2],
        'blue_warnings': warnings[1] + warnings[3],
        'seconds': time.time() - start,
        'error': None,
    }


def build_jobs(pairs, paths, layouts, length):
    """Two games (one per colour) for every pairing on every layout."""
    jobs = []
    for a, b in pairs:
        for layout in layouts:
            jobs.append((a, b, paths[a], paths[b], layout, length))
            jobs.append((b, a, paths[b], paths[a], layout, length))
    return jobs


def summarize_pool(teams, pool, results):
    """One row per opponent: each team's wins-ties-losses and average score margin against it."""
    cell = {}  # (team, opponent) -> [W, T, L, total margin, games]
    for r in results:
        if r['error']:
            continue
        for team, opp, sign in ((r['red'], r['blue'], 1), (r['blue'], r['red'], -1)):
            if team not in teams or opp not in pool:
                continue
            c = cell.setdefault((team, opp), [0, 0, 0, 0, 0])
            margin = sign * r['score']
            c[0 if margin > 0 else 1 if margin == 0 else 2] += 1
            c[3] += margin
            c[4] += 1

    width = max(len(t) for t in teams)
    print('\n=== Versus pool (wins-ties-losses, average score margin) ===')
    print(f'  {"opponent":<20}' + ''.join(f'  {t:>{max(width, 16)}}' for t in teams))
    for opp in pool + ['ALL']:
        row = f'  {opp:<20}'
        for t in teams:
            cs = [cell.get((t, o)) for o in pool] if opp == 'ALL' else [cell.get((t, opp))]
            cs = [c for c in cs if c]
            if not cs:
                row += f'  {"-":>{max(width, 16)}}'
                continue
            w, ti, l, m, n = (sum(c[k] for c in cs) for k in range(5))
            row += f'  {f"{w}-{ti}-{l} {m / n:+6.2f}":>{max(width, 16)}}'
        print(row)


def summarize(teams, results):
    stats = {t: {'W': 0, 'T': 0, 'L': 0, 'score': 0, 'games': 0, 'warn': 0, 'fail': 0} for t in teams}
    pair = {}  # (a, b) -> [wins of a, ties, wins of b]

    for r in results:
        if r['error']:
            continue
        red, blue, s = r['red'], r['blue'], r['score']
        for team, sign, warn in ((red, 1, r['red_warnings']), (blue, -1, r['blue_warnings'])):
            st = stats[team]
            st['games'] += 1
            st['score'] += sign * s
            st['warn'] += warn
            if sign * s > 0:
                st['W'] += 1
            elif s == 0:
                st['T'] += 1
            else:
                st['L'] += 1
        key = tuple(sorted((red, blue)))
        p = pair.setdefault(key, [0, 0, 0])
        winner = red if s > 0 else blue if s < 0 else None
        p[1 if winner is None else (0 if winner == key[0] else 2)] += 1

    print('\n=== Head to head (wins-ties-losses) ===')
    for (a, b), (wa, t, wb) in sorted(pair.items()):
        print(f'  {a:>12} vs {b:<12} {wa}-{t}-{wb}')

    print('\n=== Standings ===')
    print(f'  {"team":<12} {"W":>4} {"T":>4} {"L":>4} {"win%":>6} {"avg score":>10} {"time warn":>10}')
    ranked = sorted(teams, key=lambda t: (stats[t]['W'] + 0.5 * stats[t]['T']) / max(1, stats[t]['games']),
                    reverse=True)
    for t in ranked:
        st = stats[t]
        n = max(1, st['games'])
        print(f'  {t:<12} {st["W"]:>4} {st["T"]:>4} {st["L"]:>4} {100 * st["W"] / n:>5.0f}% '
              f'{st["score"] / n:>10.2f} {st["warn"]:>10}')

    problems = [r for r in results if r['error'] or r['crashed'] or r['timeout']]
    if problems:
        print('\n=== Problems ===')
        for r in problems:
            what = r['error'] or ('crash' if r['crashed'] else 'timeout forfeit')
            print(f'  {r["red"]} (red) vs {r["blue"]} (blue) on {r["layout"]}: {what}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('teams', nargs='+', help='two or more teams (see shortcuts above)')
    parser.add_argument('--layouts', nargs='*', default=DEFAULT_LAYOUTS, help='fixed layouts to play')
    parser.add_argument('-r', '--random', type=int, default=5, help='number of random mazes to add')
    parser.add_argument('--seed', type=int, default=0, help='first random maze seed (seeds are seed+1..seed+r)')
    parser.add_argument('-i', '--length', type=int, default=1200, help='game length in moves')
    parser.add_argument('-j', '--jobs', type=int, default=max(1, mp.cpu_count() // 2),
                        help='parallel games (keep below CPU count so timing stays realistic)')
    parser.add_argument('--pool', nargs='*', default=None,
                        help='play the teams only against these opponents '
                             f'(no names: {" ".join(DEFAULT_POOL)})')
    args = parser.parse_args()

    opponents = None
    if args.pool is not None:
        opponents = args.pool or [o for o in DEFAULT_POOL if '@' in o or os.path.exists(team_path(o))]
        missing = [o for o in DEFAULT_POOL if not args.pool and o not in opponents]
        if missing:
            print(f'Skipping pool opponents not found locally: {", ".join(missing)}')
        if set(opponents) & set(args.teams):
            parser.error('a team cannot also be in the pool')
        pairs = [(t, o) for t in args.teams for o in opponents]
    else:
        if len(args.teams) < 2:
            parser.error('need at least two teams (or use --pool)')
        pairs = list(itertools.combinations(args.teams, 2))
    if len(set(args.teams)) != len(args.teams):
        parser.error('teams must be distinct')

    everyone = list(dict.fromkeys(args.teams + (opponents or [])))
    paths = {t: resolve_team(t) for t in everyone}
    layouts = list(args.layouts) + [f'RANDOM{args.seed + i + 1}' for i in range(args.random)]
    jobs = build_jobs(pairs, paths, layouts, args.length)
    print(f'Playing {len(jobs)} games on {len(layouts)} layouts with {args.jobs} workers...')

    results = []
    start = time.time()
    # maxtasksperchild=1: fresh interpreter state per game
    with mp.get_context('spawn').Pool(args.jobs, maxtasksperchild=1) as workers:
        for i, r in enumerate(workers.imap_unordered(play_game, jobs), 1):
            results.append(r)
            if r['error']:
                line = f'ERROR {r["error"]}'
            else:
                flags = ' CRASH' if r['crashed'] else ' TIMEOUT' if r['timeout'] else ''
                line = f'score {r["score"]:+d} ({r["seconds"]:.0f}s){flags}'
            print(f'[{i}/{len(jobs)}] {r["red"]} vs {r["blue"]} on {r["layout"]}: {line}', flush=True)

    summarize(everyone, results)
    if opponents:
        summarize_pool(args.teams, opponents, results)
    print(f'\nTotal time: {time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
