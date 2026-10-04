"""Standalone sanity check for the kiosk page's simulation model.

Run from the repo root (uv env already provisioned, see build_data.py):

    uv run --no-project --python <venv>/bin/python tools/kiosk/validate_model.py

Two things are checked:

1. The seed pools can be recovered from the research's own pickled output.
   `sim_output/sims_*.dat` hold `SimulationResult` namedtuples whose
   `first_spreader_idx` field records the seed of every run, so the 30 distinct
   values per file are exactly the pools the study seeded from.

2. A reimplementation of the model from the brief reproduces the published
   statistics (Betweenness 87.4% escape / 0.455 mean, Closeness 99.0% / 0.562).
   The exact figures will differ slightly - a fresh RNG stream is used - but the
   ordering and the size of the gap should match.
"""
import pickle
import statistics
import sys
import time
import types
from collections import namedtuple

import numpy as np

ROOT = '/Users/haidaralifawwaz/Documents/rumour-spread-model'

# --- model parameters ------------------------------------------------------
LAMBDA, GAMMA, ETA, DELTA = 0.20, 0.05, 0.10, 0.50
MAX_STEPS, ESCAPE = 300, 0.05
TRIALS = 200

N = 4039
PUBLISHED = {
    'sims_bc': ('Betweenness', 0.874, 0.455, 0.266),
    'sims_cc': ('Closeness',   0.990, 0.562, 0.159),
}


# ---------------------------------------------------------------------------
# 1. Recover the seed pools from the pickles.
#    graph-tool is not installed, but the only thing the pickles need from it
#    is PropertyArray (the rumor_size series), so stub it as an ndarray.
# ---------------------------------------------------------------------------
def load_pools():
    gt = types.ModuleType('graph_tool')

    class PropertyArray(np.ndarray):
        pass

    gt.PropertyArray = PropertyArray
    sys.modules['graph_tool'] = gt

    SimulationResult = namedtuple(
        'SimulationResult',
        ['first_spreader_idx', 'first_spreader_degree', 'time_step', 'rumor_size'],
    )
    globals()['SimulationResult'] = SimulationResult

    pools, published = {}, {}
    for name in ['sims_bc', 'sims_cc', 'sims_deg_bc']:
        with open(f'{ROOT}/sim_output/{name}.dat', 'rb') as fh:
            sims = pickle.load(fh)
        pools[name] = sorted({int(s.first_spreader_idx) for s in sims})
        finals = [float(np.asarray(s.rumor_size)[-1]) for s in sims]
        published[name] = (
            sum(1 for v in finals if v >= ESCAPE) / len(finals),
            statistics.mean(finals),
            statistics.stdev(finals),
        )
    return pools, published


# ---------------------------------------------------------------------------
# 2. Reimplement the stepper and check it behaves like the study.
# ---------------------------------------------------------------------------
def build_csr():
    edges = np.loadtxt(f'{ROOT}/datasets/facebook_combined.txt', dtype=np.int32)
    assert edges.shape == (88234, 2), edges.shape
    src, dst = edges[:, 0], edges[:, 1]
    deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(deg)
    both = np.concatenate([src, dst])
    neighbours = np.concatenate([dst, src])[np.argsort(both, kind='stable')]
    assert offsets[-1] == len(neighbours)
    return offsets, neighbours


def run(seed, offsets, neighbours, rng):
    state = np.zeros(N, dtype=np.int8)
    state[seed] = 1
    steps = 0
    while steps < MAX_STEPS:
        spreaders = np.flatnonzero(state == 1)
        if spreaders.size == 0:
            break
        rng.shuffle(spreaders)
        for s in spreaders:
            block = neighbours[offsets[s]:offsets[s + 1]]
            st = state[block]                       # states at the moment of the visit
            ignorant = block[st == 0]
            k_s = int((st == 1).sum())
            k_r = int((st == 2).sum())

            for v in ignorant:
                state[v] = 1 if rng.random() < LAMBDA else 2

            if (k_s > 0 and rng.random() < 1 - (1 - GAMMA) ** k_s) or \
               (k_r > 0 and rng.random() < 1 - (1 - ETA) ** k_r) or \
               (rng.random() < DELTA):
                state[s] = 2
        steps += 1
    return float((state != 0).sum()) / N


def main():
    pools, published = load_pools()
    offsets, neighbours = build_csr()
    rng = np.random.default_rng(2026)

    print("=== recovered from sim_output/*.dat (the study's own runs) ===")
    for name, (esc, mean, sd) in published.items():
        label = PUBLISHED.get(name, ('Degree-matched', 0, 0, 0))[0]
        print(f'  {label:<14} escape={esc * 100:5.1f}%  mean={mean:.4f}  sd={sd:.4f}'
              f'   ({len(pools[name])} distinct seeds)')

    print('\n=== reimplementation, fresh RNG ===')
    for name in ['sims_bc', 'sims_cc']:
        label, p_esc, p_mean, p_sd = PUBLISHED[name]
        pool = pools[name]
        t0 = time.time()
        finals = [run(int(rng.choice(pool)), offsets, neighbours, rng) for _ in range(TRIALS)]
        esc = sum(1 for v in finals if v >= ESCAPE) / len(finals)
        print(f'  {label:<14} escape={esc * 100:5.1f}%  mean={statistics.mean(finals):.4f}'
              f'  sd={statistics.stdev(finals):.4f}   '
              f'published: {p_esc * 100:.1f}% / {p_mean:.3f} / {p_sd:.3f}'
              f'   ({time.time() - t0:.1f}s)')

    t0 = time.time()
    finals = [run(int(rng.integers(N)), offsets, neighbours, rng) for _ in range(TRIALS)]
    esc = sum(1 for v in finals if v >= ESCAPE) / len(finals)
    print(f'  {"Random":<14} escape={esc * 100:5.1f}%  mean={statistics.mean(finals):.4f}'
          f'  sd={statistics.stdev(finals):.4f}   ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
