import numpy as np
from scipy.optimize import root, minimize
import pandas as pd
import json
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=11)
parser.add_argument("--configs", type=int, default=500)
parser.add_argument("--starts", type=int, default=800)
parser.add_argument("--mode", type=str, default="both", choices=["mixed", "positive", "both"])
parser.add_argument("--offset", type=int, default=0)
args = parser.parse_args()

N = 3
BOUND = (N - 1) ** 2  # 4
SAVE_DISTANCE = 0     # only save perfect hits for n=3

os.makedirs("results/best_configs", exist_ok=True)
DATA_FILE = "results/raw_data.csv"

R_MIN = 1e-6
R_MAX = 1e4


def electric_field(x, pos, charges):
    x = np.asarray(x, dtype=float).flatten()
    F = np.zeros(3)
    for p, q in zip(pos, charges):
        dx = x - p
        r = float(np.linalg.norm(dx))
        if r < R_MIN or r > R_MAX:
            return np.full(3, np.nan)
        r3 = r * r * r
        F += q * dx / r3
    return F


def jacobian_E(x, pos, charges):
    x = np.asarray(x, dtype=float).flatten()
    J = np.zeros((3, 3))
    for p, q in zip(pos, charges):
        dx = x - p
        r = float(np.linalg.norm(dx))
        if r < R_MIN or r > R_MAX:
            return np.full((3, 3), np.nan)
        r2 = r * r
        r3 = r2 * r
        r5 = r3 * r2
        J += q * (np.eye(3) / r3 - 3.0 * np.outer(dx, dx) / r5)
    return J


def is_non_degenerate(x, pos, charges):
    J = jacobian_E(x, pos, charges)
    if np.any(np.isnan(J)):
        return False
    return abs(np.linalg.det(J)) > 1e-8


def sample_charges(n, mode, rng):
    if mode == "positive":
        return rng.uniform(0.1, 1.0, n)
    else:
        charges = rng.uniform(-1, 1, n)
        return np.where(np.abs(charges) < 0.01, 0.01 * np.sign(charges + 1e-15), charges)


def find_critical_points(pos, charges, n_starts):
    pos = np.array(pos, dtype=float)
    charges = np.array(charges, dtype=float)

    # Normalisation
    center = np.mean(pos, axis=0)
    pos -= center
    scale = float(np.max(np.linalg.norm(pos, axis=1))) or 1.0
    pos /= scale

    rng = np.random.default_rng()
    starts = []

    starts.append(rng.uniform(-6, 6, (n_starts // 2, 3)))

    per_charge = max(1, (n_starts * 3) // (10 * N))
    for p in pos:
        starts.append(p + rng.normal(0, 0.5, (per_charge, 3)))

    per_pair = max(1, (n_starts * 2) // (10 * N * N))
    for i in range(N):
        for j in range(i + 1, N):
            mid = (pos[i] + pos[j]) / 2.0
            starts.append(mid + rng.normal(0, 0.3, (per_pair, 3)))
            qi, qj = abs(charges[i]), abs(charges[j])
            wmid = (qi * pos[j] + qj * pos[i]) / (qi + qj)
            starts.append(wmid + rng.normal(0, 0.3, (per_pair, 3)))

    starts = np.vstack(starts)[:n_starts]

    critical_points = []

    def try_add(x):
        x = np.asarray(x, dtype=float).flatten()
        F = electric_field(x, pos, charges)
        if np.any(np.isnan(F)) or np.linalg.norm(F) > 1e-6:
            return
        if np.min(np.linalg.norm(pos - x, axis=1)) < 1e-5:
            return
        if any(np.linalg.norm(x - c) < 1e-4 for c in critical_points):
            return
        if is_non_degenerate(x, pos, charges):
            critical_points.append(x.copy())

    for x0 in starts:
        try:
            res = root(electric_field, x0, args=(pos, charges),
                       method='hybr', tol=1e-12,
                       options={'maxfev': 400})
            if res.success and np.linalg.norm(res.fun) < 1e-8:
                try_add(res.x)
                continue
        except Exception:
            pass

        try:
            def obj(x):
                F = electric_field(x, pos, charges)
                if np.any(np.isnan(F)):
                    return 1e10
                return float(np.dot(F, F))

            def grad(x):
                F = electric_field(x, pos, charges)
                J = jacobian_E(x, pos, charges)
                if np.any(np.isnan(F)) or np.any(np.isnan(J)):
                    return np.zeros(3)
                return 2.0 * J.T @ F

            res2 = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                            tol=1e-14, options={'maxiter': 500})
            if res2.fun < 1e-11:
                try_add(res2.x)
        except Exception:
            pass

    final_pos = pos * scale + center
    final_crits = (np.array(critical_points) * scale + center
                   if critical_points else None)
    return len(critical_points), final_pos, final_crits


def worker(config_id):
    rng = np.random.default_rng(config_id)

    if args.mode == "both":
        mode = "positive" if config_id % 2 == 0 else "mixed"
    else:
        mode = args.mode

    positions = rng.normal(0, 2.0, size=(N, 3))
    charges = sample_charges(N, mode, rng)

    start_time = time.time()
    num_crit, final_pos, final_crits = find_critical_points(
        positions, charges, args.starts)
    runtime = time.time() - start_time

    distance = BOUND - num_crit

    if distance <= SAVE_DISTANCE and final_crits is not None:
        suffix = "_perfect" if distance == 0 else f"_near{distance}"
        data = {
            "n": N,
            "config_id": int(config_id),
            "mode": mode,
            "charges": charges.tolist(),
            "charge_positions": final_pos.tolist(),
            "critical_points": final_crits.tolist(),
            "num_critical": int(num_crit),
            "distance_to_bound": int(distance),
        }
        with open(f"results/best_configs/n{N}{suffix}_c{config_id}.json", "w") as f:
            json.dump(data, f, indent=2)

    result = {
        "n": N,
        "config_id": config_id,
        "mode": mode,
        "num_critical_points": num_crit,
        "reached_bound": num_crit == BOUND,
        "distance_to_bound": distance,
        "runtime_seconds": round(runtime, 2),
    }
    header = not os.path.exists(DATA_FILE)
    pd.DataFrame([result]).to_csv(DATA_FILE, mode='a', header=header, index=False)

    marker = "PERFECT" if distance == 0 else ("NEAR" if distance <= 2 else "")
    print(f"  Config {config_id:04d} | n={N} | {num_crit:2d}/{BOUND} crit | "
          f"dist={distance} | {runtime:.1f}s | {mode} {marker}")
    return result


if __name__ == "__main__":
    print(f"Maxwell n={N} | bound={BOUND} | mode={args.mode} | "
          f"workers={args.workers} | configs={args.configs} | "
          f"starts={args.starts} | offset={args.offset} | "
          f"saving: distance <= {SAVE_DISTANCE}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, i)
                   for i in range(args.offset, args.offset + args.configs)]
        for future in tqdm(as_completed(futures),
                           total=args.configs, desc=f"n={N}"):
            pass

    print(f"\n=== n={N} RUN FINISHED ===")

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        dfn = df[df['n'] == N]
        if len(dfn):
            print(f"Configs run  : {len(dfn)}")
            print(f"Bound hits   : {dfn['reached_bound'].sum()}")
            print(f"Best dist    : {dfn['distance_to_bound'].min()}")
            print(f"Avg crit     : {dfn['num_critical_points'].mean():.2f}")
            print(dfn['distance_to_bound'].value_counts().sort_index())
