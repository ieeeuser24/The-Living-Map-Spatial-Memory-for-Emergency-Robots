"""Monte Carlo evaluation: nominal and stress scenarios (randomized sensor noise, drift, packet loss, faults)."""
import json
import sys
import numpy as np
from sim import Sim, Config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def scenario(name, seed):
    rng = np.random.default_rng(10_000 + seed)
    if name == "nominal":
        return Config(fail_delay=float(rng.uniform(5, 60)))
    t0 = float(rng.uniform(100, 200))
    return Config(fail_delay=float(rng.uniform(5, 60)), extra_loss=0.15, uplink_outage=(t0, t0 + 60.0),
                  p_brief_corrupt=0.3, dead_beacon=True)


def summarize(runs):
    def stat(key, scale=1.0):
        v = [r[key] for r in runs if r[key] is not None]
        return (float(np.mean(v)) * scale, float(np.std(v)) * scale, float(np.max(v)) * scale) if v else (None,) * 3
    ok = [r for r in runs if r["exec_success"]]
    return dict(
        runs=len(runs), exec_success=len(ok) / len(runs),
        recall=stat("recall"), precision=stat("precision"), beacons=stat("beacons_used"),
        coverage_at_fail=stat("coverage", 100), delivery=stat("delivery_rate", 100), latency=stat("mean_latency"),
        pos_err=stat("pos_err_mean"), pos_err_max=stat("pos_err_max"),
        exec_time=stat("exec_time"), exec_dist=stat("exec_dist"), false_pos=sum(r["false_positives"] for r in runs),
        rejects=sum(r["briefing_rejects"] for r in runs))


if __name__ == "__main__":
    out = {}
    for name in ("nominal", "stress"):
        runs = []
        for i in range(N):
            import time; t0 = time.time()
            m = Sim(i, scenario(name, i)).run()
            runs.append(m)
            print(name, i, round(time.time() - t0, 1), m['exec_state'], flush=True)
        out[name] = dict(summary=summarize(runs), runs=runs)
        print(name, json.dumps(out[name]["summary"], indent=1))
    json.dump(out, open("results.json", "w"), indent=1, default=str)
