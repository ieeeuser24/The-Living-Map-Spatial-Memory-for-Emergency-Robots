"""Scenario runner: Writer explores, Writer is disabled, Executor completes the mission from the beacon chain."""
from dataclasses import dataclass
import math
import numpy as np
from world import World
from protocol import Frame
from network import Net, CommandPost
from robots import Writer, Executor

LAT0, LON0, THETA0 = 37.2744, 9.8739, 60.0


@dataclass
class Config:
    n_beacons: int = 10
    reserve: int = 4
    extra_loss: float = 0.0           # extra packet loss probability
    uplink_outage: tuple = None       # (start, end) in s
    p_brief_corrupt: float = 0.0
    dead_beacon: bool = False         # one beacon dies when the Writer fails
    fail_delay: float = 25.0          # Writer failure this long after the victim reaches the command post
    t_max: float = 1500.0


class Sim:
    def __init__(self, seed=0, cfg=None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(seed)
        self.logs = []
        self.t = 0.0
        self.w = World()
        # the gateway knows the entrance only approximately (GPS fix and surveyed bearing)
        ft = Frame(LAT0, LON0, THETA0)
        r = self.rng
        dn, de = r.normal(0, 1.0, 2)      # averaged GPS fix, 1 m (1 sigma) per axis
        fe = Frame(LAT0 + math.degrees(dn / 6378137.0),
                   LON0 + math.degrees(de / (6378137.0 * math.cos(math.radians(LAT0)))),
                   THETA0 + r.normal(0, 0.5))
        self.net = Net(self.w, r, self.cfg, fe, ft, self.log)
        self.cp = CommandPost(self.net, r, self.cfg, self.log)
        self.net.cp = self.cp
        self.writer = Writer(self.w, r, self.net, self.cfg, self.log)
        self.exec = Executor(self.w, r, self.net, self.cp, self.cfg, self.log)
        self.t_fail = self.t_exec_start = None
        self.brief_pending = None
        self.phase = "WRITER"

    def log(self, t, msg):
        self.logs.append((t, msg))

    def step(self, dt=0.2):
        t = self.t
        w, cp, ex = self.writer, self.cp, self.exec
        w.step(t, dt)
        self.net.step(t)
        link_ok = w.state not in ("DISABLED", "STOP") and self.net.best_node(w.pos)[1] >= -110
        cp.step(t, link_ok)
        if w.state != "DISABLED" and cp.first_victim_t is not None and t >= cp.first_victim_t + self.cfg.fail_delay:
            w.state = "DISABLED"
            self.t_fail = t
            self.phase = "LOST"
            self.log(t, "WRITER FAILS (tipped over): no more heartbeat")
            if self.cfg.dead_beacon:
                cand = [n for n in self.net.nodes.values() if n.id != 0 and n.id != cp.events[-1]["beacon"]]
                if cand:
                    n = cand[int(self.rng.integers(len(cand)))]
                    n.alive = False
                    self.log(t, f"B{n.id} battery dead")
        if cp.briefing is not None and ex.state == "IDLE" and (self.brief_pending is None or t >= self.brief_pending):
            if self.brief_pending is None:
                self.brief_pending = t + 5.0
            elif ex.give_briefing(t, cp.briefing):
                self.t_exec_start = t
                self.phase = "EXECUTOR"
            else:
                cp.briefing = None
                cp.brief_ready = t + 10.0
                self.brief_pending = None
        ex.step(t, dt)
        self.t += dt

    def finished(self):
        return self.exec.state in ("DONE", "FAIL") or self.t >= self.cfg.t_max or \
            (self.writer.state == "STOP" and self.cp.first_victim_t is None)

    def run(self, hook=None):
        while not self.finished():
            self.step()
            if hook:
                hook(self)
        return self.metrics()

    # ---------------------------------------------------------------- metrics
    def metrics(self):
        w, ex, net = self.writer, self.exec, self.net
        truth = self.w.truth_events()
        code = {1: "gas", 2: "collapse", 3: "victim"}
        rec = {}
        fp = 0
        for k, p in w.recorded:
            hit = [n for n, c in truth.items() if n == code[k] and math.hypot(*(np.array(c) - p)) < 15]
            if hit:
                rec[hit[0]] = True
            else:
                fp += 1
        sent, delivered = len(net.sent), len(net.delivered)
        lat = [net.delivered[k] - net.sent[k]["t"] for k in net.delivered]
        errs = [e["err"] for e in net.errors]
        ev_off = []
        for k, p in [(n.id, n) for n in net.nodes.values() if n.id]:
            pass
        known = w.known == 0
        cov = float((known & self.w.reachable).sum()) / self.w.reachable_count
        return dict(
            beacons_used=w.used, coverage=cov, events_detected=sorted(rec), false_positives=fp,
            recall=len(rec) / 3.0, precision=(len(w.recorded) - fp) / max(1, len(w.recorded)),
            records_sent=sent, records_delivered=delivered, delivery_rate=delivered / max(1, sent),
            mean_latency=float(np.mean(lat)) if lat else None,
            pos_err_mean=float(np.mean(errs)) if errs else None, pos_err_max=float(np.max(errs)) if errs else None,
            writer_fail_t=self.t_fail, writer_done_t=w.t_done,
            exec_success=ex.state == "DONE", exec_state=ex.state,
            exec_time=(ex.t_done - self.t_exec_start) if (ex.t_done and self.t_exec_start) else None,
            exec_dist=ex.dist, briefing_rejects=ex.corrupt_seen, sim_time=self.t)
