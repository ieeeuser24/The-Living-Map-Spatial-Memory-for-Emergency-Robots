"""Writer and Executor robots: LiDAR mapping, frontier exploration, A* navigation, event logic."""
import heapq
import math
from collections import deque
import numpy as np
from scipy import ndimage
from world import NX, NY, w2c, c2w
from protocol import EVENTS
from network import verify

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
NEI4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEI8 = NEI4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


class Robot:
    def __init__(self, world, rng, start, name):
        self.w, self.rng, self.name = world, rng, name
        self.pos = np.array(start, float)
        self.drift = np.zeros(2)
        self.phi = rng.uniform(0, 2 * math.pi)
        self.theta, self.speed, self.dist = 0.0, 1.0, 0.0
        self.known = np.full((NX, NY), -1, np.int8)
        self.black = np.zeros((NX, NY), bool)
        self.path, self.ranges, self.rub = [], None, None
        self.target_cell, self.last_plan = None, -9.0
        self.reset_ids = set()
        self.hist = deque(maxlen=15)

    @property
    def est(self):
        return self.pos + self.drift

    # --------------------------------------------------------------- sensing
    def scan(self):
        ranges, rub, (fi, fj), (hi, hj) = self.w.lidar(self.pos, self.theta)
        self.known[fi, fj] = 0
        self.known[hi, hj] = 1
        for dx, dy in NEI4:
            ni, nj = np.clip(hi + dx, 0, NX - 1), np.clip(hj + dy, 0, NY - 1)
            m = self.known[ni, nj] == -1
            self.known[ni[m], nj[m]] = 1
        self.ranges, self.rub = ranges, rub

    def travel_dir(self):
        """Direction of travel over the last ~3 m (robust against local heading jitter)."""
        if len(self.hist) >= 5:
            d = self.pos - self.hist[0]
            if np.hypot(*d) > 1.0:
                return math.atan2(d[1], d[0])
        return self.theta

    def ray_at(self, ang):
        n = len(self.ranges)
        return self.ranges[int(round(((ang - self.theta) % (2 * math.pi)) / (2 * math.pi / n))) % n]

    def side_ranges(self):
        a = self.travel_dir()
        return self.ray_at(a), self.ray_at(a + math.pi / 2), self.ray_at(a - math.pi / 2)    # forward, left, right

    # -------------------------------------------------------------- movement
    def follow(self, dt):
        while self.path and np.hypot(*(self.path[0] - self.pos)) < 0.3:
            self.path.pop(0)
        if not self.path:
            return False
        tgt = self.path[0]
        for p in self.path[:4]:
            tgt = p
            if np.hypot(*(p - self.pos)) > 1.0:
                break
        d = tgt - self.pos
        n = float(np.hypot(*d))
        if n < 1e-6:
            return False
        mv = d / n * min(self.speed * dt, n)
        new = self.pos + mv
        if not self.w.is_free(*new):
            self.path = []
            return False
        self.pos = new
        self.hist.append(new.copy())
        self.theta = math.atan2(mv[1], mv[0])
        ds = float(np.hypot(*mv))
        self.dist += ds
        self.phi += self.rng.normal(0, 0.03)
        self.drift += ds * 0.012 * np.array([math.cos(self.phi), math.sin(self.phi)])
        return True

    def pose_reset(self, node):
        """BLE proximity to a beacon: adopt the beacon's recorded position (removes accumulated drift)."""
        self.drift = np.array(node.rec) - node.pos + self.rng.normal(0, 0.15, 2)

    # -------------------------------------------------------------- planning
    def _free_ok(self):
        infl = ndimage.binary_dilation(self.known == 1, iterations=1)
        return (self.known == 0) & ~infl

    def plan_frontier(self):
        ok = self._free_ok()
        unk = ndimage.binary_dilation(self.known == -1, structure=CROSS)
        fr = ok & unk & ~self.black
        sc = w2c(*self.pos)
        if not ok[sc]:
            found = None
            for r in range(1, 5):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        c = (sc[0] + dx, sc[1] + dy)
                        if 0 <= c[0] < NX and 0 <= c[1] < NY and ok[c]:
                            found = c
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None
            sc = found
        prev = {sc: None}
        dq = deque([sc])
        goal = None
        while dq:
            c = dq.popleft()
            if fr[c] and c != sc:
                goal = c
                break
            for dx, dy in NEI4:
                n = (c[0] + dx, c[1] + dy)
                if 0 <= n[0] < NX and 0 <= n[1] < NY and ok[n] and n not in prev:
                    prev[n] = c
                    dq.append(n)
        if goal is None:
            return None
        cells = []
        c = goal
        while c is not None:
            cells.append(c)
            c = prev[c]
        cells.reverse()
        self.target_cell = goal
        return [np.array(c2w(*c)) for c in cells[1:]]

    def mark_black_if_stale(self):
        if self.target_cell is None:
            return
        c = self.target_cell
        unk = ndimage.binary_dilation(self.known == -1, structure=CROSS)
        if unk[c] and np.hypot(*(np.array(c2w(*c)) - self.pos)) < 1.0:
            self.black[max(0, c[0] - 2):c[0] + 3, max(0, c[1] - 2):c[1] + 3] = True

    def astar(self, goal_w, unknown_free=True, max_exp=60000):
        blocked = ndimage.binary_dilation(self.known == 1, iterations=1)
        if not unknown_free:
            blocked |= self.known == -1
        sc = w2c(*self.pos)
        gc = w2c(*goal_w)
        gc = (min(max(gc[0], 0), NX - 1), min(max(gc[1], 0), NY - 1))
        if blocked[gc]:
            best = None
            for r in range(1, 8):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        c = (gc[0] + dx, gc[1] + dy)
                        if 0 <= c[0] < NX and 0 <= c[1] < NY and not blocked[c]:
                            d = dx * dx + dy * dy
                            if best is None or d < best[0]:
                                best = (d, c)
                if best:
                    break
            if not best:
                return None
            gc = best[1]
        blocked[sc] = False

        def h(c):
            dx, dy = abs(c[0] - gc[0]), abs(c[1] - gc[1])
            return 1.2 * (max(dx, dy) + 0.414 * min(dx, dy))

        g, prev = {sc: 0.0}, {sc: None}
        heap = [(h(sc), sc)]
        exp = 0
        while heap and exp < max_exp:
            _, c = heapq.heappop(heap)
            exp += 1
            if c == gc:
                cells = []
                while c is not None:
                    cells.append(c)
                    c = prev[c]
                cells.reverse()
                return [np.array(c2w(*x)) for x in cells[1:]]
            for dx, dy in NEI8:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < NX and 0 <= n[1] < NY) or blocked[n]:
                    continue
                ng = g[c] + (1.414 if dx and dy else 1.0)
                if ng < g.get(n, 1e9):
                    g[n], prev[n] = ng, c
                    heapq.heappush(heap, (ng + h(n), n))
        return None


class Writer(Robot):
    INSPECT_MIN = {1: 5, 3: 1, 2: 2}
    CONF_DIV = {1: 25.0, 3: 5.0, 2: 10.0}

    def __init__(self, world, rng, net, cfg, log):
        super().__init__(world, rng, (0.0, 0.0), "Writer")
        self.net, self.cfg, self.log = net, cfg, log
        self.state = "EXPLORE"
        self.cnt = {1: 0, 2: 0, 3: 0}
        self.used, self.recorded = 0, []
        self.t_inspect = self.dwell_end = 0.0
        self.cool_until = {}
        self.t_done = None
        self.false_pos = 0

    # ------------------------------------------------------------ detection
    def known_event(self, et):
        e = self.est
        rad = 22.0 if et == 1 else 12.0
        return any(k == et and math.hypot(e[0] - p[0], e[1] - p[1]) < rad for k, p in self.recorded)

    def sense(self, t):
        c = self.cnt
        g = self.w.gas_read(self.pos, self.rng)
        c[1] = c[1] + 1 if g > 8.0 else max(0, c[1] - 2)
        human = any(k == "human" for k, _ in self.w.thermal(self.pos, self.rng))
        c[3] = c[3] + 1 if human else max(0, c[3] - 1)
        rub = int(self.rub.sum())
        c[2] = c[2] + 1 if rub >= 3 else max(0, c[2] - 1)
        for et in (1, 2, 3):
            if self.known_event(et) or t < self.cool_until.get(et, 0):
                c[et] = 0

    def conf(self, et):
        return min(1.0, self.cnt[et] / self.CONF_DIV[et])

    # ------------------------------------------------------------- beacons
    def spacing(self):
        e = self.est
        return min(math.hypot(e[0] - n.rec[0], e[1] - n.rec[1]) for n in self.net.nodes.values() if n.alive)

    def begin_drop(self, t, event, conf):
        if self.cfg.n_beacons - self.used <= 0:
            self.log(t, "Writer: beacon magazine empty")
            return False
        self.net.add_beacon(t, self.pos, self.est, event, conf)
        self.used += 1
        if event in (1, 2, 3):
            self.recorded.append((event, self.est.copy()))
            self.cnt[event] = 0
        self.state, self.dwell_end, self.path = "DROP", t + 2.0, []
        return True

    def structural_drop(self, t):
        remaining = self.cfg.n_beacons - self.used
        sp = self.spacing()
        _, best_r = self.net.best_node(self.pos)
        fwd, left, right = self.side_ranges()
        kind = None
        if sp >= 6.0 and (left > 5.0 or right > 5.0):
            kind = 4
        elif sp >= 6.0 and fwd < 2.5 and left < 2.5 and right < 2.5 and not self.rub.any():
            kind = 5
        elif best_r < -105 or sp > 25.0:
            kind = 0
        if kind is None:
            return False
        critical = best_r < -110
        if remaining > self.cfg.reserve or (critical and remaining > 0):
            return self.begin_drop(t, kind, 0.8)
        return False

    # ----------------------------------------------------------------- step
    def step(self, t, dt):
        if self.state in ("DISABLED", "STOP"):
            return
        self.scan()
        for n in self.net.nodes.values():
            if n.alive and np.hypot(*(n.pos - self.pos)) < 3.0 and n.id not in self.reset_ids:
                self.pose_reset(n)
                self.reset_ids.add(n.id)
            elif np.hypot(*(n.pos - self.pos)) > 6.0:
                self.reset_ids.discard(n.id)
        if self.state == "DROP":
            if t >= self.dwell_end:
                self.state = "EXPLORE"
            return
        self.sense(t)
        if self.state == "INSPECT":
            best = max((1, 2, 3), key=self.conf)
            if self.conf(best) >= 0.6:
                if not self.begin_drop(t, best, self.conf(best)):
                    self.state = "EXPLORE"
                    self.cool_until[best] = t + 30
                return
            if all(v == 0 for v in self.cnt.values()) or t - self.t_inspect > 8.0:
                self.state = "EXPLORE"
            return
        if self.state == "EXPLORE":
            for et, thr in self.INSPECT_MIN.items():
                if self.cnt[et] >= thr:
                    self.state, self.t_inspect, self.path = "INSPECT", t, []
                    return
            if self.structural_drop(t):
                return
            if (not self.path) or t - self.last_plan >= 1.0:
                self.mark_black_if_stale()
                p = self.plan_frontier()
                self.last_plan = t
                if p is None:
                    self.state, self.t_done = "RETURN", t
                    self.log(t, "Writer: exploration complete, returning to entrance")
                    self.path = []
                else:
                    self.path = p
            self.follow(dt)
        elif self.state == "RETURN":
            if not self.path:
                self.path = self.astar(np.array([0.0, 0.0]), unknown_free=False) or self.astar(np.array([0.0, 0.0])) or []
            self.follow(dt)
            if np.hypot(*self.pos) < 1.5:
                self.state = "STOP"
                self.log(t, "Writer: back at the entrance")


class Executor(Robot):
    def __init__(self, world, rng, net, cp, cfg, log):
        super().__init__(world, rng, (-4.0, 0.0), "Executor")
        self.net, self.cp, self.cfg, self.log = net, cp, cfg, log
        self.state, self.nodes, self.idx = "IDLE", [], 1
        self.t_brief = self.t_done = self.t_verified = None
        self.gas_cnt, self.gas_done, self.det, self.t_target = 0, False, 0, 0.0
        self.victim_est = None
        self.corrupt_seen = 0

    def give_briefing(self, t, br):
        data = verify(br)
        if data is None:
            self.corrupt_seen += 1
            self.log(t, "Executor: briefing signature check FAILED, re-requesting")
            return False
        self.nodes, self.state, self.t_brief, self.idx = data["path"], "GO", t, 1
        self.log(t, f"Executor: briefing verified ({len(self.nodes) - 1} beacons), entering tunnel")
        return True

    def step(self, t, dt):
        if self.state in ("IDLE", "DONE", "FAIL"):
            return
        self.scan()
        if self.state == "GO" and self.gas_check(t):
            pass
        if self.state == "GO":
            bid, x, y = self.nodes[self.idx]
            node = self.net.nodes.get(bid)
            near = node is not None and node.alive and np.hypot(*(node.pos - self.pos)) < 3.0
            est_near = np.hypot(x - self.est[0], y - self.est[1]) < 1.5
            if near or (est_near and (node is None or not node.alive)):
                if near:
                    self.pose_reset(node)
                self.log(t, f"Executor reaches B{bid} ({'BLE pose reset' if near else 'odometry only'})")
                self.idx += 1
                self.path = []
                if self.idx >= len(self.nodes):
                    self.state, self.t_target, self.det, self.path = "TARGET", t, 0, []
                    return
                bid, x, y = self.nodes[self.idx]
            if (not self.path) or t - self.last_plan >= 1.0:
                goal = np.array([x, y]) - self.drift
                self.path = self.astar(goal) or []
                self.last_plan = t
            self.follow(dt)
        elif self.state in ("TARGET", "SEARCH", "APPROACH"):
            th = [p for k, p in self.w.thermal(self.pos, self.rng) if k == "human"]
            if th:
                self.det += 1
                self.victim_est = th[0]
            if self.state != "APPROACH" and self.det >= 3:
                self.state, self.t_verified, self.path = "APPROACH", t, []
                self.log(t, "Executor: victim verified by thermal camera, approaching")
            if self.state == "APPROACH":
                if (not self.path) or t - self.last_plan >= 1.0:
                    self.path = self.astar(self.victim_est) or []
                    self.last_plan = t
                self.follow(dt)
                if np.hypot(*(self.pos - self.w.VICTIM)) < 1.5:
                    self.state, self.t_done = "DONE", t
                    self.log(t, "Executor: victim reached, locator delivered. MISSION COMPLETE")
            else:
                if self.state == "TARGET" and t - self.t_target > 8.0:
                    self.state = "SEARCH"
                    self.log(t, "Executor: no heat signature at the beacon, local frontier search")
                if self.state == "SEARCH":
                    if (not self.path) or t - self.last_plan >= 1.0:
                        self.mark_black_if_stale()
                        self.path = self.plan_frontier() or []
                        self.last_plan = t
                    self.follow(dt)
                    if t - self.t_target > 120.0:
                        self.state, self.t_done = "FAIL", t
                        self.log(t, "Executor: mission FAILED (target not found)")

    def gas_check(self, t):
        if self.gas_done:
            return False
        g = self.w.gas_read(self.pos, self.rng)
        self.gas_cnt = self.gas_cnt + 1 if g > 8.0 else max(0, self.gas_cnt - 2)
        if self.gas_cnt >= 15:
            self.gas_done = True
            self.cp.refresh(t + 2.0, 1, self.est)
        return False
