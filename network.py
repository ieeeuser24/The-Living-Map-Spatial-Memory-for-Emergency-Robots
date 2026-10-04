"""Beacon chain (multi-hop LoRa relay), entrance gateway and command post."""
import hashlib
import hmac
import json
import math
import numpy as np
from protocol import pack, unpack, rssi, p_success, confidence, EVENTS

KEY = b"living-map-demo-key"


class Node:
    def __init__(self, nid, pos, rec, parent, hop):
        self.id, self.pos, self.rec, self.parent, self.hop = nid, np.array(pos, float), tuple(rec), parent, hop
        self.alive = True
        self.outbox = []            # [raw bytes, next attempt time, tries]


class Net:
    def __init__(self, world, rng, cfg, frame_est, frame_true, log):
        self.w, self.rng, self.cfg, self.fe, self.ft, self.log = world, rng, cfg, frame_est, frame_true, log
        self.nodes = {0: Node(0, (0, 0), (0, 0), 0, 0)}      # node 0 = entrance gateway
        self.seen, self.cp_queue, self.hops = set(), [], []
        self.sent, self.delivered, self.errors = {}, {}, []
        self.cp = None

    # ------------------------------------------------------------------ link
    def link(self, a, b):
        r = rssi(self.eff_dist(a, b), self.w.los(a, b), self.rng)
        return r, p_success(r) * (1 - self.cfg.extra_loss)

    def eff_dist(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if self.w.los(a, b):
            return float(np.hypot(*(a - b)))
        return max(float(np.hypot(*(a - b))), self.w.geo_dist(a, b))

    def best_node(self, pos, hop_lt=None, exclude=()):
        best, br = None, -999.0
        for n in self.nodes.values():
            if not n.alive or n.id in exclude or (hop_lt is not None and n.hop >= hop_lt):
                continue
            r = rssi(self.eff_dist(n.pos, pos), self.w.los(n.pos, pos), None)
            if r > br:
                best, br = n, r
        return best, br

    def uplink_up(self, t):
        o = self.cfg.uplink_outage
        return not (o and o[0] <= t < o[1])

    # ---------------------------------------------------------------- beacons
    def add_beacon(self, t, pos_true, est, event, conf):
        parent, _ = self.best_node(pos_true)
        nid = max(self.nodes) + 1
        bearing = math.degrees(math.atan2(parent.rec[1] - est[1], parent.rec[0] - est[0]))
        raw = pack(nid, parent.id, event, est[0], est[1], bearing, t, conf, parent.hop + 1)
        rec = unpack(raw)
        node = Node(nid, pos_true, (rec["x"], rec["y"]), parent.id, parent.hop + 1)
        node.outbox.append([raw, t + 0.5, 0])
        self.nodes[nid] = node
        self.sent[(nid, event, int(t))] = dict(t=t, true=np.array(pos_true, float), event=event, beacon=nid)
        self.log(t, f"Writer drops B{nid} ({EVENTS[event]}), parent B{parent.id}, hop {parent.hop + 1}")
        return node

    def step(self, t):
        for n in list(self.nodes.values()):
            if n.id == 0 or not n.alive or not n.outbox:
                continue
            raw, nt, tries = n.outbox[0]
            if t < nt:
                continue
            parent = self.nodes[n.parent]
            ok = False
            if parent.alive:
                _, p = self.link(n.pos, parent.pos)
                ok = self.rng.random() < p
            if ok:
                n.outbox.pop(0)
                self.hops.append((t, n.pos.copy(), parent.pos.copy()))
                if parent.id == 0:
                    self.gateway_receive(raw, t)
                else:
                    parent.outbox.append([raw, t + 1.0, 0])
            else:
                tries += 1
                if tries >= 3:
                    alt, _ = self.best_node(n.pos, hop_lt=n.hop, exclude=(n.id, n.parent))
                    if alt is not None:
                        n.parent, n.hop = alt.id, alt.hop + 1
                        self.log(t, f"B{n.id} re-routes via B{alt.id}")
                    n.outbox[0] = [raw, t + 10.0, 0]
                else:
                    n.outbox[0] = [raw, t + 1.0, tries]
        if self.uplink_up(t):
            ready = [q for q in self.cp_queue if t >= q[0]]
            self.cp_queue = [q for q in self.cp_queue if t < q[0]]
            for _, rec, gps in ready:
                self.cp.on_record(rec, gps, t)

    def gateway_receive(self, raw, t):
        rec = unpack(raw)
        if rec is None:
            return
        key = (rec["beacon_id"], rec["event"], rec["ts"])
        if key in self.seen:
            return
        self.seen.add(key)
        gps = self.fe.to_gps(rec["x"], rec["y"])
        if key in self.sent:
            self.delivered[key] = t
            truth = self.ft.to_gps(*self.sent[key]["true"])
            self.errors.append(dict(key=key, event=rec["event"], err=self.fe.gps_diff_m(gps, truth)))
        self.log(t, f"Gateway: record B{rec['beacon_id']} ({EVENTS[rec['event']]}) translated to GPS")
        self.cp_queue.append((t + 1.5, rec, gps))


class CommandPost:
    def __init__(self, net, rng, cfg, log):
        self.net, self.rng, self.cfg, self.log = net, rng, cfg, log
        self.events, self.chain = [], {0: dict(x=0.0, y=0.0, parent=None)}
        self.last_hb, self.writer_lost, self.briefing, self.brief_ready = 0.0, False, None, None
        self.first_victim_t = None
        self.sent_n = 0

    def on_record(self, rec, gps, t):
        self.chain[rec["beacon_id"]] = dict(x=rec["x"], y=rec["y"], parent=rec["prev_id"])
        if rec["event"] in (1, 2, 3):
            self.events.append(dict(beacon=rec["beacon_id"], etype=rec["event"], gps=gps, x=rec["x"], y=rec["y"],
                                    c0=rec["conf"], t0=rec["ts"], t_rx=t))
            if rec["event"] == 3 and self.first_victim_t is None:
                self.first_victim_t = t
            self.log(t, f"Command post: {EVENTS[rec['event']]} event on map (B{rec['beacon_id']})")

    def conf(self, e, t):
        return confidence(e["c0"], e["t0"], t, e["etype"])

    def refresh(self, t, etype, xy):
        for e in self.events:
            if e["etype"] == etype and math.hypot(e["x"] - xy[0], e["y"] - xy[1]) < 18:
                e["c0"], e["t0"] = 0.9, t
                self.log(t, f"Command post: {EVENTS[etype]} re-verified by Executor")
                return

    def step(self, t, writer_alive_link):
        if writer_alive_link and int(t * 5) % 50 == 0:
            self.last_hb = t + 2.0
        if not self.writer_lost and t > 40 and t - self.last_hb > 30:
            self.writer_lost = True
            self.log(t, "Command post: Writer heartbeat lost -> preparing briefing")
        if self.writer_lost and self.briefing is None and self.brief_ready is None:
            v = [e for e in self.events if e["etype"] == 3]
            if v:
                self.brief_ready = t + 5.0
        if self.brief_ready is not None and t >= self.brief_ready and self.briefing is None:
            self.briefing = self.build_briefing(t)
            self.log(t, "Command post: signed mission briefing sent to Executor")

    def build_briefing(self, t):
        target = [e for e in self.events if e["etype"] == 3][0]
        path, b = [], target["beacon"]
        while b is not None:
            c = self.chain[b]
            path.append([b, c["x"], c["y"]])
            b = c["parent"] if b != 0 else None
        path.reverse()
        hazards = [dict(type=e["etype"], x=e["x"], y=e["y"], conf=round(self.conf(e, t), 2), age=round(t - e["t0"]))
                   for e in self.events]
        payload = json.dumps(dict(path=path, target=target["beacon"], hazards=hazards), separators=(",", ":"))
        mac = hmac.new(KEY, payload.encode(), hashlib.sha256).hexdigest()
        if self.rng.random() < self.cfg.p_brief_corrupt:
            payload = payload.replace("[", "{", 1)         # transmission error
        return dict(payload=payload, mac=mac)


def verify(briefing):
    mac = hmac.new(KEY, briefing["payload"].encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, briefing["mac"]):
        return None
    return json.loads(briefing["payload"])
