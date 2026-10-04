"""Renders the simulation demo video (MP4, 1280x720, 25 fps)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import imageio.v2 as imageio
from sim import Sim, Config
from world import X0, Y0, NX, NY, RES
from protocol import EVENTS

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = sys.argv[2] if len(sys.argv) > 2 else "demo.mp4"
FPS = 25
BG, PANEL, TXT = "#0F1318", "#171C23", "#E6EAF0"
AMBER, RED, GREEN, BLUE = "#F5B041", "#E74C3C", "#58D68D", "#5DADE2"


def card(lines, size=(12.8, 7.2)):
    fig = plt.figure(figsize=size, dpi=100, facecolor=BG)
    y = 0.86
    for text, fs, col, weight in lines:
        fig.text(0.06, y, text, fontsize=fs, color=col, weight=weight, va="top")
        y -= fs / 380.0 + 0.025
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


class Renderer:
    def __init__(self, sim):
        self.s = sim
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        self.fig = fig
        self.ax = fig.add_axes([0.015, 0.105, 0.655, 0.745], facecolor="#000000")
        ax = self.ax
        ax.set_xlim(-8, 128)
        ax.set_ylim(-34, 36)
        ax.set_aspect("equal")
        ax.tick_params(colors="#8892A0", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#2C343F")
        ax.set_xlabel("x (m), entrance frame", color="#8892A0", fontsize=7)
        ax.set_ylabel("y (m)", color="#8892A0", fontsize=7)
        self.im = ax.imshow(np.zeros((NY, NX, 3)), origin="lower", extent=[X0, X0 + NX * RES, Y0, Y0 + NY * RES],
                            interpolation="nearest", zorder=1)
        (self.wtrail,) = ax.plot([], [], color=RED, lw=0.8, alpha=0.55, zorder=3)
        (self.etrail,) = ax.plot([], [], color=GREEN, lw=0.8, alpha=0.6, zorder=3)
        self.wm = Polygon(np.zeros((3, 2)), closed=True, fc=RED, ec="white", lw=0.8, zorder=8)
        self.em = Polygon(np.zeros((3, 2)), closed=True, fc=GREEN, ec="white", lw=0.8, zorder=8)
        self.west = Circle((0, 0), 1.3, fill=False, ec=RED, lw=0.8, ls="--", zorder=7)
        self.eest = Circle((0, 0), 1.3, fill=False, ec=GREEN, lw=0.8, ls="--", zorder=7)
        for p in (self.wm, self.em, self.west, self.eest):
            ax.add_patch(p)
        ax.add_patch(Rectangle((-1.2, -1.2), 2.4, 2.4, fc="#5DADE2", ec="white", lw=0.8, zorder=6))
        ax.text(0, -3.6, "GATEWAY\n(entrance, GPS)", color=BLUE, fontsize=6.5, ha="center", va="top", zorder=9)
        self.bea, self.flash, self.icons = {}, [], {}
        self.wpts, self.epts = [], []
        self.title = fig.text(0.015, 0.955, "The Living Map: Spatial Memory for Emergency Robots", color=TXT, fontsize=15, weight="bold", va="center")
        self.banner = fig.text(0.015, 0.905, "", color=AMBER, fontsize=11, weight="bold", va="center")
        fig.text(0.675, 0.955, "TSYP14 · Phase 1 · simulation demo", color="#8892A0", fontsize=9, va="center")
        # right panels
        self.cp_ax = fig.add_axes([0.68, 0.47, 0.305, 0.38], facecolor=PANEL)
        self.log_ax = fig.add_axes([0.68, 0.105, 0.305, 0.34], facecolor=PANEL)
        for a, t in ((self.cp_ax, "COMMAND POST: live map table (GPS)"), (self.log_ax, "MESSAGE LOG")):
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#2C343F")
            a.set_title(t, color=AMBER, fontsize=8, loc="left", pad=3)
        self.cp_txt = self.cp_ax.text(0.02, 0.97, "", color=TXT, fontsize=6.6, family="monospace", va="top", transform=self.cp_ax.transAxes)
        self.log_txt = self.log_ax.text(0.02, 0.97, "", color="#C9D1DB", fontsize=6.4, family="monospace", va="top", transform=self.log_ax.transAxes)
        self.bar = fig.text(0.015, 0.045, "", color=TXT, fontsize=9.5, family="monospace", va="center")
        ax.legend(handles=[
            plt.Line2D([], [], marker="^", color="none", mfc=RED, mec="white", ms=7, label="Writer (solid = true pose)"),
            plt.Line2D([], [], marker="o", color="none", mfc="none", mec=RED, ms=7, ls="--", label="estimated pose (drift)"),
            plt.Line2D([], [], marker="^", color="none", mfc=GREEN, mec="white", ms=7, label="Executor"),
            plt.Line2D([], [], marker="o", color="none", mfc=AMBER, mec="#8E5B00", ms=7, label="LoRa beacon")],
            loc="lower left", fontsize=6.5, facecolor=PANEL, edgecolor="#2C343F", labelcolor=TXT, framealpha=0.9)
        self.stride_t = -1

    @staticmethod
    def tri(p, th, s=2.2):
        c, sn = np.cos(th), np.sin(th)
        pts = np.array([[s, 0], [-s * 0.7, s * 0.6], [-s * 0.7, -s * 0.6]])
        return pts @ np.array([[c, sn], [-sn, c]]) + p

    def base_image(self):
        s = self.s
        img = np.zeros((NX, NX * 0 + NY, 3))
        g = s.w.grid
        img[g == 1] = (0.06, 0.06, 0.07)
        img[g == 0] = (0.11, 0.14, 0.19)
        kw, ke = s.writer.known, s.exec.known
        img[kw == 1] = (0.28, 0.30, 0.34)
        img[ke == 1] = (0.28, 0.30, 0.34)
        img[kw == 0] = (0.55, 0.72, 0.90)
        img[(ke == 0) & (kw != 0)] = (0.62, 0.86, 0.66)
        return np.transpose(img, (1, 0, 2))

    def draw(self, t):
        s = self.s
        w, ex, net, cp = s.writer, s.exec, s.net, s.cp
        self.im.set_data(self.base_image())
        self.wpts.append(w.pos.copy())
        self.wtrail.set_data(*np.array(self.wpts).T)
        self.wm.set_xy(self.tri(w.pos, w.theta))
        self.west.center = tuple(w.est)
        if ex.state != "IDLE":
            self.epts.append(ex.pos.copy())
            self.etrail.set_data(*np.array(self.epts).T)
            self.em.set_xy(self.tri(ex.pos, ex.theta))
            self.eest.center = tuple(ex.est)
            self.em.set_visible(True)
            self.eest.set_visible(True)
        else:
            self.em.set_visible(False)
            self.eest.set_visible(False)
        self.wm.set_alpha(0.35 if w.state == "DISABLED" else 1.0)
        # beacons
        for nid, n in net.nodes.items():
            if nid == 0:
                continue
            if nid not in self.bea:
                par = net.nodes[n.parent]
                (ln,) = self.ax.plot([par.pos[0], n.pos[0]], [par.pos[1], n.pos[1]], color=AMBER, lw=0.7, ls=":", alpha=0.8, zorder=4)
                c = Circle(tuple(n.pos), 1.5, fc=AMBER, ec="#8E5B00", lw=1.0, zorder=6)
                self.ax.add_patch(c)
                lb = self.ax.text(n.pos[0], n.pos[1] + 2.6, f"B{nid}", color=AMBER, fontsize=7, weight="bold", ha="center", zorder=9)
                self.bea[nid] = (ln, c, lb)
            if not n.alive:
                self.bea[nid][1].set_facecolor("#555555")
        # relay flashes
        for a in self.flash:
            a.remove()
        self.flash = []
        for (th, a, b) in net.hops[-6:]:
            if t - th < 1.2:
                (l,) = self.ax.plot([a[0], b[0]], [a[1], b[1]], color="#FF8C00", lw=2.4, alpha=0.95, zorder=5)
                self.flash.append(l)
        # event icons appear when the Writer has recorded them
        for e in cp.events:
            k = (e["beacon"], e["etype"])
            if k in self.icons:
                continue
            truth = {1: s.w.GAS_C, 2: s.w.COLLAPSE_C, 3: s.w.VICTIM}[e["etype"]]
            if e["etype"] == 1:
                a = Circle(tuple(truth), 8.0, fc="#F39C12", ec="#F39C12", alpha=0.28, zorder=2)
                self.ax.add_patch(a)
                self.ax.text(truth[0], truth[1] + 9.5, "gas pocket", color="#F39C12", fontsize=7, ha="center", zorder=9)
            elif e["etype"] == 2:
                a = Rectangle((truth[0] - 1.5, -1.6), 3.0, 3.2, fc="#C0392B", ec="white", lw=0.6, hatch="////", zorder=3)
                self.ax.add_patch(a)
                self.ax.text(truth[0], truth[1] + 3.4, "collapse", color="#E6B0AA", fontsize=7, ha="center", zorder=9)
            else:
                a = Circle(tuple(truth), 1.3, fc="#FF5252", ec="white", lw=1.0, zorder=5)
                self.ax.add_patch(a)
                self.ax.text(truth[0], truth[1] + 2.8, "victim (heat)", color="#FF8A80", fontsize=7, ha="center", zorder=9)
            self.icons[k] = a
        # banner
        if ex.state == "DONE":
            ban, col = "MISSION COMPLETE: the Executor reached the victim using only the beacon chain", GREEN
        elif s.phase == "EXECUTOR":
            ban, col = "PHASE 3 · Executor follows the beacon chain, resets its pose at each beacon", GREEN
        elif s.phase == "LOST":
            ban, col = "PHASE 2 · Writer lost: the knowledge survives in the beacons and at the command post", RED
        else:
            ban, col = "PHASE 1 · Writer explores the GPS-denied tunnel and deposits beacons", AMBER
        self.banner.set_text(ban)
        self.banner.set_color(col)
        # command post table
        rows = ["EVENT     BEACON  LAT        LON       CONF  STATUS", "-" * 52]
        for e in cp.events:
            c = cp.conf(e, t)
            st = "RE-VERIFY" if c < 0.3 else ("aging" if c < 0.5 else "valid")
            rows.append(f"{EVENTS[e['etype']]:<9} B{e['beacon']:<5}  {e['gps'][0]:.5f}  {e['gps'][1]:.5f}  {c:.2f}  {st}")
        if not cp.events:
            rows.append("(no event received yet)")
        rows += ["", f"beacon chain known: {len(cp.chain) - 1} beacons"]
        rows.append("Writer link:  " + ("LOST (heartbeat timeout)" if cp.writer_lost else "alive"))
        rows.append("Briefing:     " + ("sent" if cp.briefing is not None else "-") + (" (signed, HMAC)" if cp.briefing is not None else ""))
        rows.append("Uplink:       " + ("LTE up" if net.uplink_up(t) else "DOWN (store-and-forward)"))
        self.cp_txt.set_text("\n".join(rows))
        self.log_txt.set_text("\n".join(f"{lt:6.1f}s {m[:44]}" for lt, m in s.logs[-17:]))
        self.bar.set_text(f"t = {t:6.1f} s   Writer: {w.state:<8}  Beacons: {w.used:>2}/{s.cfg.n_beacons}   "
                          f"Delivered: {len(net.delivered)}/{len(net.sent)}   Executor: {ex.state}")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def main():
    cfg = Config()
    sim = Sim(SEED, cfg)
    wr = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    title = card([
        ("The Living Map", 34, TXT, "bold"),
        ("Spatial Memory for Emergency Robots", 20, AMBER, "bold"),
        ("TSYP14 Technical Challenge · IEEE RAS × IEEE AESS · Phase 1 · Simulation demo", 12, "#8892A0", "normal"),
        ("", 10, TXT, "normal"),
        ("Scenario: underground mine / tunnel (GPS-denied, no network)", 14, TXT, "normal"),
        ("•  Writer robot explores, detects gas / collapse / victim and drops LoRa beacons", 12, "#C9D1DB", "normal"),
        ("•  Beacons store and relay 20-byte records to the entrance gateway (multi-hop)", 12, "#C9D1DB", "normal"),
        ("•  Gateway translates local coordinates to GPS; command post shows a live map", 12, "#C9D1DB", "normal"),
        ("•  The Writer fails; a signed briefing sends the Executor along the beacon chain", 12, "#C9D1DB", "normal"),
        ("", 10, TXT, "normal"),
        (f"Python 2D simulation · random seed {SEED} · all robot, radio and sensor effects are modeled", 10, "#8892A0", "normal")])
    for _ in range(FPS * 5):
        wr.append_data(title)
    rd = Renderer(sim)
    state = {"n": 0}

    def hook(s):
        state["n"] += 1
        stride = 2
        if s.phase == "LOST" and s.exec.state == "IDLE":
            stride = 6
        if state["n"] % stride == 0:
            wr.append_data(rd.draw(s.t))

    m = sim.run(hook)
    last = rd.draw(sim.t)
    for _ in range(FPS * 3):
        wr.append_data(last)
    lines = [("Results of this run", 26, TXT, "bold"), ("", 8, TXT, "normal"),
             (f"Events detected: {', '.join(m['events_detected'])}   (false positives: {m['false_positives']})", 13, "#C9D1DB", "normal"),
             (f"Beacons used: {m['beacons_used']}/10   Records delivered to gateway: {m['records_delivered']}/{m['records_sent']}", 13, "#C9D1DB", "normal"),
             (f"Mean event position error after GPS translation: {m['pos_err_mean']:.1f} m (max {m['pos_err_max']:.1f} m)", 13, "#C9D1DB", "normal"),
             (f"Executor mission: {'SUCCESS' if m['exec_success'] else 'FAILED'} in {m['exec_time']:.0f} s from the briefing", 13, GREEN if m["exec_success"] else RED, "bold")]
    if os.path.exists("results.json"):
        R = json.load(open("results.json"))
        for name in ("nominal", "stress"):
            sm = R[name]["summary"]
            lines.append((f"{name.capitalize()} ({sm['runs']} randomized runs): Executor success {sm['exec_success'] * 100:.0f} %, "
                          f"event recall {sm['recall'][0] * 100:.0f} %, mean position error {sm['pos_err'][0]:.1f} m",
                          12, AMBER, "normal"))
    lines += [("", 8, TXT, "normal"), ("Code, scenario and Monte Carlo scripts: see the repository (/sim).", 11, "#8892A0", "normal")]
    end = card(lines)
    for _ in range(FPS * 6):
        wr.append_data(end)
    wr.close()
    print("video written", OUT, "frames", state["n"] // 2)


if __name__ == "__main__":
    main()
