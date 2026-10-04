"""Tunnel world: occupancy grid, LiDAR ray casting, gas field and thermal sources."""
import numpy as np
from scipy import ndimage

RES = 0.5                       # grid resolution (m per cell)
X0, Y0 = -6.0, -36.0            # world coordinates of grid corner (m)
NX, NY = int(140 / RES), int(76 / RES)


def w2c(x, y):
    return int((x - X0) // RES), int((y - Y0) // RES)


def c2w(ix, iy):
    return X0 + (ix + 0.5) * RES, Y0 + (iy + 0.5) * RES


class World:
    """Reference scenario: 125 m main tunnel (entrance at x = 0), a 32 m dead-end branch (J1),
    a branch with a 90 degree bend (J2), a gas pocket, a collapse and a trapped victim."""

    GAS_C = np.array([62.0, 0.0])
    GAS_AMP, GAS_SIGMA, GAS_BASE, GAS_NOISE = 45.0, 3.5, 5.0, 1.0
    VICTIM = np.array([106.0, -28.5])
    COLLAPSE_C = np.array([111.5, 0.0])
    DECOY = np.array([20.0, 1.0])          # hot machinery (small heat source)
    THERMAL_RANGE = 8.0

    def __init__(self):
        g = np.ones((NX, NY), np.uint8)

        def carve(x0, x1, y0, y1):
            i0, j0 = w2c(x0, y0)
            i1, j1 = w2c(x1, y1)
            g[i0:i1, j0:j1] = 0

        carve(-5, 125, -1.5, 1.5)        # main tunnel
        carve(38.5, 41.5, 0, 32)         # branch A (junction J1, dead end)
        carve(86.5, 89.5, -30, 0)        # branch B (junction J2)
        carve(86.5, 108, -30, -27)       # bend of branch B towards the victim
        self.rubble = np.zeros(g.shape, bool)
        i0, j0 = w2c(110, -1.5)
        i1, j1 = w2c(113, 1.5)
        g[i0:i1, j0:j1] = 1
        self.rubble[i0:i1, j0:j1] = True
        self.grid = g
        lab, _ = ndimage.label(g == 0)
        ci, cj = w2c(0, 0)
        self.reachable = lab == lab[ci, cj]
        self.reachable_count = int(self.reachable.sum())

    # ---------------------------------------------------------------- geometry
    def is_free(self, x, y):
        i, j = w2c(x, y)
        return 0 <= i < NX and 0 <= j < NY and self.grid[i, j] == 0

    def los(self, p, q, step=0.25):
        d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        n = max(2, int(d / step))
        xs = np.linspace(p[0], q[0], n)
        ys = np.linspace(p[1], q[1], n)
        ix = ((xs - X0) // RES).astype(int)
        iy = ((ys - Y0) // RES).astype(int)
        ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        if not ok.all():
            return False
        return not self.grid[ix, iy].any()

    def lidar(self, pos, theta, n_rays=120, max_r=10.0, step=0.5):
        """Returns ranges, rubble flags per ray, free cells and hit cells (index arrays)."""
        ang = theta + np.arange(n_rays) * (2 * np.pi / n_rays)
        rs = np.arange(step, max_r + step, step)
        cx = pos[0] + np.outer(np.cos(ang), rs)
        cy = pos[1] + np.outer(np.sin(ang), rs)
        ix = ((cx - X0) // RES).astype(int)
        iy = ((cy - Y0) // RES).astype(int)
        inb = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        ixc, iyc = np.clip(ix, 0, NX - 1), np.clip(iy, 0, NY - 1)
        occ = (self.grid[ixc, iyc] == 1) | ~inb
        hit = occ.any(axis=1)
        first = np.where(hit, occ.argmax(axis=1), len(rs))
        ranges = np.where(hit, rs[np.minimum(first, len(rs) - 1)], max_r)
        k = np.arange(len(rs))[None, :]
        free_mask = (k < first[:, None]) & inb
        a = np.nonzero(hit)[0]
        hit_ix, hit_iy = ixc[a, first[a]], iyc[a, first[a]]
        rub = np.zeros(n_rays, bool)
        rub[a] = self.rubble[hit_ix, hit_iy]
        return ranges, rub, (ixc[free_mask], iyc[free_mask]), (hit_ix, hit_iy)

    def geo_dist(self, p, q):
        """Shortest distance along the tunnel network (m); used for non-line-of-sight radio paths."""
        if not hasattr(self, "_geo"):
            self._geo = {}
        s = w2c(*p)
        if s not in self._geo:
            dist = {s: 0}
            frontier = [s]
            while frontier:
                nxt = []
                for c in frontier:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        n = (c[0] + dx, c[1] + dy)
                        if 0 <= n[0] < NX and 0 <= n[1] < NY and self.grid[n] == 0 and n not in dist:
                            dist[n] = dist[c] + 1
                            nxt.append(n)
                frontier = nxt
            self._geo[s] = dist
        return self._geo[s].get(w2c(*q), 400) * RES

    # ----------------------------------------------------------------- sensors
    def gas_read(self, pos, rng):
        d2 = float(np.sum((np.asarray(pos) - self.GAS_C) ** 2))
        return self.GAS_BASE + self.GAS_AMP * np.exp(-d2 / (2 * self.GAS_SIGMA ** 2)) + rng.normal(0, self.GAS_NOISE)

    def thermal(self, pos, rng, p_detect=0.9, p_misclass=0.03):
        """Returns list of (class, measured position). 'human' blobs are candidate victims."""
        out = []
        if np.hypot(*(self.VICTIM - pos)) <= self.THERMAL_RANGE and self.los(pos, self.VICTIM) and rng.random() < p_detect:
            out.append(("human", self.VICTIM + rng.normal(0, 0.3, 2)))
        if np.hypot(*(self.DECOY - pos)) <= self.THERMAL_RANGE and self.los(pos, self.DECOY) and rng.random() < p_detect:
            out.append(("human" if rng.random() < p_misclass else "small", self.DECOY))
        return out

    def truth_events(self):
        return {"gas": self.GAS_C, "collapse": self.COLLAPSE_C, "victim": self.VICTIM}
