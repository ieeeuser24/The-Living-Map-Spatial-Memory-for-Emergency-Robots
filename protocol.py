"""20-byte beacon record, LoRa link model, frame translation and confidence aging."""
import math
import struct
import numpy as np

FMT = "<BHHBhhBIBBBH"          # 20 bytes, little endian, no padding
assert struct.calcsize(FMT) == 20
EVENTS = {0: "relay", 1: "gas", 2: "collapse", 3: "victim", 4: "junction", 5: "dead end"}
TAU = {0: 6 * 3600.0, 1: 900.0, 2: 6 * 3600.0, 3: 3600.0, 4: 6 * 3600.0, 5: 6 * 3600.0}   # aging constants (s)
PRIORITY = {0: 0, 1: 2, 2: 2, 3: 3, 4: 1, 5: 1}


def crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def pack(beacon_id, prev_id, event, x, y, bearing_deg, ts, conf, hop):
    body = struct.pack(FMT[:-1], 1, beacon_id, prev_id, event,
                       int(round(x * 10)), int(round(y * 10)),
                       int(round((bearing_deg % 360) / 2)) % 180, int(ts),
                       int(max(0, min(255, round(conf * 255)))), hop,
                       (PRIORITY[event] & 3) << 6 | event)
    return body + struct.pack("<H", crc16(body))


def unpack(raw):
    if len(raw) != 20 or crc16(raw[:-2]) != struct.unpack("<H", raw[-2:])[0]:
        return None
    v, bid, pid, ev, x, y, hd, ts, cf, hop, pa, _ = struct.unpack(FMT, raw)
    return dict(version=v, beacon_id=bid, prev_id=pid, event=ev, x=x / 10.0, y=y / 10.0,
                bearing=hd * 2.0, ts=ts, conf=cf / 255.0, hop=hop, priority=pa >> 6)


# ------------------------------------------------------------------ LoRa link
TX_DBM = 14.0


def rssi(dist, los, rng, sigma=2.0):
    """Tunnel path loss at 868 MHz: low exponent in line of sight, extra corner loss and steeper slope otherwise."""
    d = max(1.0, dist)
    pl = 35 + 20 * math.log10(d) if los else 35 + 25 + 35 * math.log10(d)
    return TX_DBM - pl + (rng.normal(0, sigma) if rng is not None else 0.0)


def p_success(r):
    if r >= -110:
        return 0.98
    if r <= -120:
        return 0.0
    return 0.98 * (r + 120) / 10.0


# ---------------------------------------------------------- frame translation
R_EARTH = 6378137.0


class Frame:
    """Entrance-anchored local frame -> GPS (flat-earth ENU approximation)."""

    def __init__(self, lat0, lon0, theta0_deg):
        self.lat0, self.lon0, self.th = lat0, lon0, math.radians(theta0_deg)

    def enu(self, x, y):
        e = x * math.sin(self.th) - y * math.cos(self.th)
        n = x * math.cos(self.th) + y * math.sin(self.th)
        return e, n

    def to_gps(self, x, y):
        e, n = self.enu(x, y)
        return (self.lat0 + math.degrees(n / R_EARTH),
                self.lon0 + math.degrees(e / (R_EARTH * math.cos(math.radians(self.lat0)))))

    @staticmethod
    def gps_diff_m(a, b):
        dn = math.radians(a[0] - b[0]) * R_EARTH
        de = math.radians(a[1] - b[1]) * R_EARTH * math.cos(math.radians(a[0]))
        return math.hypot(de, dn)


def confidence(c0, t0, t, etype):
    return c0 * math.exp(-(t - t0) / TAU[etype])
