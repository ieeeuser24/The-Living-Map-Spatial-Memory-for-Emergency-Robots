# The Living Map: simulation (TSYP14, Phase 1, tunnel / mines scenario)

A self-contained Python 2D simulation of the two-robot system: a **Writer** robot explores a GPS-denied tunnel and
drops LoRa **beacons**; the beacons relay 20-byte records to an entrance **gateway**; a **command post** shows a live
GPS map and briefs an **Executor** robot, which follows the beacon chain after the Writer has failed.

## Quick start
```bash
pip install -r requirements.txt
python run_single.py 1          # one run: metrics + event log
python run_demo.py 1 demo.mp4   # render the demo video (1280x720, 25 fps, about 50 s)
python run_batch.py 60          # Monte Carlo: 60 nominal + 60 stress runs -> results.json
```

## Files
| File | Content |
|---|---|
| `world.py` | Tunnel map (occupancy grid), LiDAR ray casting, gas field, thermal sources, line of sight |
| `protocol.py` | 20-byte record (pack / unpack, CRC-16), LoRa RSSI model, frame translation to GPS, confidence aging |
| `network.py` | Beacons, multi-hop store-and-forward relay with ACK/retries and re-routing, gateway, command post, signed briefing |
| `robots.py` | Writer (frontier exploration, event detection, beacon-drop policy, state machine) and Executor (A*, pose reset, verification) |
| `sim.py` | Scenario runner, fault injection, metrics |
| `run_single.py`, `run_batch.py`, `run_demo.py` | Entry points |

## Scenario
Main tunnel of 125 m (entrance at x = 0), junction J1 with a 32 m dead-end branch, junction J2 with a branch that has a
90 degree bend, a gas pocket (x = 62 m), a collapse (x = 110 m), a trapped victim at the end of the J2 branch and a
small hot machine (decoy for the thermal camera). Robots move at 1 m/s, time step 0.2 s.

## What is modeled
- **Writer:** 2D LiDAR (120 rays, 10 m), frontier exploration with BFS, odometry drift of about 1.2 % of path length,
  pose reset to the beacon's recorded position within 3 m (BLE proximity), gas / thermal / collapse detection with
  multi-frame confidence (confirm at c >= 0.6), state machine EXPLORE / INSPECT / DROP / RETURN / STOP.
- **Beacon-drop policy:** event confirmed, junction or bend, dead end, or link margin low (RSSI < -105 dBm or more than
  25 m from the chain); 10 beacons, at least 4 reserved for events.
- **Radio:** 868 MHz LoRa, 14 dBm, line-of-sight and tunnel-path (corner) path-loss model, shadowing, packet success
  from RSSI, ACK with 3 retries and re-routing, hop-by-hop relay toward the gateway.
- **Record:** 20 bytes (version, beacon id, previous id, event, x, y, bearing, timestamp, confidence, hop, priority, CRC-16).
- **Gateway / command post:** de-duplication, frame translation (entrance GPS fix with 1 m error, bearing with 0.5 degree
  error), uplink with store-and-forward during outages, confidence aging c(t) = c0 exp(-(t - t0)/tau),
  heartbeat timeout, HMAC-signed briefing.
- **Executor:** briefing verification, A* navigation on its own LiDAR map, pose reset at each beacon, gas re-verification
  (refreshes the aged event), thermal verification and approach of the victim, local frontier search if no heat
  signature is seen at the target beacon.
- **Stress scenario:** +15 % packet loss, a 60 s uplink outage, 30 % corrupted briefings, one beacon dead after the
  Writer fails.

## Assumptions and limitations (please read)
- Local mapping is assumed SLAM-consistent: robots plan in the true local frame, while the **global** position estimate
  used for reporting drifts and is corrected at beacons.
- AES-128 link encryption, beacon batteries and duty cycling, BLE signal strength (a fixed 3 m range is used) and
  gradient-following navigation are **not** simulated.
- The Writer is disabled 5 to 60 s after the victim record reaches the command post, so event recall is measured at
  failure time and the collapse is sometimes not yet found.
- Results are simulation outcomes, not hardware measurements; they validate the logic of the architecture.
