"""Run one scenario and print the metrics and the event log.   usage: python run_single.py [seed]"""
import json
import sys
from sim import Sim

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
s = Sim(seed)
print(json.dumps(s.run(), indent=1, default=str))
for t, msg in s.logs:
    print(f"{t:7.1f}  {msg}")
