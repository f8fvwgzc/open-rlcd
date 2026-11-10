"""Write the synthetic labelled tickets the example trains on: `python make_tickets.py [--n 3000] [--seed 7]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from ticketpack.synth import write

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).with_name("data") / "tickets.jsonl"))
    a = ap.parse_args()
    info = write(a.out, n=a.n, seed=a.seed)
    print(f"wrote {info['rows']} tickets to {info['path']}")
    print(f"  queues: {info['queues']}")
    print(f"  urgent: {info['urgent']}   languages: {info['languages']}")
