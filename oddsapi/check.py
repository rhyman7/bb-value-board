"""Free check: the key works, the credit balance, and that both basketball sports are listed.
Then one historical events call (1 credit) to confirm history is on this plan.

    python -m oddsapi.check
"""
from . import client

if __name__ == "__main__":
    sports, h = client.get("/sports", all="true")
    keys = {s["key"] for s in sports}
    print("basketball_nba listed:", "basketball_nba" in keys, "| basketball_ncaab listed:", "basketball_ncaab" in keys)
    print("remaining:", h.get("x-requests-remaining"), "used:", h.get("x-requests-used"))
    try:
        ev, _ = client.get("/historical/sports/basketball_nba/events", date="2025-01-15T23:00:00Z")
        print("historical events ok:", len(ev.get("data", [])), "NBA events at 2025-01-15 23:00Z; snapshot", ev.get("timestamp"))
    except Exception as e:
        print("historical events failed:", repr(e))
    print("credits used by this check:", client.used_total)
