"""Walk-forward player projections 2024-26 for the King of the Court backtest (about 40 s)."""
import pandas as pd, numpy as np, time
from nbamodel import players
t=time.time()
x=players.walk([2023,2024,2025,2026])
x=x[x.season>=2024]
p=players.project(x)
keep=['season','date','game_id','team_id','athlete_id','p_minutes','p_points','p_rebounds','p_assists','avg_points','avg_rebounds','avg_assists','a_points','a_rebounds','a_assists','a_minutes']
p[keep].to_parquet('data/kotc_walk.parquet')
print(len(p), time.time()-t)
