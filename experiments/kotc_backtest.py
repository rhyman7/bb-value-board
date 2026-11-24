"""King of the Court backtest: chance each player leads a night in PRA vs who actually did, on 2024-26
walk-forward projections (experiments/kotc_walk.py writes the input). Picked SD = 0.15*mu + 6.5
(best calibrated: top pick given 26%, won 28%, 475 nights). Monte Carlo noise in log loss is ~0.01."""
import pandas as pd, numpy as np
p=pd.read_parquet('data/kotc_walk.parquet')
p=p[(p.p_minutes>=8)].copy()
p['mu']=p.p_points+p.p_rebounds+p.p_assists
p['a']=(p.a_points+p.a_rebounds+p.a_assists)
p=p.dropna(subset=['a'])
rng=np.random.default_rng(0)
def probs(mu,sd,gi,ng,N=20000,gs=0.06):
    d=mu*(1+rng.normal(0,gs,(N,ng))[:,gi])+rng.normal(0,1,(N,len(mu)))*sd
    d=np.round(d); mx=d.max(1,keepdims=True); w=(d==mx); return (w/w.sum(1,keepdims=True)).mean(0)
V={'A 0.15mu+5':lambda m:0.15*m+5,'E 0.15mu+6.5':lambda m:0.15*m+6.5,'F 0.2mu+5':lambda m:0.2*m+5,'G 0.25mu+5':lambda m:0.25*m+5}
res={k:[] for k in V}; top=[]
for (s,d),n in p.groupby(['season','date']):
    n=n[n.mu>=18]
    if n.game_id.nunique()<2 or len(n)<3: continue
    act=n.a.to_numpy(); win=(act==act.max()); wshare=win/win.sum()
    gi=pd.factorize(n.game_id)[0]; ng=gi.max()+1
    for k,f in V.items():
        pr=probs(n.mu.to_numpy(),f(n.mu.to_numpy()),gi,ng)
        res[k].append(-(wshare*np.log(np.maximum(pr,1e-4))).sum())
        if k.startswith('E'):
            i=pr.argmax(); top.append((pr[i],wshare[i],n.game_id.nunique()))
print('nights',len(top))
for k in V: print(k,'logloss',np.mean(res[k]).round(4))
t=pd.DataFrame(top,columns=['p','w','ng'])
t['b']=pd.cut(t.p,[0,.2,.3,.4,.5,.7,1])
print(t.groupby('b',observed=True).agg(n=('p','size'),pred=('p','mean'),hit=('w','mean')).round(3))
print('overall top pick pred',t.p.mean().round(3),'hit',t.w.mean().round(3))
