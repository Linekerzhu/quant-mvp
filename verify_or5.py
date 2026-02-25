import pandas as pd, numpy as np
from src.labels.triple_barrier import TripleBarrierLabeler

lab = TripleBarrierLabeler()
dates = pd.date_range('2024-01-01', periods=50, freq='B')

def make_df(adj_open, adj_high, adj_low, adj_close):
    n = len(adj_open)
    return pd.DataFrame({
        'symbol': 'T',
        'date': dates[:n],
        'adj_open': adj_open,
        'adj_close': adj_close,
        'adj_high': adj_high,
        'adj_low': adj_low,
        'raw_open': adj_open,
        'raw_high': adj_high,
        'raw_low': adj_low,
        'raw_close': adj_close,
        'volume': [5000]*n,
        'atr_20': [2.0]*n,
        'features_valid': True,
        'can_trade': True
    })

n = 20
results = []

# Test 1: Loss Gap
o = [100.0]*n; h = [101.0]*n; l = [99.0]*n; c = [100.0]*n
o[2]=94.0; h[2]=95.0; l[2]=93.0; c[2]=94.5
r = lab.label_events(make_df(o, h, l, c))
ev = r[r['event_valid']==True].iloc[0]
ok = ev['label_barrier']=='loss_gap' and abs(ev['label_return']-np.log(94/100))<1e-6
results.append(('Loss Gap', ok))

# Test 2: Profit Gap
o = [100.0]*n; h = [101.0]*n; l = [99.0]*n; c = [100.0]*n
o[2]=107.0; h[2]=108.0; l[2]=106.0; c[2]=107.5
r = lab.label_events(make_df(o, h, l, c))
ev = r[r['event_valid']==True].iloc[0]
ok = ev['label_barrier']=='profit_gap' and abs(ev['label_return']-np.log(107/100))<1e-6
results.append(('Profit Gap', ok))

# Test 3: Collision
o = [100.0]*n; h = [101.0]*n; l = [99.0]*n; c = [100.0]*n
o[2]=100.0; h[2]=105.0; l[2]=95.0; c[2]=101.0
r = lab.label_events(make_df(o, h, l, c))
ev = r[r['event_valid']==True].iloc[0]
ok = ev['label_barrier']=='loss_collision' and ev['label']==-1
results.append(('Collision→Loss', ok))

# Test 4: Normal profit
o = [100.0]*n; h = [101.0]*n; l = [99.0]*n; c = [100.0]*n
o[2]=101.0; h[2]=105.0; l[2]=100.5; c[2]=104.0
r = lab.label_events(make_df(o, h, l, c))
ev = r[r['event_valid']==True].iloc[0]
ok = ev['label_barrier']=='profit'
results.append(('Normal Profit', ok))

# Test 5: Normal loss
o = [100.0]*n; h = [101.0]*n; l = [99.0]*n; c = [100.0]*n
o[2]=99.0; h[2]=99.5; l[2]=95.0; c[2]=95.5
r = lab.label_events(make_df(o, h, l, c))
ev = r[r['event_valid']==True].iloc[0]
ok = ev['label_barrier']=='loss'
results.append(('Normal Loss', ok))

# Report
print('='*50)
for name, passed in results:
    print(f" {'✅' if passed else '❌'} {name}")
all_ok = all(p for _,p in results)
print('='*50)
print(f" RESULT: {'ALL PASSED ✅' if all_ok else 'SOME FAILED ❌'}")
