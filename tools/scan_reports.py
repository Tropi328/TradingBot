import json, os
for fn in sorted(os.listdir('reports')):
    if fn.endswith('.json') and 'XAUUSD' in fn and not fn.startswith('daily'):
        try:
            with open(f'reports/{fn}', 'r', encoding='utf-8') as _fh:
                r = json.load(_fh)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        t = r.get('trades', 0)
        sc = r.get('signal_candidates', 0)
        os_val = r.get('orders_submitted', 0)
        print(f'{fn}: trades={t}, sig_cand={sc}, orders={os_val}, avg_score={r.get("avg_score","?")}, spread_mode={r.get("spread_mode","?")}')
