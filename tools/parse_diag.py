import json, sys
with open('reports/diag_output.json', 'r', encoding='utf-8-sig') as _fh:
    data = json.load(_fh)
for sym, r in data.get('reports',{}).items():
    print(f'=== {sym} ===')
    for key in ['trades','signal_candidates','orders_submitted','trades_filled','win_rate','total_pnl','equity_end','avg_score']:
        print(f'  {key}: {r.get(key)}')
    print()
    print('  DECISION COUNTS:')
    for k,v in sorted(r.get('decision_counts',{}).items(), key=lambda x:-x[1]):
        print(f'    {k}: {v}')
    print()
    print('  TOP BLOCKERS (top 25):')
    for k,v in sorted(r.get('top_blockers',{}).items(), key=lambda x:-x[1])[:25]:
        print(f'    {k}: {v}')
    print()
    print('  EXEC FAIL:')
    for k,v in sorted(r.get('execution_fail_breakdown',{}).items(), key=lambda x:-x[1]):
        print(f'    {k}: {v}')
    print()
    print('  MISSING FEATURES:')
    for k,v in sorted(r.get('missing_feature_counts',{}).items(), key=lambda x:-x[1]):
        print(f'    {k}: {v}')
    print()
    print('  REJECTED BY REASON:')
    for k,v in sorted(r.get('rejected_by_reason',{}).items(), key=lambda x:-x[1]):
        print(f'    {k}: {v}')
    print()
    print('  SCORE BINS:')
    for k,v in sorted(r.get('count_score_bins',{}).items(), key=lambda x:-x[1]):
        print(f'    {k}: {v}')
    print()
    print('  SPREAD GATE ADJ:')
    for k,v in r.get('spread_gate_adjustments',{}).items():
        print(f'    {k}: {v}')
    print()
    for key in ['min_size_overrides_count','margin_capped_count','forced_closes_count']:
        print(f'  {key}: {r.get(key, 0)}')
