"""Registry, certification, preview isolation, freeze and resume invariants."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rlasset import study_v5 as s


def test_registry_and_selection_require_complete_family():
    assert s.CANDIDATES == ('v4risk_ppo', 'residual_vol_ppo', 'residual_trend_ppo')
    assert s.SEEDS == (7, 17, 29)
    assert s.BASELINES == ('equal_50', 'equal_100', 'cash', 'vol_target10', 'trend_target10')
    rows = {name: [{'total_return': .1, 'max_drawdown': .1}] * 3 for name in s.CANDIDATES}
    rules = {name: {'total_return': .09, 'max_drawdown': .01} for name in s.RULE_CANDIDATES}
    result = s.choose_candidates(rows, rules)
    assert result['candidate'] == 'v4risk_ppo'  # Fixed tie order.
    assert result['deploy_candidate'] == 'vol_target10'
    assert result['selected_on'] == 'validation'
    with pytest.raises(ValueError, match='mandatory'):
        s.choose_candidates({}, rules)
    rows['v4risk_ppo'] = rows['v4risk_ppo'][:2]
    with pytest.raises(ValueError, match='seed'):
        s.choose_candidates(rows, rules)


def test_status_missing_and_uncertified_data(tmp_path):
    assert s.status(tmp_path, tmp_path/'run', 20000)['state'] == 'blocked_data'
    for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv'):
        (tmp_path/name).write_text('not market data')
    state = s.status(tmp_path, tmp_path/'run', 20000)
    assert state['missing_inputs'] == []
    assert state['state'] == 'blocked_data'
    assert 'verification' in state['certification_blocker']


def test_certification_fails_closed_and_detects_changed_data(tmp_path):
    names = ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv')
    for name in names:
        (tmp_path/name).write_text(name)
    record = {'certified_ready': False, 'input_sha256': {name: s.digest(tmp_path/name) for name in names}}
    path = tmp_path/'restoration_verification.json'
    s.write_json(path, record)
    with pytest.raises(ValueError, match='not certified'):
        s.require_certified_data(tmp_path)
    record['certified_ready'] = True
    s.write_json(path, record)
    assert s.require_certified_data(tmp_path) == record
    (tmp_path/names[0]).write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        s.require_certified_data(tmp_path)


def test_formal_budget_and_freeze_gate(tmp_path):
    with pytest.raises(ValueError, match='20000'):
        s.identity(tmp_path, 100)
    with pytest.raises(ValueError, match='Freeze all'):
        s.evaluate(tmp_path, tmp_path, 20000, 5)
    with pytest.raises(ValueError, match='cost'):
        s.evaluate(tmp_path, tmp_path, 20000, 9)


def test_baseline_preview_never_reads_test(tmp_path, monkeypatch):
    fold = {'name': 'fold_01', 'validation': [120,130], 'test': [200,210]}
    monkeypatch.setattr(s, 'checked_manifest', lambda *args: {'folds':[fold]})
    monkeypatch.setattr(s, 'load_inputs', lambda *args: (None,None))
    seen=[]
    def env(panel, actions, interval, candidate):
        seen.append(interval)
        assert interval == fold['validation']
        return object()
    monkeypatch.setattr(s, 'make_env', env)
    monkeypatch.setattr(s, 'run_policy', lambda *args: (pd.DataFrame({'fixture':[1]}), {'total_return':0.}))
    monkeypatch.setattr(s, 'audit_saved_ledger', lambda *args: None)
    (tmp_path/'manifest.json').write_text('{}')
    result = s.baseline_stage(tmp_path, tmp_path, 20000)
    assert len(seen) == len(s.BASELINES)
    assert list(result['folds']['fold_01']) == ['validation']
    s.check_record_signature(json.loads((tmp_path/'baseline_preview/summary.json').read_text()))


def test_record_seal_and_hash_identity_dependencies():
    record = s.signed_record({'candidate':'residual_vol_ppo','validation_metrics':{'total_return':.1}})
    s.check_record_signature(record)
    record['validation_metrics']['total_return'] = .9
    with pytest.raises(ValueError, match='record'):
        s.check_record_signature(record)
    source = Path(s.__file__).read_text()
    for name in ('residual_env.py', 'study_v4.py', 'statistics.py', 'research-protocol-v5.md', 'audit_research_v3.py', 'restoration_verification.json'):
        assert name in source


def test_baseline_policies_are_explicit_zero_anchors():
    from rlasset.residual_env import trend_volatility_target
    close=np.exp(np.arange(160)[:,None]*np.array([[.001,.002,.003]]))
    env=SimpleNamespace(n_assets=3,signal_close=close,t=150)
    for name, gated in [('vol_target10',False),('trend_target10',True)]:
        np.testing.assert_array_equal(s.baseline_policy(name)(env,None),trend_volatility_target(close,150,gated))
    np.testing.assert_array_equal(s.baseline_policy('cash')(env,None),[1,0,0,0])
    with pytest.raises(ValueError,match='candidate'):
        s.candidate_config('new_unregistered_ppo')
