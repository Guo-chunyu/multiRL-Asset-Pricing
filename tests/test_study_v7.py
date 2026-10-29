"""Registry, certification, preview isolation, freeze and resume invariants."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rlasset import study_v7 as s


def test_registry_and_selection_require_complete_family():
    assert s.CANDIDATES == ('lognet_alltrue', 'lognet_phase', 'drawdown_alltrue', 'drawdown_phase')
    assert s.SEEDS == (7, 17, 29)
    assert s.BASELINES == ('cash', 'equal50', 'vol10', 'trend10', 'equal_target_mix', 'trailing20')
    rows = {name: [{'total_return': .1, 'max_drawdown': .1}] * 3 for name in s.CANDIDATES}
    rules = {name: {'total_return': .09, 'max_drawdown': .01} for name in s.RULE_CANDIDATES}
    result = s.choose_candidates(rows, rules)
    assert result['candidate'] == 'lognet_alltrue'  # Fixed tie order.
    assert result['deploy_candidate'] == 'cash'
    assert result['selected_on'] == 'validation'
    with pytest.raises(ValueError, match='mandatory'):
        s.choose_candidates({}, rules)
    rows['lognet_alltrue'] = rows['lognet_alltrue'][:2]
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
    record = s.signed_record({'candidate':'lognet_alltrue','validation_metrics':{'total_return':.1}})
    s.check_record_signature(record)
    record['validation_metrics']['total_return'] = .9
    with pytest.raises(ValueError, match='record'):
        s.check_record_signature(record)
    source = Path(s.__file__).read_text()
    for name in ('selector_env_v6.py', 'study_v4.py', 'statistics.py', 'research-protocol-v7.md', 'audit_research_v3.py', 'restoration_verification.json'):
        assert name in source


def test_rules_are_scored_once_and_fixed_expert_separate():
    rows = {name: [{'total_return': 0., 'max_drawdown': 0.}] * 3 for name in s.CANDIDATES}
    rules = {name: {'total_return': .1, 'max_drawdown': .1} for name in s.BASELINES}
    rules['equal_target_mix']['total_return'] = .2
    result = s.choose_candidates(rows, rules)
    assert result['deploy_candidate'] == 'equal_target_mix'
    assert result['best_validation_fixed_expert'] == 'cash'
    assert result['rule_replicates'] == 1
    assert len(result['deploy_scores']) == 10


def test_policy_dispatch_uses_discrete_and_single_account_targets():
    env = SimpleNamespace(fixed_expert_action=lambda name:s.EXPERTS.index(name),
                          equal_target_mix=lambda:np.array([.5,.5]), trailing_argmax=lambda:3)
    assert s.baseline_policy('cash')(env,None) == 0
    assert s.baseline_policy('trailing20')(env,None) == 3
    np.testing.assert_array_equal(s.baseline_policy('equal_target_mix')(env,None), [.5,.5])
    with pytest.raises(ValueError): s.baseline_policy('unknown')
    with pytest.raises(ValueError): s.candidate_config('unknown')


def test_registered_budget_is_exact():
    from rlasset.finrl_adapter import training_budget
    budget = training_budget('ppo', 20000)
    assert budget['expected_actual_steps'] == 20000
    assert budget['rollout_steps'] == 100


def test_actual_effective_v6_config_and_maskable_pair():
    from test_selector_env_v6 import panel_fixture
    from rlasset.finrl_adapter import make_agent
    panel = panel_fixture()
    envs = [s.make_env(panel, None, [120, 280], name, training=True) for name in s.CANDIDATES]
    original = make_agent(envs[0], algorithm='ppo', seed=7, total_timesteps=20000,
                          learning_rate=1e-4, policy='MlpPolicy')
    models = [s.make_maskable_agent(env, 7) for env in envs]
    for key in ('learning_rate','n_steps','batch_size','n_epochs','gamma','gae_lambda',
                'ent_coef','vf_coef','max_grad_norm','target_kl','normalize_advantage','policy_kwargs'):
        assert all(getattr(model,key) == getattr(original,key) for model in models)
    assert original.ent_coef == 0.0
    assert all(model.clip_range(1) == original.clip_range(1) for model in models)
    assert all(model.clip_range_vf is original.clip_range_vf is None for model in models)
    assert len({type(model) for model in models}) == 1
    assert [env.drawdown_penalty for env in envs] == [False,False,True,True]
    assert [env.mask_mode for env in envs] == ['control','phase','control','phase']


def test_predict_receives_current_mask():
    seen=[]
    class Model:
        def predict(self, obs, **kwargs):
            seen.append(kwargs)
            return 0,None
    env=SimpleNamespace(action_masks=lambda:np.array([True,False,False,False]))
    assert s.model_policy(Model())(env,None)==0
    assert seen[0]['deterministic'] is True
    np.testing.assert_array_equal(seen[0]['action_masks'],[True,False,False,False])


def test_expected_episode_schedule_matches_independent_rng():
    schedule=s.episode_schedule([120,500],7,20000)
    assert len(schedule)==159
    assert all(end-start==126 for start,end in schedule)
    assert schedule==s.episode_schedule([120,500],7,20000)
    assert schedule!=s.episode_schedule([120,500],17,20000)


def test_executed_counts_ignore_dummy_requests():
    frame=pd.DataFrame({'rebalance_executed':[False,True,False,True,True],
                        'executed_expert':[None,'equal50',None,'equal50','cash'],
                        'requested_expert':[None,'equal50','cash','equal50','cash'],
                        'cost_fraction':[0,.001,0,.001,.001]})
    metrics=s.execution_metrics(frame)
    assert metrics['executed_rebalances']==3
    assert metrics['executed_expert_switches']==1
    assert metrics['executed_fraction_cash']==1/3
    assert metrics['sum_daily_cost_fraction']==.003
