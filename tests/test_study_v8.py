"""v8 registry, immutable-control provenance and causal diagnostic gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rlasset import study_v8 as s


def test_registry_and_complete_selection():
    assert s.CANDIDATES == ('lognet_raw', 'lognet_scaled', 'drawdown_raw', 'drawdown_scaled')
    assert s.NEW_CANDIDATES == ('lognet_scaled', 'drawdown_scaled')
    assert s.SEEDS == (7,17,29)
    rows = {n:[{'total_return':.1,'max_drawdown':.1}]*3 for n in s.CANDIDATES}
    rules = {n:{'total_return':.09,'max_drawdown':.01} for n in s.BASELINES}
    result=s.choose_candidates(rows,rules)
    assert result['candidate']=='lognet_raw'
    assert result['deploy_candidate']=='cash'
    assert result['rule_replicates']==1
    assert len(result['deploy_scores'])==10
    with pytest.raises(ValueError,match='mandatory'): s.choose_candidates({},rules)
    rows['lognet_raw']=rows['lognet_raw'][:2]
    with pytest.raises(ValueError,match='seeds'): s.choose_candidates(rows,rules)


def test_configuration_only_count_changes():
    assert s.PPO_CONFIG == s.old.PPO_CONFIG
    assert s.PPO_CONFIG is not s.old.PPO_CONFIG
    assert s.PPO_CONFIG['ent_coef']==0
    for objective in ('lognet','drawdown'):
        raw=s.candidate_config(objective+'_raw')
        scaled=s.candidate_config(objective+'_scaled')
        assert raw=={**scaled,'count_scale':1}
        assert scaled['count_scale']==20
    with pytest.raises(ValueError): s.candidate_config('lognet_alltrue')


def test_budget_and_test_freeze_gate(tmp_path):
    with pytest.raises(ValueError,match='20000'): s.identity(tmp_path,tmp_path,100)
    with pytest.raises(ValueError,match='Freeze all'): s.evaluate(tmp_path,tmp_path,tmp_path,20000,5)
    with pytest.raises(ValueError,match='cost'): s.evaluate(tmp_path,tmp_path,tmp_path,20000,9)
    assert s.status(tmp_path,tmp_path,tmp_path,20000)['new_expected']==24


def test_baselines_only_read_validation(tmp_path,monkeypatch):
    fold={'name':'fold_01','validation':[120,130],'test':[200,210]}
    monkeypatch.setattr(s,'checked_manifest',lambda *args:{'folds':[fold]})
    monkeypatch.setattr(s,'load_inputs',lambda *args:(None,None))
    seen=[]
    def env(panel,actions,interval,candidate):
        seen.append(interval)
        assert interval==fold['validation']
        return object()
    monkeypatch.setattr(s,'make_env',env)
    monkeypatch.setattr(s,'run_policy',lambda *args:(pd.DataFrame({'fixture':[1]}),{'total_return':0.}))
    monkeypatch.setattr(s,'audit_saved_ledger',lambda *args:None)
    (tmp_path/'manifest.json').write_text('{}')
    result=s.baseline_stage(tmp_path,tmp_path,tmp_path,20000)
    assert len(seen)==6
    s.check_record_signature(json.loads((tmp_path/'baseline_preview/summary.json').read_text()))


def test_control_runtime_checks_fail_closed(tmp_path,monkeypatch):
    from zipfile import ZipFile
    runtime={'Python':'3.12','Stable-Baselines3':'2.7.1','PyTorch':'2.14','GPU Enabled':'False',
             'Numpy':'2.3','Cloudpickle':'3.1','Gymnasium':'1.2','sb3-contrib':'2.7.1'}
    monkeypatch.setattr(s,'runtime_info',lambda:runtime)
    path=tmp_path/'model.zip'
    def write(numpy):
        with ZipFile(path,'w') as z:
            text=''.join(f'- {k}: {numpy if k=="Numpy" else v}\n' for k,v in runtime.items())
            z.writestr('system_info.txt',text)
            z.writestr('_stable_baselines3_version','2.7.1')
    write('2.3')
    assert s.check_checkpoint_runtime(path)['verified_runtime']==runtime
    write('1.0')
    with pytest.raises(ValueError,match='Numpy'): s.check_checkpoint_runtime(path)


def test_diagnostic_excludes_ignored_days_and_common_state_path():
    import torch
    from test_masked_selector_v7 import panel_fixture
    torch.set_num_threads(1)
    panel=panel_fixture()
    raw=s.make_env(panel,None,[120,150],'lognet_raw')
    scaled=s.make_env(panel,None,[120,150],'lognet_scaled')
    a=s.make_agent(raw,7)
    b=s.make_agent(scaled,7)
    first,metrics,diag=s.run_model_diagnostics(raw,a,fixed_equal50=True)
    second,other,diag2=s.run_model_diagnostics(scaled,b,fixed_equal50=True)
    assert diag['all_decisions']['states']==6
    assert diag['mature_decisions']['states']==2
    assert diag['state_path']=='fixed_equal50'
    assert diag['phase_zero_only']
    assert metrics==other
    np.testing.assert_array_equal(first.net_return,second.net_return)
    assert diag2['mature_decisions']['states']==2
    assert 0 <= diag['all_decisions']['mean_first_tanh_saturation'] <= 1
    assert 0 <= diag['all_decisions']['mean_entropy_nats'] <= np.log(4)+1e-6


def test_raw_matches_existing_v7_observations_and_masks():
    from test_masked_selector_v7 import panel_fixture
    panel=panel_fixture()
    old=s.old.make_env(panel,None,[120,160],'lognet_phase')
    raw=s.make_env(panel,None,[120,160],'lognet_raw')
    np.testing.assert_array_equal(old.reset(seed=7)[0],raw.reset(seed=7)[0])
    for i in range(40):
        a=old.step(i%4); b=raw.step(i%4)
        np.testing.assert_array_equal(a[0],b[0])
        assert a[1:]==b[1:]


def test_no_mutation_of_old_registry_or_sources():
    assert s.old.CANDIDATES==('lognet_alltrue','lognet_phase','drawdown_alltrue','drawdown_phase')
    source=Path(s.__file__).read_text()
    assert 'old.CANDIDATES =' not in source
    for dep in ('scaled_selector_v8.py','research-protocol-v8.md','test_scaled_selector_v8.py', 'source_control_identity'):
        assert dep in source


def test_record_seal_detects_diagnostics_change():
    record=s.signed_record({'validation_diagnostics':{'mean_entropy':.2}})
    s.check_record_signature(record)
    record['validation_diagnostics']['mean_entropy']=1.
    with pytest.raises(ValueError,match='record'): s.check_record_signature(record)


def test_control_record_parity_fails_before_any_record_write(tmp_path,monkeypatch):
    from types import SimpleNamespace
    ledger=pd.DataFrame({'net_return':[0.,.1]})
    metrics={'total_return':.1,'max_drawdown':0.}
    monkeypatch.setattr(s,'run_model_diagnostics',lambda *args,**kwargs:(ledger,metrics,{'probe':True}))
    monkeypatch.setattr(s,'runtime_info',lambda:{'runtime':'fixture'})
    monkeypatch.setattr(s,'audit_saved_ledger',lambda *args:None)
    (tmp_path/'lognet_raw.zip').write_bytes(b'checkpoint')
    original=tmp_path/'original.csv'
    pd.DataFrame({'net_return':[0.,.2]}).to_csv(original,index=False)
    source={'validation_metrics':metrics,'episode_schedule':[[120,246]],'training_seconds':1.}
    model=SimpleNamespace(num_timesteps=20000)
    fold={'name':'fold_01'}
    args=(tmp_path,'lognet_raw',7,fold,model,object(),tmp_path,20000,'reused_v7_control',0.,source,original)
    with pytest.raises(ValueError,match='replay changed'):
        s.write_model_record(*args)
    assert not (tmp_path/'lognet_raw.json').exists()
    assert not (tmp_path/'validation_lognet_raw.csv').exists()
    ledger.to_csv(original,index=False)
    s.write_model_record(*args)
    record=json.loads((tmp_path/'lognet_raw.json').read_text())
    s.check_record_signature(record)
    assert record['training_seconds']==0
    assert record['source_training_seconds']==1
    assert record['kind']=='reused_v7_control'
    assert record['common_state_diagnostics']=={'probe':True}


def test_checked_control_rejects_model_hash_and_source_linkage_changes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    runtime={'runtime':'fixture'}
    metrics={'total_return':.1,'max_drawdown':.01}
    monkeypatch.setattr(s,'runtime_info',lambda:runtime)
    monkeypatch.setattr(s,'validation_metrics',lambda *args:metrics)
    model=tmp_path/'lognet_raw.zip'; model.write_bytes(b'model')
    validation=tmp_path/'validation_lognet_raw.csv'; validation.write_text('fixture')
    fold={'name':'fold_01','train':[120,500],'validation':[0,2]}
    schedule=s.old.episode_schedule(fold['train'],7,20000)
    source=s.signed_record({'candidate':'lognet_phase','model_sha256':s.digest(model),
                            'episode_schedule':schedule,'validation_metrics':metrics})
    record={'candidate':'lognet_raw','seed':7,'fold':'fold_01','actual_steps':20000,
            'requested_steps':20000,'config':s.candidate_config('lognet_raw'),
            'ppo_config':s.PPO_CONFIG,'model_sha256':s.digest(model),
            'validation_sha256':s.digest(validation),'validation_metrics':metrics,
            'episode_schedule':schedule,'runtime':runtime,'backend':'sb3_contrib.MaskablePPO',
            'kind':'reused_v7_control','source_record':source}
    path=tmp_path/'lognet_raw.json'
    s.write_json(path,s.signed_record(record))
    panel=SimpleNamespace(dates=pd.date_range('2020-01-01',periods=3))
    assert s.checked_record(tmp_path,'lognet_raw',7,fold,panel,20000)['kind']=='reused_v7_control'
    model.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): s.checked_record(tmp_path,'lognet_raw',7,fold,panel,20000)
    model.write_bytes(b'model')
    record['source_record']=s.signed_record({**source,'candidate':'drawdown_phase'})
    s.write_json(path,s.signed_record(record))
    with pytest.raises(ValueError,match='source record'): s.checked_record(tmp_path,'lognet_raw',7,fold,panel,20000)
