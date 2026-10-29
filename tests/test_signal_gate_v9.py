from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from rlasset import signal_gate_v9 as s
from test_masked_selector_v7 import panel_fixture


def test_ranks_weights_and_ticker_ties():
    close=np.ones((25,3)); close[20]=[1.3,1.1,1.2]
    np.testing.assert_array_equal(s.relative_momentum50(close,20,('A','B','C')), [.5,.25,1/12,1/6])
    close[20]=1
    np.testing.assert_array_equal(s.relative_momentum50(close,20,('C','A','B')), [.5,1/12,.25,1/6])


def test_future_data_do_not_change_target_and_copy():
    p=np.ones((40,3)); p[20]=[1.1,.9,1.]
    a=s.relative_momentum50(p,20,('A','B','C'))
    p[21:]=np.nan
    np.testing.assert_array_equal(a,s.relative_momentum50(p,20,('A','B','C')))
    assert a.sum()==1 and a[0]==.5 and np.all(a[1:]<=.5)


@pytest.mark.parametrize('t',[True,19,25,20.5])
def test_bad_indices(t):
    with pytest.raises(ValueError):s.relative_momentum50(np.ones((25,3)),t,('A','B','C'))


def test_bad_prefix_and_tickers():
    close=np.ones((25,3));close[3,0]=np.nan
    with pytest.raises(ValueError):s.relative_momentum50(close,20,('A','B','C'))
    with pytest.raises(ValueError):s.relative_momentum50(np.ones((25,3)),20,('A','A','C'))


def rows_fixture():
    return [dict(cost_bps=c,year=y,rule=r,total_return=.11 if r=='relative_momentum50' else .1,max_drawdown=.1)
            for c in s.COSTS for y in s.YEARS for r in s.RULES]


def test_gate_all_costs_and_three_wins_required():
    rows=rows_fixture(); assert s.gate_decision(rows)['passed']
    for r in rows:
        if r['rule']=='relative_momentum50' and r['year'] in (2018,2020) and r['cost_bps']==20:r['total_return']=.1
    out=s.gate_decision(rows)
    assert not out['passed'] and out['cost_gates'][-1]['strict_score_wins']==2
    assert out['decision']=='do_not_train_new_selector'


def test_drawdown_boundary_and_no_missing_or_duplicate_rows():
    rows=rows_fixture(); rows[1]['total_return']=.2;rows[1]['max_drawdown']=.12
    assert s.gate_decision(rows)['passed']
    rows[1]['max_drawdown']=.1201
    assert not s.gate_decision(rows)['passed']
    for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError):s.gate_decision(bad)
    rows[0]['total_return']=np.nan
    with pytest.raises(ValueError):s.gate_decision(rows)


def test_validation_calendar_has_no_test_key():
    intervals=s.validation_intervals(pd.bdate_range('2015-01-01','2026-10-22'))
    assert [p['year'] for p in intervals]==list(s.YEARS)
    assert all(set(p)=={'year','interval','dates'} for p in intervals)


def test_review_gate(tmp_path):
    with pytest.raises(ValueError,match='approval'):s.check_approval(tmp_path)
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'preflight_review.json').write_text(json.dumps({'status':'passed','manifest_sha256':'bad'}))
    with pytest.raises(ValueError,match='match'):s.check_approval(tmp_path)
    (tmp_path/'preflight_review.json').write_text(json.dumps({'status':'passed','manifest_sha256':s.digest(tmp_path/'manifest.json')}))
    s.check_approval(tmp_path)


@pytest.mark.parametrize('rule',s.RULES)
def test_synthetic_account_calendar_and_target_exposure(rule):
    panel=panel_fixture()
    frame,metrics=s.run_rule(panel,None,[120,150],rule,5)
    assert len(frame)==31 and metrics['return_days']==30
    trades=frame[frame.rebalance_executed.fillna(False).astype(bool)]
    assert len(trades)==6
    assert all(row[0]==.5 and np.isclose(sum(row[1:]),.5) for row in trades.requested_rule_target)
    if rule=='equal50':assert metrics['mean_requested_target_l1_from_equal50']==0
    else:assert metrics['fraction_requested_targets_different_from_equal50']==1
    np.testing.assert_allclose(frame.portfolio_value,np.cumprod(1+frame.net_return),atol=1e-12)


def test_synthetic_corporate_actions_match_direct_account():
    from rlasset.corporate_actions import CorporateActions, Dividend
    from rlasset.robust_env import RobustPortfolioEnv
    panel=panel_fixture(); actions=CorporateActions.empty(panel)
    for key in ('open','close','high','low'):
        getattr(panel,key)[125:,0]/=2
        getattr(panel,key)[130:,1]-=3
    actions.split_ratio[125,0]=2; actions.cash_per_share[130,1]=3
    actions=replace(actions,dividends=(Dividend(1,130,139,3.),))
    frame,_=s.run_rule(panel,actions,[120,150],'relative_momentum50',5)
    ref=RobustPortfolioEnv(panel,120,150,actions=actions,cost_bps=5,risk_aversion=0)
    for i in range(30):
        target=s.relative_momentum50(ref.signal_close,ref.t,panel.tickers) if i%5==0 else None
        ref.step_weights(target)
    np.testing.assert_allclose(frame.portfolio_value,[r['portfolio_value'] for r in ref.history],atol=0,rtol=0)
    assert frame.receivable.max()>0 and frame.dividend_paid.sum()>0


def test_evaluate_creates_all_24_registered_output_paths(tmp_path, monkeypatch):
    """Synthetic orchestration regression: never reads real financial data."""
    periods=[{'year':year,'interval':[120+i*10,125+i*10]} for i,year in enumerate(s.YEARS)]
    manifest={'validation_periods':periods}
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(s,'prepare',lambda *args:manifest)
    monkeypatch.setattr(s,'check_approval',lambda *args:None)
    monkeypatch.setattr(s,'load_inputs',lambda *args:('synthetic_panel','synthetic_actions'))
    calls=[]
    def run(panel,actions,interval,rule,cost):
        assert panel=='synthetic_panel' and actions=='synthetic_actions'
        calls.append((tuple(interval),rule,cost))
        metrics={'total_return':.01,'annual_return':.01,'max_drawdown':.02,
                 'annual_volatility':.05,'mean_turnover':.01,
                 'mean_requested_target_l1_from_equal50':0. if rule=='equal50' else 1/6}
        return pd.DataFrame({'synthetic':[1]}),metrics
    monkeypatch.setattr(s,'run_rule',run)
    saved=[]
    def save(path,ledger):
        assert path.parent.is_dir(), 'evaluate must create the ledger directory'
        assert ledger.synthetic.tolist()==[1]
        ledger.to_csv(path,index=False)
        saved.append(path)
    monkeypatch.setattr(s,'save_ledger',save)
    audited=[]
    def audit(path,data_dir,cost):
        assert path.is_file()
        audited.append((path,cost))
    monkeypatch.setattr(s,'audit_saved_ledger',audit)
    result=s.evaluate(tmp_path,tmp_path)
    expected={tmp_path/f'validation_{cost}bps'/str(year)/f'{rule}.csv'
              for cost in s.COSTS for year in s.YEARS for rule in s.RULES}
    assert len(calls)==len(saved)==len(audited)==24 and set(saved)==expected
    assert {(p.parent.parent.name,c) for p,c in audited}=={(f'validation_{c}bps',c) for c in s.COSTS}
    assert {interval for interval,_,_ in calls}=={tuple(p['interval']) for p in periods}
    assert not list(tmp_path.glob('test*'))
    assert not result['passed']  # equal synthetic scores do not count as wins
    assert (tmp_path/'gate_decision.json').is_file()
    assert len(json.loads((tmp_path/'run_record.json').read_text())['ledger_sha256'])==24
