"""v6 causal selection, independent raw ledgers and telescoping objectives."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.corporate_actions import CorporateActions, Dividend
from rlasset.data import MarketPanel
from rlasset.residual_env import trend_volatility_target
from rlasset.robust_env import RobustPortfolioEnv
from rlasset.selector_env_v6 import FixedExpertSelectorEnv, EXPERT_NAMES


def panel_fixture(days=300, flat=False):
    rng = np.random.default_rng(89)
    returns = rng.normal(.0003, .018, (days, 1)) + rng.normal(0, .006, (days, 3))
    if flat:
        returns[:] = 0
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    return MarketPanel(pd.bdate_range('2020-01-01', periods=days), ('A','B','C'),
                       close.copy(), close, close*1.01, close*.99, np.full_like(close, 10000))


def env(panel=None, **kwargs):
    config = dict(start=120, end=280)
    config.update(kwargs)
    return FixedExpertSelectorEnv(panel if panel is not None else panel_fixture(), **config)


def test_exact_frozen_rules_and_no_hidden_warmup():
    e = env()
    obs, info = e.reset(seed=12)
    assert EXPERT_NAMES == ('cash','equal50','vol10','trend10')
    targets = e.expert_targets()
    np.testing.assert_array_equal(targets[0], [1,0,0,0])
    np.testing.assert_array_equal(targets[1], [.5,1/6,1/6,1/6])
    for index, gate in [(2,False),(3,True)]:
        np.testing.assert_array_equal(targets[index], trend_volatility_target(e.base.signal_close,e.t,gate))
    assert obs.shape == (66,) and e.observation_space.contains(obs)
    np.testing.assert_array_equal(obs[-21:-8], np.zeros(13))
    assert e.trailing_argmax() == 0
    assert all(s.value == s.cash == 1 and not s.shares.any() and len(s.history)==1 for s in [e.base,*e.shadows])
    targets[:] = 999
    assert e.expert_targets().max() <= 1


@pytest.mark.parametrize('penalty', [False,True])
def test_reward_telescopes_to_terminal_log_nav_and_running_max_dd(penalty):
    e = env(drawdown_penalty=penalty)
    total, maxdd, peak = 0.,0.,1.
    for i in range(160):
        _, reward, term, trunc, info = e.step((i//10)%4)
        peak = max(peak,e.value)
        previous = maxdd
        maxdd = max(maxdd,1-e.value/peak)
        expected = 100*(np.log1p(info['net_return'])-(.5*(maxdd-previous) if penalty else 0))
        assert reward == pytest.approx(expected, abs=1e-13)
        total += reward
    assert term and not trunc
    assert total == pytest.approx(100*(np.log(e.value)-(.5*maxdd if penalty else 0)),abs=1e-11)
    assert any(r['current_drawdown']>0 and r['max_drawdown_increment']==0 for r in e.history[1:])
    for r in e.history[1:]:
        if r['max_drawdown_increment']==0:
            assert r['reward'] == pytest.approx(100*r['log_net_return'])


def test_raw_dividend_split_shadows_equal_standalone_with_own_costs():
    panel = panel_fixture()
    actions = CorporateActions.empty(panel)
    for key in ('open','close','high','low'):
        getattr(panel,key)[125:,0] /= 2
        getattr(panel,key)[130:,1] -= 3
    actions.split_ratio[125,0] = 2
    actions.cash_per_share[130,1] = 3
    actions = replace(actions, dividends=(Dividend(1,130,139,3.),))
    e = env(panel, actions=actions, end=160)
    standalone = [RobustPortfolioEnv(panel,120,160,actions=actions,risk_aversion=0) for _ in range(4)]
    for day in range(40):
        targets = e.expert_targets()
        e.step(day%4)
        for i, s in enumerate(standalone):
            _,_,term,trunc,info = s.step_weights(targets[i])
            assert info == e.shadows[i].history[-1]
            np.testing.assert_array_equal(s.shares,e.shadows[i].shares)
    assert any(r['receivable']>0 for r in e.shadows[1].history)
    assert any(r['dividend_paid']>0 for r in e.shadows[1].history)
    assert len({id(s.shares) for s in [e.base,*e.shadows]}) == 5


def test_future_prices_volume_and_corporate_events_do_not_change_prefix():
    panel = panel_fixture()
    future = replace(panel, **{k:getattr(panel,k).copy() for k in ('open','close','high','low','volume')})
    for key in ('open','close','high','low','volume'):
        getattr(future,key)[161:] *= 3
    actions = CorporateActions.empty(future)
    actions.cash_per_share[180,1] = 5
    actions.split_ratio[190,0] = 1.25
    actions = replace(actions,dividends=(Dividend(1,180,190,5.),))
    a,b = env(panel,end=160),env(future,end=160,actions=actions)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for i in range(40):
        np.testing.assert_array_equal(a.expert_targets(),b.expert_targets())
        x,y = a.step(i%4),b.step(i%4)
        np.testing.assert_array_equal(x[0],y[0])
        assert x[1:] == y[1:]


def test_reset_sampled_fragment_all_cash_and_terminated_truncated_match():
    e = env(training=True)
    starts = []
    for seed in (None,14,16):
        e.reset(seed=seed)
        starts.append(e.t)
        assert e.base.end-e.base.start == 126
        assert e.performance_history == [[],[],[],[]]
        assert all((s.start,s.end,s.t,s.rebalance_phase)==(e.base.start,e.base.end,e.t,0) for s in e.shadows)
        for i in range(126):
            _,_,term,trunc,_ = e.step(2)
            assert (term,trunc) == (False,i==125)
        assert all(len(h)==126 for h in e.performance_history)
        with pytest.raises(RuntimeError): e.step(0)
    assert len(set(starts)) > 1
    a = env(end=121)
    assert a.step(0)[2:4] == (True,False)


def test_calendar_and_truncation_mismatch_detected(monkeypatch):
    e = env(training=True)
    original = e.shadows[0].step_weights
    def corrupt(target):
        obs,r,term,trunc,info = original(target)
        return obs,r,term,not trunc,info
    monkeypatch.setattr(e.shadows[0], 'step_weights',corrupt)
    with pytest.raises(RuntimeError, match='truncated'): e.step(0)
    e = env()
    e.shadows[1].t += 1
    with pytest.raises(RuntimeError, match='calendar'): e.step(0)


def test_offcadence_requests_logged_ignored_not_executed_or_queued():
    e = env(panel_fixture(flat=True), end=130)
    e.step(1)
    for action in (3,2,0,3):
        _,_,_,_,info = e.step(action)
        assert info['requested_expert_index']==action and info['action_ignored']
        assert info['executed_expert_index'] is None
        assert info['previous_executed_expert']==1
        assert info['turnover']==0 and info['cost_fraction']==0
    info = e.step(0)[-1]
    assert info['executed_expert_index']==0 and not info['action_ignored']
    assert info['cost_fraction']>0


def test_switching_and_equal_target_mix_use_actual_not_average_shadow_returns():
    panel = panel_fixture(flat=True)
    e = env(panel,end=135,cost_bps=50)
    standalone = RobustPortfolioEnv(panel,120,135,cost_bps=50,risk_aversion=0)
    for i in range(15):
        target = e.equal_target_mix() if i<5 else e.expert_targets()[i//5%2]
        x = e.step_weights(target)
        y = standalone.step_weights(target)
        assert x[-1]['net_return']==y[-1]['net_return']
        assert e.value==standalone.value
    assert e.value != pytest.approx(np.mean([s.value for s in e.shadows]), abs=1e-8)
    assert sum(r['cost_fraction'] for r in e.history)>0


def test_trailing_tie_order_window_and_empty_count():
    e=env()
    e.performance_history = [[.1]+[0.]*20, [0.]*21, [0.]*21, [0.]*21]
    assert e.trailing_argmax()==0
    e.performance_history[2][-1]=.01
    assert e.trailing_argmax()==2
    assert e._observation()[-9]==20


@pytest.mark.parametrize('action', [-1,4,1.5,True,[1,2],np.nan])
def test_invalid_actions_do_not_advance(action):
    e=env()
    with pytest.raises(ValueError): e.step(action)
    assert e.t==120 and len(e.history)==1


def test_gym_contract():
    check_env(env(training=True),skip_render_check=True)
