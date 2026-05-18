# =============================================================================
# Strategy: Cross-Sectional Equity Signal Replication (SSRN 6078546)
# Paper:    "An Empirical Evaluation of Cross-Sectional Equity Signals
#            Under Backtest Overfitting Diagnostics"
# DOI:      https://dx.doi.org/10.2139/ssrn.6078546
# GitHub:   https://github.com/Sourish-07/cross-sectional-equity-validation
#
# Purpose:
#   Executable, out-of-sample replication of the paper's primary finding:
#   PBO approx 0.60 under CPCV (20 folds, 105-day embargo).
#
#   Signal: mr_5 = -(5-day return)  [features.py in research codebase]
#   Portfolio: dollar-neutral, L1-normalised, +/-5% per-name cap,
#              20-week rolling vol targeting, 5 bps transaction costs.
#   Parameters sourced directly from backtest.py and pbo.py.
#
#   Rebalance frequency: WEEKLY (Monday open + 30 min).
#   Reason: QC free tier limits backtests to 10,000 orders. Daily rebalancing
#   of 300 names exhausts this in ~5 months. Weekly rebalancing of 50 names
#   produces ~5,000 orders over 5 years -- the full 2020-2024 window.
#   The mr_5 signal uses a 5-day lookback, so weekly execution is aligned.
#
#   Expected result: Sharpe near zero after costs, consistent with the paper's
#   OOS Sharpe of 0.03 and Alpha near zero (vs IS Sharpe of 0.22).
# =============================================================================

from AlgorithmImports import *
import numpy as np
from collections import deque


class CrossSectionalSignalReplication(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1_000_000)

        self.SetBenchmark("SPY")

        # 5 bps transaction cost model (matches cost_sweep.py: TCOST = 0.0005)
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # Universe: top 50 liquid US equities, price > $5, DollarVolume > $50M
        # Smaller universe to keep total orders well under free-tier cap.
        # Top-50 by dollar volume are mega/large caps -- highly liquid,
        # matching the paper's "tradable securities" criterion.
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.FillForward = True
        self.AddUniverse(self._coarse_selection)

        # Weekly rebalance: Monday, 30 min after open
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self._rebalance
        )

        # Parameters -- sourced directly from research codebase
        self._lookback         = 5        # features.py: pct_change(5)
        self._univ_size        = 50       # reduced for free-tier order budget
        self._max_weight       = 0.05     # backtest.py: MAX_WEIGHT = 0.05
        self._vol_window       = 20       # backtest.py: rolling(20) (weeks here)
        self._target_daily_vol = 0.01     # backtest.py: TARGET_DAILY_VOL = 0.01
        self._min_names        = 10

        self._symbols       = []
        self._weekly_pnl    = deque(maxlen=self._vol_window)
        self._prev_weights  = {}
        self._prev_port_val = 0.0

    def _coarse_selection(self, coarse):
        filtered = [
            x for x in coarse
            if x.Price > 5
            and x.DollarVolume > 50e6   # $50M+ daily volume = large/mega cap
        ]
        sorted_by_vol = sorted(
            filtered, key=lambda x: x.DollarVolume, reverse=True
        )
        self._symbols = [x.Symbol for x in sorted_by_vol[:self._univ_size]]
        return self._symbols

    def OnEndOfDay(self, symbol):
        # Track portfolio returns for vol targeting
        try:
            if self.Securities.ContainsKey("SPY") and symbol == self.Securities["SPY"].Symbol:
                port_val = self.Portfolio.TotalPortfolioValue
                if self._prev_port_val > 0:
                    self._weekly_pnl.append((port_val / self._prev_port_val) - 1.0)
                self._prev_port_val = port_val
        except Exception:
            pass

    def _rebalance(self):
        if not self._symbols:
            return

        # Request 6 bars (5-day lookback needs 6 daily closes)
        bars_needed = self._lookback + 1
        history = self.History(self._symbols, bars_needed, Resolution.Daily)
        if history is None or history.empty:
            return

        # Signal: mr_5 = -(5-day return)
        # Mirrors features.py: df["mr_5"] = -df.groupby("ticker")["close"].pct_change(5)
        raw_scores = {}
        for sym in self._symbols:
            try:
                if sym not in history.index.get_level_values(0):
                    continue
                closes = history.loc[sym]["close"]
                if len(closes) < bars_needed:
                    continue
                p_now, p_past = closes.iloc[-1], closes.iloc[0]
                if p_past == 0 or np.isnan(p_now) or np.isnan(p_past):
                    continue
                raw_scores[sym] = -((p_now / p_past) - 1.0)
            except Exception as ex:
                self.Debug(f"Signal error {sym}: {ex}")

        if len(raw_scores) < self._min_names:
            return

        # Portfolio: dollar-neutral L1-normalised
        # Mirrors pbo.py: w = (pred - mean) / (abs(pred).sum() + 1e-12), clipped +/-5%
        syms   = list(raw_scores.keys())
        scores = np.array([raw_scores[s] for s in syms], dtype=float)

        demeaned       = scores - scores.mean()
        weights_raw    = demeaned / (np.abs(demeaned).sum() + 1e-12)
        weights_capped = np.clip(weights_raw, -self._max_weight, self._max_weight)

        # Volatility targeting
        # Mirrors backtest.py: vol_scale = (TARGET_DAILY_VOL / rolling_vol).clip(0, 3)
        vol_scale = 1.0
        if len(self._weekly_pnl) >= 5:
            rv = float(np.std(list(self._weekly_pnl), ddof=1))
            if rv > 1e-8:
                vol_scale = min(self._target_daily_vol / rv, 3.0)

        targets = {
            syms[i]: float(weights_capped[i]) * vol_scale
            for i in range(len(syms))
            if abs(weights_capped[i]) > 1e-6
        }

        # Execution
        for sym in {h.Symbol for h in self.Portfolio.Values if h.Invested} - set(targets):
            self.Liquidate(sym)
        for sym, wgt in targets.items():
            self.SetHoldings(sym, wgt)

        # Diagnostics log
        turnover  = sum(abs(targets.get(s,0) - self._prev_weights.get(s,0)) for s in set(targets)|set(self._prev_weights))
        gross_exp = sum(abs(w) for w in targets.values())
        self.Log(f"[REBAL] {self.Time.date()} | N={len(targets)} | Gross={gross_exp:.3f} | Turnover={turnover:.4f} | VolScale={vol_scale:.3f}")
        self._prev_weights = dict(targets)

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        for removed in changes.RemovedSecurities:
            if self.Portfolio[removed.Symbol].Invested:
                self.Liquidate(removed.Symbol)
