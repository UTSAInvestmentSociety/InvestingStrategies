#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy 01 — Cross-Sectional Momentum (Monthly, Long–Short) with Bloomberg
STRICTLY via blpapi (no pdblp)

Requirements:
  - Bloomberg Terminal (running) + Desktop API permission
  - Python packages: blpapi, pandas, numpy, matplotlib (optional)

Install:
  pip install blpapi pandas numpy matplotlib
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import blpapi

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    # Bloomberg connection
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Universe (Bloomberg tickers). Change freely.
    universe: List[str] = None

    # Data fields and dates
    field: str = "PX_LAST"      # or 'TOT_RETURN_INDEX_GROSS_DVDS' where available
    start_date: str = "2010-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Historical request options
    periodicity: str = "DAILY"             # DAILY | WEEKLY | MONTHLY
    fill_option: str = "ALL_CALENDAR_DAYS" # or "ACTIVE_DAYS_ONLY"
    fill_method: str = "PREVIOUS_VALUE"    # or "NIL_VALUE"

    # Momentum parameters
    lookback_months: int = 12   # 12-month lookback
    skip_months: int = 1        # skip most recent 1 month (12_1)
    top_n: int = 3              # number of longs
    bottom_n: int = 3           # number of shorts

    # Portfolio/Backtest
    rebalance_freq: str = "M"   # monthly
    transaction_cost_bps: float = 5.0  # roundtrip bps applied on turnover
    max_weight_per_asset: float = 0.25 # per-side cap

    # Output
    out_dir: str = "./out_momentum_blpapi"
    save_plots: bool = False    # set True if you want matplotlib plots
    seed: int = 42

cfg = Config()
if cfg.universe is None:
    # Simple diversified ETF set
    cfg.universe = [
        "SPY US Equity",  # US equities
        "EFA US Equity",  # Developed ex-US equities
        "EEM US Equity",  # Emerging equities
        "AGG US Equity",  # US aggregate bonds
        "LQD US Equity",  # IG credit
        "HYG US Equity",  # HY credit
        "IEF US Equity",  # 7-10y UST
        "TLT US Equity",  # 20y+ UST
        "GLD US Equity",  # Gold
        "DBC US Equity",  # Broad commodities
    ]

np.random.seed(cfg.seed)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------
# Bloomberg session helpers (blpapi)
# -----------------------------
class BBG:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.session: Optional[blpapi.Session] = None
        self.ref_service = None

    def start(self):
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        self.session = blpapi.Session(opts)
        if not self.session.start():
            raise RuntimeError("Failed to start Bloomberg session.")
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata")
        self.ref_service = self.session.getService("//blp/refdata")

    def stop(self):
        try:
            if self.session is not None:
                self.session.stop()
        except Exception:
            pass

    def _send_request(self, request: blpapi.Request) -> List[blpapi.Message]:
        cid = self.session.sendRequest(request)
        msgs = []
        while True:
            ev = self.session.nextEvent(self.timeout_ms)
            for msg in ev:
                for rcid in msg.correlationIds():
                    if rcid == cid:
                        msgs.append(msg)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        return msgs

    def bdh_single_field(
        self,
        securities: List[str],
        field: str,
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
        fill_option: str = "ALL_CALENDAR_DAYS",
        fill_method: str = "PREVIOUS_VALUE",
    ) -> pd.DataFrame:
        """
        HistoricalDataRequest for a SINGLE field across multiple securities.
        Returns wide DataFrame (index=date, columns=tickers).
        """
        sdate = pd.to_datetime(start_date).strftime("%Y%m%d")
        edate = pd.to_datetime(end_date).strftime("%Y%m%d")

        req = self.ref_service.createRequest("HistoricalDataRequest")
        sec_el = req.getElement("securities")
        for s in securities:
            sec_el.appendValue(s)
        fld_el = req.getElement("fields")
        fld_el.appendValue(field)

        req.set("startDate", sdate)
        req.set("endDate", edate)
        req.set("periodicitySelection", periodicity)
        req.set("nonTradingDayFillOption", fill_option)
        req.set("nonTradingDayFillMethod", fill_method)

        msgs = self._send_request(req)
        frames = {}
        bad = []

        for msg in msgs:
            if msg.hasElement("responseError"):
                raise RuntimeError(f"HistoricalDataRequest error: {msg.getElement('responseError')}")
            if not msg.hasElement("securityData"):
                continue
            sdata = msg.getElement("securityData")
            sec = sdata.getElementAsString("security")

            if sdata.hasElement("securityError"):
                bad.append((sec, sdata.getElement("securityError").getElementAsString("message")))
                continue

            fdata = sdata.getElement("fieldData")
            recs = []
            for i in range(fdata.numValues()):
                el = fdata.getValueAsElement(i)
                dt = el.getElementAsDatetime("date")
                dt = pd.Timestamp(dt.year, dt.month, dt.day)
                val = np.nan
                if el.hasElement(field):
                    try:
                        val = el.getElementAsFloat(field)
                    except Exception:
                        try:
                            val = float(el.getElementAsString(field))
                        except Exception:
                            val = np.nan
                recs.append((dt, val))
            if recs:
                sdf = pd.DataFrame(recs, columns=["date", sec]).set_index("date").sort_index()
                frames[sec] = sdf

        if bad:
            print("BDH failed tickers (first few):", bad[:5])
        if not frames:
            raise RuntimeError(f"No historical data returned for field {field}")

        # Merge by date
        df = None
        for k, sdf in frames.items():
            df = sdf if df is None else df.join(sdf, how="outer")
        return df.sort_index()


# -----------------------------
# Signal & Backtest
# -----------------------------
def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample to month-end last available price."""
    return prices.resample("M").last()

def momentum_12_1(monthly_prices: pd.DataFrame, lookback=12, skip=1) -> pd.DataFrame:
    """
    12-month momentum excluding the most recent 1 month.
    Signal at t: (P_{t-skip} / P_{t-(lookback+skip)}) - 1
    """
    if lookback <= 0 or skip < 0:
        raise ValueError("Bad parameters for momentum lookback/skip.")
    s1 = monthly_prices.shift(skip)
    s2 = monthly_prices.shift(lookback + skip)
    signal = (s1 / s2) - 1.0
    return signal

def rank_long_short(signal_row: pd.Series, top_n: int, bottom_n: int) -> Tuple[pd.Index, pd.Index]:
    """Pick top and bottom assets (ties broken by ticker name order)."""
    if signal_row.dropna().empty:
        return pd.Index([]), pd.Index([])
    sr = signal_row.dropna().sort_values(ascending=False)
    longs = sr.head(top_n).index
    shorts = sr.tail(bottom_n).index
    return longs, shorts

def equal_weighted_weights(
    longs: pd.Index,
    shorts: pd.Index,
    max_weight_per_asset: float = 0.25
) -> pd.Series:
    """
    Dollar-neutral: sum(weights_long)=+1, sum(weights_short)=-1
    Equal-weight within each side, clipped by max per-asset cap.
    """
    w = {}
    if len(longs) > 0:
        w_long_each = min(1.0 / len(longs), max_weight_per_asset)
        for t in longs:
            w[t] = w_long_each
        s = sum(w[t] for t in longs)
        for t in longs:
            w[t] /= s if s else 1.0
    if len(shorts) > 0:
        w_short_each = min(1.0 / len(shorts), max_weight_per_asset)
        for t in shorts:
            w[t] = -w_short_each
        s = -sum(w[t] for t in shorts)
        for t in shorts:
            w[t] /= s if s else 1.0
    # exact neutrality fix
    if len(longs) > 0 and len(shorts) > 0:
        sum_long = sum(v for v in w.values() if v > 0)
        sum_short = -sum(v for v in w.values() if v < 0)
        if sum_long:
            for t in longs: w[t] /= sum_long
        if sum_short:
            for t in shorts: w[t] /= sum_short
    return pd.Series(w, dtype=float)

def portfolio_returns(
    weights_prev: pd.Series,
    prices_monthly: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> float:
    """Compute portfolio return from t0 to t1 using simple returns between month-ends."""
    p0 = prices_monthly.loc[t0]
    p1 = prices_monthly.loc[t1]
    asset_ret = (p1 - p0) / p0
    gross = float(np.nansum(weights_prev.reindex(asset_ret.index).fillna(0.0) * asset_ret.fillna(0.0)))
    return gross

def turnover_cost(
    w_prev: pd.Series,
    w_new: pd.Series,
    tc_bps: float
) -> float:
    """Turnover = sum |w_new - w_prev|; cost = bps * turnover."""
    aligned = pd.concat([w_prev.fillna(0), w_new.fillna(0)], axis=1)
    aligned.columns = ["prev", "new"]
    t = (aligned["new"] - aligned["prev"]).abs().sum()
    return float((tc_bps / 10000.0) * t)

def performance_stats(series: pd.Series, freq: str = "M") -> Dict[str, float]:
    """CAGR, annualized vol, Sharpe (rf=0), max drawdown."""
    ret = series.dropna()
    if ret.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    periods = 12 if freq.upper().startswith("M") else 252
    cum = (1 + ret).prod()
    years = len(ret) / periods
    cagr = cum ** (1 / max(years, 1e-9)) - 1
    vol = ret.std(ddof=1) * math.sqrt(periods)
    sharpe = (ret.mean() * periods) / vol if vol > 0 else np.nan
    eq = (1 + ret).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}


# -----------------------------
# Main
# -----------------------------
def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Connect to Bloomberg
    bb = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bb.start()
    print("Connected to Bloomberg.")

    # Fetch prices
    print(f"Fetching {cfg.field} for {len(cfg.universe)} tickers from {cfg.start_date} to {cfg.end_date}...")
    prices = bb.bdh_single_field(
        cfg.universe,
        cfg.field,
        cfg.start_date,
        cfg.end_date,
        periodicity=cfg.periodicity,
        fill_option=cfg.fill_option,
        fill_method=cfg.fill_method,
    )

    # Cleaning
    prices = prices.dropna(how="all").ffill()

    # Month-end prices & momentum signal
    mpx = to_month_end(prices)
    signal = momentum_12_1(mpx, lookback=cfg.lookback_months, skip=cfg.skip_months)

    # Rebalance dates
    rebal_dates = signal.index[signal.notna().sum(axis=1) > 0].sort_values()
    if len(rebal_dates) < cfg.lookback_months + cfg.skip_months + 2:
        raise SystemExit("Not enough data to run the momentum strategy. Extend date range or reduce lookback.")

    # Backtest
    port_rets_gross = []
    port_rets_net = []
    weights_store = {}
    signal_store = {}
    w_prev = pd.Series(dtype=float)

    for i in range(1, len(rebal_dates)):
        t1 = rebal_dates[i]       # end of period
        t0 = rebal_dates[i - 1]   # start of period

        sig_row = signal.loc[t0]
        longs, shorts = rank_long_short(sig_row, cfg.top_n, cfg.bottom_n)
        w_new = equal_weighted_weights(longs, shorts, max_weight_per_asset=cfg.max_weight_per_asset)

        # Turnover cost at rebalance
        tc = turnover_cost(w_prev, w_new, cfg.transaction_cost_bps)

        # Realized gross return t0->t1
        gross = portfolio_returns(w_new, mpx, t0, t1)
        net = gross - tc

        port_rets_gross.append(pd.Series({t1: gross}))
        port_rets_net.append(pd.Series({t1: net}))
        weights_store[t0] = w_new
        signal_store[t0] = sig_row
        w_prev = w_new

    # Combine
    gross_series = pd.concat(port_rets_gross).sort_index()
    net_series = pd.concat(port_rets_net).sort_index()

    stats_gross = performance_stats(gross_series, freq="M")
    stats_net = performance_stats(net_series, freq="M")

    weights_df = pd.DataFrame(weights_store).T.sort_index()
    signal_df = pd.DataFrame(signal_store).T.sort_index()

    # Save
    prices.to_csv(os.path.join(cfg.out_dir, "prices_daily.csv"))
    mpx.to_csv(os.path.join(cfg.out_dir, "prices_monthly.csv"))
    signal_df.to_csv(os.path.join(cfg.out_dir, "momentum_signals_12_1.csv"))
    weights_df.to_csv(os.path.join(cfg.out_dir, "weights_long_short.csv"))
    gross_series.to_csv(os.path.join(cfg.out_dir, "returns_gross_monthly.csv"))
    net_series.to_csv(os.path.join(cfg.out_dir, "returns_net_monthly.csv"))

    # Summary
    def pretty(d: Dict[str, float]) -> str:
        return f"CAGR: {d['CAGR']:.2%} | Vol: {d['Vol']:.2%} | Sharpe: {d['Sharpe']:.2f} | MaxDD: {d['MaxDD']:.2%}"

    print("\n=== Cross-Sectional Momentum (Monthly, Long–Short) — blpapi ===")
    print(f"Universe size: {len(cfg.universe)}  |  Long N: {cfg.top_n}  Short N: {cfg.bottom_n}")
    print(f"Lookback: {cfg.lookback_months}m  Skip: {cfg.skip_months}m  Rebalance: {cfg.rebalance_freq}")
    print(f"Transaction cost (bps): {cfg.transaction_cost_bps:.1f}")
    print("Gross:", pretty(stats_gross))
    print("Net:  ", pretty(stats_net))
    print(f"Files saved in: {os.path.abspath(cfg.out_dir)}")

    if cfg.save_plots:
        try:
            import matplotlib.pyplot as plt
            eq_gross = (1 + gross_series).cumprod()
            eq_net = (1 + net_series).cumprod()

            plt.figure(figsize=(10,6))
            eq_gross.plot(label="Gross")
            eq_net.plot(label="Net")
            plt.title("Momentum Long–Short Equity Curve (Monthly)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "equity_curve.png"), dpi=150)
            plt.close()
            print("Saved plot: equity_curve.png")
        except Exception as e:
            print("Plotting skipped:", e)

    # Close Bloomberg
    bb.stop()


if __name__ == "__main__":
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
