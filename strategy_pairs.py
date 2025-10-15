#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs Trading / Statistical Arbitrage — Bloomberg blpapi demo (Engle–Granger)

What this script does
---------------------
1) Pulls **daily prices** (PX_LAST) for a user-defined equity universe via **blpapi** (no pdblp).
2) (Optional) Pulls **GICS sector** names and only forms pairs **within the same sector**.
3) On an in-sample training window, tests every same-sector pair for **cointegration** (Engle–Granger) and
   simple sanity (high correlation). Selects the top K pairs by p-value.
4) For each selected pair, runs a **rolling mean-reversion backtest**:
   - Estimate hedge ratio (y = a + b x) on a rolling lookback.
   - Build spread s_t = y_t - (a + b x_t); compute rolling z-score.
   - **Entry** when |z| ≥ entry_z; **exit** when |z| ≤ exit_z (with optional stop at stop_z).
   - Trade weights are dollar-neutral (**long spread** = long y, short b·x; **short spread** vice versa).
   - Apply **transaction costs** on turnover.
5) Aggregates an **equal-weight portfolio across pairs**, reports performance, and saves CSV outputs.

Requirements
------------
- Bloomberg Terminal running with Desktop API permission
- Python packages: blpapi, pandas, numpy, statsmodels, matplotlib (optional)

Install:
    pip install blpapi pandas numpy statsmodels matplotlib
"""

import os
import sys
import math
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import blpapi
from statsmodels.tsa.stattools import coint

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# =========================
# Config
# =========================
@dataclass
class Config:
    # Bloomberg
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Universe (edit freely; script will sector-filter pairs)
    universe: List[str] = None

    # Dates
    start_date: str = "2014-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Fields/options
    price_field: str = "PX_LAST"
    periodicity: str = "DAILY"
    fill_option: str = "ACTIVE_DAYS_ONLY"
    fill_method: str = "NIL_VALUE"
    sector_field: str = "GICS_SECTOR_NAME"

    # Pair selection
    train_days: int = 500         # in-sample window for cointegration test (~2y of trading days)
    min_overlap: int = 400        # require at least this many common days in training
    min_corr: float = 0.80        # min Pearson corr of log prices in training
    max_pval: float = 0.05        # Engle–Granger p-value threshold
    max_pairs: int = 8            # use up to K best pairs per run

    # Backtest (rolling)
    beta_lookback: int = 60       # days to estimate hedge ratio (y = a + b x)
    z_lookback: int = 60          # days for z-score mean/std
    entry_z: float = 2.0
    exit_z: float = 0.75
    stop_z: float = 4.0           # optional hard stop; set None to disable
    tc_bps: float = 5.0           # transaction cost (bps) applied on turnover
    cap_per_pair: float = 1.0     # gross exposure per pair is normalized to 1.0

    # Output
    out_dir: str = "./out_pairs_blpapi"
    save_plots: bool = True

cfg = Config()
if cfg.universe is None:
    # A compact, sector-diverse set with plausible pairs
    cfg.universe = [
        # Consumer Staples (beverages)
        "KO US Equity", "PEP US Equity",
        # Energy (integrated oils)
        "XOM US Equity", "CVX US Equity",
        # Payments
        "V US Equity", "MA US Equity",
        # Banks
        "JPM US Equity", "BAC US Equity", "WFC US Equity",
        # Industrials (delivery)
        "FDX US Equity", "UPS US Equity",
        # Retail Home Improvement
        "HD US Equity", "LOW US Equity",
        # Pharma
        "PFE US Equity", "MRK US Equity",
        # Semis
        "AMD US Equity", "NVDA US Equity",
    ]


# =========================
# Bloomberg helpers (strict blpapi)
# =========================
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
            raise RuntimeError("Failed to start Bloomberg session")
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata")
        self.ref_service = self.session.getService("//blp/refdata")

    def stop(self):
        try:
            if self.session is not None:
                self.session.stop()
        except Exception:
            pass

    def _send(self, req: blpapi.Request) -> List[blpapi.Message]:
        cid = self.session.sendRequest(req)
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

    def bdh_single_field(self, tickers: List[str], field: str,
                         start_date: str, end_date: str,
                         periodicity="DAILY",
                         fill_option="ACTIVE_DAYS_ONLY",
                         fill_method="NIL_VALUE") -> pd.DataFrame:
        sdate = pd.to_datetime(start_date).strftime("%Y%m%d")
        edate = pd.to_datetime(end_date).strftime("%Y%m%d")
        req = self.ref_service.createRequest("HistoricalDataRequest")
        sec_el = req.getElement("securities")
        for s in tickers:
            sec_el.appendValue(s)
        fld_el = req.getElement("fields")
        fld_el.appendValue(field)
        req.set("startDate", sdate)
        req.set("endDate", edate)
        req.set("periodicitySelection", periodicity)
        req.set("nonTradingDayFillOption", fill_option)
        req.set("nonTradingDayFillMethod", fill_method)
        msgs = self._send(req)
        frames = {}
        for msg in msgs:
            if not msg.hasElement("securityData"):
                continue
            sdata = msg.getElement("securityData")
            sec = sdata.getElementAsString("security")
            if sdata.hasElement("securityError"):
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
                frames[sec] = pd.DataFrame(recs, columns=["date", sec]).set_index("date")
        df = None
        for _, sdf in frames.items():
            df = sdf if df is None else df.join(sdf, how="outer")
        return df.sort_index() if df is not None else pd.DataFrame()

    def ref_sectors(self, tickers: List[str], field: str = "GICS_SECTOR_NAME") -> pd.Series:
        req = self.ref_service.createRequest("ReferenceDataRequest")
        sec_el = req.getElement("securities")
        for s in tickers:
            sec_el.appendValue(s)
        fld_el = req.getElement("fields")
        fld_el.appendValue(field)
        msgs = self._send(req)
        rows = []
        for msg in msgs:
            if not msg.hasElement("securityData"):
                continue
            arr = msg.getElement("securityData")
            for i in range(arr.numValues()):
                sdata = arr.getValueAsElement(i)
                sec = sdata.getElementAsString("security")
                if sdata.hasElement("securityError"):
                    rows.append((sec, None))
                    continue
                fdata = sdata.getElement("fieldData")
                val = fdata.getElementAsString(field) if fdata.hasElement(field) else None
                rows.append((sec, val))
        return pd.Series({k: v for k, v in rows})


# =========================
# Utilities
# =========================
def perf_stats(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    periods = 252
    cum = float((1 + r).prod())
    years = len(r) / periods
    cagr = cum ** (1 / max(years, 1e-9)) - 1
    vol = r.std(ddof=1) * math.sqrt(periods)
    sharpe = (r.mean() * periods) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}

def pretty_stats(name: str, d: Dict[str, float]):
    print(f"{name:>16}: CAGR {d['CAGR']:.2%} | Vol {d['Vol']:.2%} | Sharpe {d['Sharpe']:.2f} | MaxDD {d['MaxDD']:.2%}")

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().pct_change()

def ols_hedge(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Estimate y = a + b x via OLS; returns (a, b)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])

def zscore(series: pd.Series) -> pd.Series:
    m = series.mean()
    s = series.std(ddof=1)
    return (series - m) / (s if s and s > 0 else np.nan)


# =========================
# Pair selection (Engle–Granger)
# =========================
def select_pairs(px: pd.DataFrame,
                 sectors: pd.Series,
                 train_days: int,
                 min_overlap: int,
                 min_corr: float,
                 max_pval: float,
                 max_pairs: int) -> List[Tuple[str, str, float, float]]:
    """
    Returns a list of (y, x, pval, corr) for selected pairs using the first 'train_days' rows.
    """
    px = px.dropna(how="all")
    train = px.iloc[:train_days].copy()
    # Map sector
    sec_map = sectors.reindex(train.columns).fillna("Unknown")
    pairs = []
    for sec in sec_map.dropna().unique():
        tickers = sec_map.index[sec_map == sec].tolist()
        for a, b in itertools.combinations(sorted(tickers), 2):
            df = train[[a, b]].dropna()
            if len(df) < min_overlap:
                continue
            # Corr on log prices to avoid drift
            lp = np.log(df)
            corr = float(lp[a].corr(lp[b]))
            if corr < min_corr:
                continue
            # Engle–Granger coint test (include constant)
            try:
                tstat, pval, _ = coint(lp[a].values, lp[b].values, trend='c')  # test residuals from a ~ b
            except Exception:
                continue
            if np.isnan(pval) or pval > max_pval:
                continue
            pairs.append((a, b, float(pval), corr))
    # sort by pval then by corr desc
    pairs.sort(key=lambda x: (x[2], -x[3]))
    return pairs[:max_pairs]


# =========================
# Backtest a single pair
# =========================
def backtest_pair(px: pd.DataFrame,
                  y: str, x: str,
                  beta_lb: int, z_lb: int,
                  entry_z: float, exit_z: float, stop_z: Optional[float],
                  tc_bps: float, cap: float) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      - daily return series for the pair strategy
      - a DataFrame with diagnostics per day (z, beta, position, weights)
    """
    df = px[[y, x]].dropna().copy()
    if len(df) < max(beta_lb, z_lb) + 10:
        return pd.Series(dtype=float), pd.DataFrame()

    r = pct_change(df).fillna(0.0)
    dates = df.index

    pos = 0  # -1 short spread, +1 long spread, 0 flat
    w_prev = np.array([0.0, 0.0])  # weights on [y, x]
    out = []
    rets = []

    for i in range(max(beta_lb, z_lb), len(df) - 1):
        t = dates[i]
        t1 = dates[i+1]

        y_w = df[y].iloc[i - beta_lb + 1: i + 1].values
        x_w = df[x].iloc[i - beta_lb + 1: i + 1].values
        a, b = ols_hedge(y_w, x_w)

        # spread & rolling z
        spread_hist = df[y].iloc[i - z_lb + 1: i + 1] - (a + b * df[x].iloc[i - z_lb + 1: i + 1])
        m, s = spread_hist.mean(), spread_hist.std(ddof=1)
        z = (df[y].iloc[i] - (a + b * df[x].iloc[i]) - m) / (s if s and s > 0 else np.nan)

        # desired position
        pos_desired = pos
        if np.isfinite(z):
            if pos == 0:
                if z >= entry_z:
                    pos_desired = -1   # short spread: short y, long b*x
                elif z <= -entry_z:
                    pos_desired = +1   # long spread: long y, short b*x
            else:
                if abs(z) <= exit_z:
                    pos_desired = 0
                elif stop_z is not None and abs(z) >= stop_z:
                    pos_desired = 0

        # convert desired position to dollar-neutral weights on y,x
        gross = abs(1.0) + abs(b)
        if gross == 0 or not np.isfinite(gross):
            w_new = np.array([0.0, 0.0])
        else:
            wy = (pos_desired / gross) * cap
            wx = (-pos_desired * b / gross) * cap
            w_new = np.array([wy, wx])

        # turnover cost at t (on change from w_prev to w_new)
        turnover = np.sum(np.abs(w_new - w_prev))
        tc = (tc_bps / 10000.0) * turnover

        # realized return on t->t1
        ri = r.iloc[i+1].values  # next-day returns [y, x]
        gross_ret = float(np.dot(w_new, ri))
        net_ret = gross_ret - tc

        # store
        rets.append(pd.Series({t1: net_ret}))
        out.append({
            "date": t,
            "a": a, "b": b, "z": z,
            "pos_prev": pos, "pos_new": pos_desired,
            "wy": w_new[0], "wx": w_new[1],
            "turnover": turnover, "tc": tc
        })

        # advance
        pos = pos_desired
        w_prev = w_new

    ret_series = pd.concat(rets).sort_index() if rets else pd.Series(dtype=float)
    diag = pd.DataFrame(out).set_index("date") if out else pd.DataFrame()
    return ret_series, diag


# =========================
# Main
# =========================
def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Connect Bloomberg
    bb = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bb.start()
    print("Connected to Bloomberg.")

    # Prices
    print("Fetching prices ...")
    px = bb.bdh_single_field(cfg.universe, cfg.price_field, cfg.start_date, cfg.end_date,
                             periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    px = px.ffill().dropna(how="all")

    # Sectors (for same-sector pairing)
    print("Fetching sectors ...")
    sectors = bb.ref_sectors(cfg.universe, cfg.sector_field)
    sectors = sectors.fillna("Unknown")

    # Select pairs on training window
    print("Selecting pairs (Engle–Granger on training window) ...")
    pairs = select_pairs(px, sectors, cfg.train_days, cfg.min_overlap, cfg.min_corr, cfg.max_pval, cfg.max_pairs)
    if not pairs:
        print("No eligible pairs found. Try relaxing min_corr/max_pval, extending train_days, or changing universe.")
        bb.stop()
        return

    print("\nSelected pairs (y, x, pval, corr):")
    for y, x, pval, corr in pairs:
        print(f"  {y:>14} ~ {x:<14}  p={pval:.4f}  corr={corr:.2f}")

    # Backtest each pair
    pair_returns = {}
    pair_stats = []
    for y, x, pval, corr in pairs:
        r, diag = backtest_pair(px, y, x,
                                cfg.beta_lookback, cfg.z_lookback,
                                cfg.entry_z, cfg.exit_z, cfg.stop_z,
                                cfg.tc_bps, cfg.cap_per_pair)
        if r.empty:
            continue
        pair_returns[f"{y} ~ {x}"] = r
        st = perf_stats(r)
        pair_stats.append((f"{y} ~ {x}", st["CAGR"], st["Vol"], st["Sharpe"], st["MaxDD"]))
        # save diagnostics
        diag.to_csv(os.path.join(cfg.out_dir, f"diag_{y.replace(' ','_')}_{x.replace(' ','_')}.csv"))

    if not pair_returns:
        print("No pair produced a valid backtest window.")
        bb.stop()
        return

    # Portfolio: equal-weight across pairs (on days where returns exist)
    print("\nAggregating portfolio (equal-weight across pairs) ...")
    returns_df = pd.concat(pair_returns, axis=1).sort_index()
    port_ret = returns_df.mean(axis=1)

    # Stats
    print("\n=== Pair-level performance ===")
    for name, cagr, vol, shp, mdd in pair_stats:
        print(f"{name:>30}: CAGR {cagr:.2%} | Vol {vol:.2%} | Sharpe {shp:.2f} | MaxDD {mdd:.2%}")

    print("\n=== Portfolio performance (equal-weight) ===")
    pretty_stats("Pairs Portfolio", perf_stats(port_ret))

    # Save outputs
    returns_df.to_csv(os.path.join(cfg.out_dir, "pair_returns_daily.csv"))
    port_ret.to_csv(os.path.join(cfg.out_dir, "portfolio_returns_daily.csv"))
    pd.DataFrame(pairs, columns=["y", "x", "pval", "corr"]).to_csv(os.path.join(cfg.out_dir, "selected_pairs.csv"), index=False)

    # Plots
    if cfg.save_plots and HAVE_PLT:
        try:
            eq = (1 + port_ret.fillna(0)).cumprod()
            plt.figure(figsize=(10,6))
            eq.plot(label="Pairs Portfolio")
            plt.title("Pairs Trading — Equity Curve (Daily)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "equity_curve.png"), dpi=140)
            plt.close()

            # z-sample from first pair (if diag saved)
            first_key = next(iter(pair_returns))
            y, x = first_key.split(" ~ ")
            diag_path = os.path.join(cfg.out_dir, f"diag_{y.replace(' ','_')}_{x.replace(' ','_')}.csv")
            if os.path.exists(diag_path):
                d = pd.read_csv(diag_path, parse_dates=["date"]).set_index("date")
                plt.figure(figsize=(10,4))
                d["z"].clip(-5, 5).plot()
                plt.axhline(cfg.entry_z, ls="--", c="r"); plt.axhline(-cfg.entry_z, ls="--", c="r")
                plt.axhline(cfg.exit_z, ls=":", c="g"); plt.axhline(-cfg.exit_z, ls=":", c="g")
                plt.title(f"Z-score (clipped) — {first_key}")
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.out_dir, f"zscore_{y.replace(' ','_')}_{x.replace(' ','_')}.png"), dpi=140)
                plt.close()
            print(f"Saved plots to: {os.path.abspath(cfg.out_dir)}")
        except Exception as e:
            print("Plotting skipped:", e)

    # Close
    bb.stop()
    print(f"\nFiles saved to: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
