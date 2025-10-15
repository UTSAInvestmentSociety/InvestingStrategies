#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimum-Variance / Mean-Variance Optimization (rolling backtest)
STRICTLY via Bloomberg blpapi (no pdblp)

What it does
------------
- Pulls historical prices from Bloomberg (TR index preferred, else price)
- Builds monthly returns for a user universe
- Each month, estimates μ and Σ on a rolling lookback window
- Solves two Markowitz problems with LONG-ONLY constraints:
    (1) Global Minimum-Variance (GMV)
    (2) Max-Sharpe via target-return sweep on the efficient frontier
- Compares to naïve 1/N
- Tracks out-of-sample performance for all three strategies
- Optionally plots an efficient frontier snapshot and equity curves
- Saves CSV outputs (weights & returns)

Requirements
------------
- Bloomberg Terminal running (Desktop API enabled)
- Python packages: blpapi, pandas, numpy, cvxpy, matplotlib (optional)

Install:
    pip install blpapi pandas numpy cvxpy matplotlib
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import blpapi

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# Optional: prefer a QP solver well-suited for convex QPs
import cvxpy as cp

# =========================
# Config
# =========================
@dataclass
class Config:
    # Bloomberg connection
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Universe (replace with your research universe or historical index members)
    universe: List[str] = None

    # Dates
    start_date: str = "2012-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Historical request options
    periodicity: str = "DAILY"                    # DAILY | WEEKLY | MONTHLY (we resample to monthly)
    fill_option: str = "ALL_CALENDAR_DAYS"        # or "ACTIVE_DAYS_ONLY"
    fill_method: str = "PREVIOUS_VALUE"           # or "NIL_VALUE"

    # Fields
    tri_field: str = "TOT_RETURN_INDEX_GROSS_DVDS"  # preferred for total returns
    px_field: str = "PX_LAST"                       # fallback if TRI not available

    # Optimization / backtest
    lookback_months: int = 60            # rolling estimation window (e.g., 60 months)
    rebalance_freq: str = "M"            # monthly
    long_only: bool = True               # no shorting
    max_weight_per_name: float = 0.20    # per-asset cap (e.g., 20%)
    rf_annual: float = 0.0               # risk-free (annualized) for Sharpe; set 0 for simplicity

    # Estimation stabilizers
    mu_shrink_to_zero: float = 0.5       # blend μ: mu_hat = (1-a)*sample_mean + a*0
    cov_diag_shrink: float = 0.10        # Σ_hat = (1-a)*sample + a*diag(sample)

    # Efficient frontier sweep for Max-Sharpe
    n_frontier_pts: int = 25             # target-return grid size

    # Output & plots
    out_dir: str = "./out_mvo_blpapi"
    save_plots: bool = True

cfg = Config()
if cfg.universe is None:
    # A small, diversified ETF universe (can replace with equities)
    cfg.universe = [
        "SPY US Equity",  # US Equities
        "EFA US Equity",  # Dev ex-US
        "EEM US Equity",  # EM
        "IEF US Equity",  # 7-10y UST
        "TLT US Equity",  # 20y+ UST
        "LQD US Equity",  # IG Credit
        "HYG US Equity",  # HY Credit
        "GLD US Equity",  # Gold
        "DBC US Equity",  # Commodities
        "VNQ US Equity",  # US REITs
    ]

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)


# =========================
# Bloomberg session helper (strict blpapi)
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
        tickers: List[str],
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
        for s in tickers:
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

        df = None
        for k, sdf in frames.items():
            df = sdf if df is None else df.join(sdf, how="outer")
        return df.sort_index()


# =========================
# Utilities
# =========================
def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("M").last()

def monthly_returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    mpx = to_month_end(px).ffill()
    return mpx.pct_change()

def shrink_mu(mu: pd.Series, a: float) -> pd.Series:
    """Shrink μ to zero by weight 'a'."""
    a = float(np.clip(a, 0.0, 1.0))
    return (1 - a) * mu  # + a * 0

def shrink_cov_to_diag(S: pd.DataFrame, a: float) -> pd.DataFrame:
    """Σ_hat = (1-a) Σ + a diag(Σ). Keeps PSD if 0<=a<=1."""
    a = float(np.clip(a, 0.0, 1.0))
    D = np.diag(np.diag(S.values))
    return pd.DataFrame((1 - a) * S.values + a * D, index=S.index, columns=S.columns)

def perf_stats(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    periods = 12
    cum = float((1 + r).prod())
    years = len(r) / periods
    cagr = cum ** (1 / max(years, 1e-9)) - 1
    vol = r.std(ddof=1) * math.sqrt(periods)
    sharpe = (r.mean() * periods) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}

def print_stats(name: str, d: Dict[str, float]):
    print(f"{name:>10}:  CAGR {d['CAGR']:.2%} | Vol {d['Vol']:.2%} | Sharpe {d['Sharpe']:.2f} | MaxDD {d['MaxDD']:.2%}")


# =========================
# Optimizers (cvxpy)
# =========================
def solve_gmv(Sigma: pd.DataFrame, long_only: bool, w_cap: float) -> pd.Series:
    names = list(Sigma.columns)
    n = len(names)
    # Variables
    w = cp.Variable(n)
    # Objective
    obj = cp.Minimize(cp.quad_form(w, Sigma.values))
    # Constraints
    cons = [cp.sum(w) == 1.0]
    if long_only:
        cons += [w >= 0.0]
    if w_cap is not None and w_cap < 1.0:
        cons += [w <= w_cap]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        # Try alternative solver
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        raise RuntimeError("GMV optimization failed (infeasible or solver error).")
    return pd.Series(np.asarray(w.value).ravel(), index=names)

def solve_min_var_for_target_return(mu: pd.Series,
                                    Sigma: pd.DataFrame,
                                    r_target: float,
                                    long_only: bool,
                                    w_cap: float) -> Optional[pd.Series]:
    names = list(Sigma.columns)
    n = len(names)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Sigma.values))
    cons = [cp.sum(w) == 1.0, mu.values @ w >= r_target]
    if long_only:
        cons += [w >= 0.0]
    if w_cap is not None and w_cap < 1.0:
        cons += [w <= w_cap]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        # Try a second solver for robustness
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return pd.Series(np.asarray(w.value).ravel(), index=names)

def efficient_frontier(mu: pd.Series,
                       Sigma: pd.DataFrame,
                       n_pts: int,
                       long_only: bool,
                       w_cap: float) -> Tuple[np.ndarray, np.ndarray, List[pd.Series]]:
    """
    Sweep target returns (between small and large percentiles of μ) and solve
    min-variance for each target. Returns arrays of (risk, return) and list of weights.
    """
    mu_vals = mu.values
    lo = float(np.percentile(mu_vals, 10))
    hi = float(np.percentile(mu_vals, 90))
    if hi <= lo:
        # fallback to +/- small epsilon around mean
        m = float(mu_vals.mean())
        lo, hi = m - abs(m)*0.5 - 1e-4, m + abs(m)*0.5 + 1e-4

    targets = np.linspace(lo, hi, n_pts)
    risks, rets, W = [], [], []
    for r_t in targets:
        w = solve_min_var_for_target_return(mu, Sigma, r_t, long_only, w_cap)
        if w is None:
            continue
        port_var = float(w.values @ Sigma.values @ w.values)
        port_ret = float(mu.values @ w.values)
        risks.append(math.sqrt(max(port_var, 0.0)))
        rets.append(port_ret)
        W.append(w)
    return np.array(risks), np.array(rets), W

def pick_max_sharpe_from_frontier(risks: np.ndarray, rets: np.ndarray, W: List[pd.Series], rf_monthly: float) -> pd.Series:
    if len(W) == 0:
        raise RuntimeError("Frontier contains no feasible portfolios; cannot pick Max-Sharpe.")
    # Monthly Sharpe = (E[r] - rf) / sigma
    sharpe = (rets - rf_monthly) / np.where(risks > 1e-12, risks, np.nan)
    idx = np.nanargmax(sharpe)
    return W[int(idx)]


# =========================
# Backtest
# =========================
def run_backtest():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Connect & fetch prices
    bbg = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bbg.start()
    print("Connected to Bloomberg.")

    print("Fetching TRI and PX_LAST series...")
    tri = bbg.bdh_single_field(cfg.universe, cfg.tri_field, cfg.start_date, cfg.end_date,
                               periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    px  = bbg.bdh_single_field(cfg.universe, cfg.px_field,  cfg.start_date, cfg.end_date,
                               periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)

    # Prefer TRI if coverage decent
    tri_coverage = tri.notna().mean().mean() if not tri.empty else 0.0
    prices = tri if tri_coverage > 0.5 else px
    if tri_coverage <= 0.5:
        print("Warning: TRI coverage low; using PX_LAST.")

    # 2) Monthly returns
    rets = monthly_returns_from_prices(prices)
    rets = rets.loc[rets.index[1:]]  # drop first NaN row

    # 3) Rolling backtest
    dates = list(rets.index)
    if len(dates) <= cfg.lookback_months + 12:
        print("Warning: short sample; consider extending start_date.")

    # Containers
    gmv_r, msh_r, eq_r = [], [], []
    gmv_w_hist: Dict[pd.Timestamp, pd.Series] = {}
    msh_w_hist: Dict[pd.Timestamp, pd.Series] = {}
    eq_w_hist:  Dict[pd.Timestamp, pd.Series] = {}

    rf_m = (1 + cfg.rf_annual) ** (1/12) - 1  # monthly RF

    for i in range(cfg.lookback_months, len(dates)-1):
        asof = dates[i]        # end of lookback window
        nxt  = dates[i+1]      # next month (realized return)
        window = rets.iloc[i - cfg.lookback_months : i]  # T x N

        # Drop assets with insufficient data
        valid_cols = window.columns[window.notna().mean() > 0.95]  # require 95% coverage
        W = window[valid_cols].dropna()
        if W.empty or len(valid_cols) < 3:
            # Not enough data; skip
            continue

        mu_hat = W.mean(axis=0)              # monthly mean
        mu_hat = shrink_mu(mu_hat, cfg.mu_shrink_to_zero)

        Sigma = W.cov()
        Sigma = shrink_cov_to_diag(Sigma, cfg.cov_diag_shrink)

        # GMV (long-only, caps)
        w_gmv = solve_gmv(Sigma, long_only=cfg.long_only, w_cap=cfg.max_weight_per_name)

        # Frontier & max Sharpe (long-only)
        risks, rets_f, W_f = efficient_frontier(mu_hat, Sigma, cfg.n_frontier_pts, cfg.long_only, cfg.max_weight_per_name)
        w_msh = pick_max_sharpe_from_frontier(risks, rets_f, W_f, rf_m)

        # 1/N equal-weight on same valid set
        w_eq = pd.Series(1.0 / len(valid_cols), index=valid_cols)

        # Compute next-month realized return for each
        next_r = rets.loc[nxt, valid_cols].fillna(0.0)
        gmv_r.append(pd.Series({nxt: float((w_gmv.reindex(valid_cols, fill_value=0) * next_r).sum())}))
        msh_r.append(pd.Series({nxt: float((w_msh.reindex(valid_cols, fill_value=0) * next_r).sum())}))
        eq_r.append (pd.Series({nxt: float((w_eq  .reindex(valid_cols, fill_value=0) * next_r).sum())}))

        # Store weights
        gmv_w_hist[asof] = w_gmv
        msh_w_hist[asof] = w_msh
        eq_w_hist[asof]  = w_eq

    # 4) Combine returns
    gmv_series = (pd.concat(gmv_r).sort_index() if gmv_r else pd.Series(dtype=float))
    msh_series = (pd.concat(msh_r).sort_index() if msh_r else pd.Series(dtype=float))
    eq_series  = (pd.concat(eq_r ).sort_index() if eq_r  else pd.Series(dtype=float))

    # 5) Stats
    print("\n=== Out-of-sample performance (monthly) ===")
    print_stats("GMV", perf_stats(gmv_series))
    print_stats("MaxShp", perf_stats(msh_series))
    print_stats("EqualW", perf_stats(eq_series))

    # 6) Save outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    gmv_series.to_csv(os.path.join(cfg.out_dir, "gmv_returns_monthly.csv"))
    msh_series.to_csv(os.path.join(cfg.out_dir, "maxsharpe_returns_monthly.csv"))
    eq_series.to_csv (os.path.join(cfg.out_dir, "equal_weight_returns_monthly.csv"))
    pd.DataFrame(gmv_w_hist).T.to_csv(os.path.join(cfg.out_dir, "weights_gmv.csv"))
    pd.DataFrame(msh_w_hist).T.to_csv(os.path.join(cfg.out_dir, "weights_maxsharpe.csv"))
    pd.DataFrame(eq_w_hist ).T.to_csv(os.path.join(cfg.out_dir, "weights_equal.csv"))

    # 7) Frontier & plots (snapshot at last rebalance used)
    if cfg.save_plots and HAVE_PLT and len(msh_w_hist) > 0:
        try:
            # Pick last estimation window
            last_date = sorted(msh_w_hist.keys())[-1]
            idx = dates.index(last_date)
            window = rets.iloc[idx - cfg.lookback_months : idx]
            valid_cols = window.columns[window.notna().mean() > 0.95]
            W = window[valid_cols].dropna()
            mu_hat = shrink_mu(W.mean(axis=0), cfg.mu_shrink_to_zero)
            Sigma  = shrink_cov_to_diag(W.cov(), cfg.cov_diag_shrink)
            risks, rets_f, W_f = efficient_frontier(mu_hat, Sigma, 50, cfg.long_only, cfg.max_weight_per_name)

            # Points: GMV, Max-Sharpe, Equal
            w_gmv = solve_gmv(Sigma, cfg.long_only, cfg.max_weight_per_name)
            var_gmv = float(w_gmv.values @ Sigma.values @ w_gmv.values)
            ret_gmv = float(mu_hat.values @ w_gmv.values)

            w_msh = pick_max_sharpe_from_frontier(risks, rets_f, W_f, rf_m)
            var_msh = float(w_msh.values @ Sigma.values @ w_msh.values)
            ret_msh = float(mu_hat.values @ w_msh.values)

            w_eq  = pd.Series(1.0 / len(valid_cols), index=valid_cols)
            var_eq = float(w_eq.values @ Sigma.values @ w_eq.values)
            ret_eq = float(mu_hat.values @ w_eq.values)

            # Plot frontier
            plt.figure(figsize=(9,6))
            plt.plot(risks, rets_f, lw=2)
            plt.scatter(math.sqrt(max(var_gmv,0)), ret_gmv, marker="o", label="GMV")
            plt.scatter(math.sqrt(max(var_msh,0)), ret_msh, marker="*", s=120, label="Max Sharpe")
            plt.scatter(math.sqrt(max(var_eq ,0)), ret_eq , marker="s", label="1/N")
            plt.xlabel("Ex-ante risk (σ, monthly)")
            plt.ylabel("Ex-ante return (μ, monthly)")
            plt.title(f"Efficient Frontier snapshot @ {last_date.date()}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "efficient_frontier_snapshot.png"), dpi=140)
            plt.close()

            # Equity curves
            plt.figure(figsize=(9,6))
            (1+gmv_series).cumprod().plot(label="GMV")
            (1+msh_series).cumprod().plot(label="Max Sharpe")
            (1+eq_series ).cumprod().plot(label="1/N")
            plt.title("Equity Curves (OOS monthly)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "equity_curves.png"), dpi=140)
            plt.close()
            print(f"Saved plots to: {os.path.abspath(cfg.out_dir)}")
        except Exception as e:
            print("Plotting skipped:", repr(e))

    # Close session
    bbg.stop()
    print(f"\nFiles saved to: {os.path.abspath(cfg.out_dir)}")


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    try:
        run_backtest()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
