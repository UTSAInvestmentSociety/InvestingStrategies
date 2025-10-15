#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index Tracking & Smart Beta — RGUSTSC Index with hard-coded members (blpapi only)

What this script does
---------------------
- Tracks the **RGUSTSC Index** using a fixed list of semiconductor tickers.
- Builds three portfolios and compares them to the index:
    1) Tracking-error minimizer (long-only, sum=1) via projected gradient descent
    2) Equal-weight smart beta
    3) Inverse-volatility (1/σ) smart beta
- Monthly rebalancing; 60-day lookback for estimation.
- Saves daily returns and weights; optionally plots equity curves.

Requirements
------------
    pip install blpapi pandas numpy matplotlib
Bloomberg Terminal must be running with Desktop API enabled.
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import blpapi

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)


# =========================
# Config
# =========================
@dataclass
class Config:
    # Bloomberg
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Index and hard-coded members
    index_ticker: str = "RGUSTSC Index"
    members: List[str] = None  # filled below

    # History window
    start_date: str = "2018-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Price field / BDH options
    price_field: str = "PX_LAST"
    periodicity: str = "DAILY"
    fill_option: str = "ACTIVE_DAYS_ONLY"
    fill_method: str = "NIL_VALUE"

    # Rolling estimation & rebalancing
    lookback_days: int = 60
    rebalance_freq: str = "M"  # monthly

    # Tracking optimizer params
    max_iter: int = 2000
    tol: float = 1e-8

    # Output
    out_dir: str = "./out_index_tracking_rgustsc"
    save_plots: bool = True

cfg = Config()
cfg.members = [
    "NVDA UW Equity","AVGO UW Equity","INTC UW Equity","QCOM UW Equity","MU UW Equity",
    "AMD UW Equity","AVT UW Equity","TXN UW Equity","ADI UW Equity","MRVL UW Equity",
    "GFS UW Equity","ON UW Equity","MCHP UW Equity","SWKS UW Equity","QRVO UW Equity",
    "VSH UN Equity","MPWR UW Equity","CRUS UW Equity","AEIS UW Equity","DIOD UW Equity",
    "SYNA UW Equity","SMTC UW Equity","MTSI UW Equity","FORM UW Equity","ALGM UW Equity",
    "SLAB UW Equity","AOSL UW Equity","RMBS UW Equity","ALAB UW Equity","CRDO UW Equity",
    "LSCC UW Equity","POWI UW Equity","MXL UW Equity","AMBA UW Equity","SKYT UR Equity",
    "SITM UQ Equity","INDI UR Equity","LASR UW Equity","CEVA UW Equity","NVTS UQ Equity",
    "KOPN UR Equity","NVEC UR Equity","AEVA UW Equity","ATOM UR Equity",
]


# =========================
# Bloomberg helper (strict blpapi)
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

    def _send(self, request: blpapi.Request) -> List[blpapi.Message]:
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


# =========================
# Utilities
# =========================
def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(index.to_period("M").to_timestamp("M")))).intersection(index)

def perf_stats(r: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    cum = float((1 + r).prod())
    years = len(r) / freq
    cagr = cum ** (1 / max(years, 1e-9)) - 1
    vol = r.std(ddof=1) * math.sqrt(freq)
    sharpe = (r.mean() * freq) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd}

def info_ratio(active: pd.Series, freq: int = 252) -> float:
    a = active.dropna()
    sd = a.std(ddof=1) * math.sqrt(freq)
    mu = a.mean() * freq
    return (mu / sd) if sd > 0 else np.nan

def tracking_error(active: pd.Series, freq: int = 252) -> float:
    a = active.dropna()
    return a.std(ddof=1) * math.sqrt(freq)

def inverse_vol_weights(returns_window: pd.DataFrame) -> pd.Series:
    vol = returns_window.std(ddof=1)
    w = 1.0 / vol.replace(0, np.nan)
    w = w.fillna(0.0)
    return w / w.sum() if w.sum() > 0 else pd.Series(0.0, index=returns_window.columns)

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection to {w >= 0, sum(w)=1}."""
    v = v.ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    theta = (cssv[rho_idx[-1]] - 1) / float(rho_idx[-1] + 1) if len(rho_idx) else 0.0
    return np.maximum(v - theta, 0.0)

def te_minimizer_weights(R: np.ndarray, r: np.ndarray,
                         w0: Optional[np.ndarray] = None,
                         max_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    """
    Minimize ||R w - r||^2 s.t. w >= 0, sum(w)=1 via projected gradient descent.
    R: T x N asset returns (lookback window)
    r: T vector of index returns (lookback window)
    """
    T, N = R.shape
    w = np.full(N, 1.0 / N) if w0 is None else w0.copy()

    # Estimate Lipschitz constant L for step size (2 * lambda_max(R'R)/T)
    G = (R.T @ R) * (2.0 / T)
    try:
        v = np.random.randn(N); v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            v = G @ v
            v /= (np.linalg.norm(v) + 1e-12)
        L = float(v @ (G @ v))
        if not np.isfinite(L) or L <= 0:
            L = 1.0
    except Exception:
        L = 1.0
    alpha = 1.0 / L
    Rt = R.T

    for _ in range(max_iter):
        resid = (R @ w - r)
        grad = (2.0 / T) * (Rt @ resid)
        w_new = project_to_simplex(w - alpha * grad)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w


# =========================
# Main
# =========================
def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Connect
    bbg = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bbg.start()
    print("Connected to Bloomberg.")

    # Members (hard-coded list)
    members = pd.DataFrame({"ticker": cfg.members})
    tickers = members["ticker"].tolist()
    print(f"Using hard-coded members for {cfg.index_ticker}: {len(tickers)} names")

    # Fetch historical prices
    print("Fetching historical prices ...")
    px_idx = bbg.bdh_single_field([cfg.index_ticker], cfg.price_field, cfg.start_date, cfg.end_date,
                                  periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    px_stk = bbg.bdh_single_field(tickers, cfg.price_field, cfg.start_date, cfg.end_date,
                                  periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)

    # Clean & align
    px_idx = px_idx.ffill().dropna()
    px_stk = px_stk.ffill().dropna(how="all")
    common_dates = px_idx.index.intersection(px_stk.index)
    px_idx = px_idx.reindex(common_dates)
    px_stk = px_stk.reindex(common_dates)

    # Daily returns
    r_idx = px_idx.pct_change().iloc[1:, 0]   # Series
    r_stk = px_stk.pct_change().iloc[1:, :]   # DataFrame

    # Rebalance dates (month-end with enough lookback history)
    rebal_dates = month_ends(r_idx.index)
    rebal_dates = rebal_dates[rebal_dates >= (r_idx.index.min() + pd.tseries.offsets.BDay(cfg.lookback_days))]
    if len(rebal_dates) < 3:
        print("Too few rebalance dates; extend date range or reduce lookback.")
        bbg.stop()
        return

    # Containers
    te_w_hist: Dict[pd.Timestamp, pd.Series] = {}
    ew_w_hist: Dict[pd.Timestamp, pd.Series] = {}
    iv_w_hist: Dict[pd.Timestamp, pd.Series] = {}
    ret_te, ret_ew, ret_iv = [], [], []

    # Static EW base (rebalance to active names each month)
    ew_base = pd.Series(1.0 / r_stk.shape[1], index=r_stk.columns)

    print("Running monthly rebalancing ...")
    for i in range(1, len(rebal_dates)):
        t0, t1 = rebal_dates[i-1], rebal_dates[i]

        est_idx = r_idx.loc[:t0].index[-cfg.lookback_days:]
        R = r_stk.loc[est_idx].dropna(axis=1, how="any")
        idx_est = r_idx.loc[est_idx]

        # Relax if too few complete columns
        if R.shape[1] < 8:
            R = r_stk.loc[est_idx].fillna(0.0)
            idx_est = r_idx.loc[est_idx]

        # TE-minimizer weights
        w0 = np.full(R.shape[1], 1.0 / max(1, R.shape[1]))
        w_te = pd.Series(te_minimizer_weights(R.values, idx_est.values, w0=w0,
                                              max_iter=cfg.max_iter, tol=cfg.tol), index=R.columns)

        # Equal-weight (rebalance to active)
        w_ew = ew_base.reindex(R.columns, fill_value=0.0)
        w_ew = w_ew / w_ew.sum() if w_ew.sum() > 0 else w_ew

        # Inverse-volatility
        w_iv = inverse_vol_weights(R)
        w_iv = w_iv / w_iv.sum() if w_iv.sum() > 0 else w_iv

        te_w_hist[t0], ew_w_hist[t0], iv_w_hist[t0] = w_te, w_ew, w_iv

        # Realize returns over (t0, t1]
        win = (r_stk.index > t0) & (r_stk.index <= t1)
        ret_window = r_stk.loc[win, R.columns]
        if ret_window.empty:
            continue

        ret_te.append((ret_window * w_te.reindex(ret_window.columns, fill_value=0.0)).sum(axis=1))
        ret_ew.append((ret_window * w_ew.reindex(ret_window.columns, fill_value=0.0)).sum(axis=1))
        ret_iv.append((ret_window * w_iv.reindex(ret_window.columns, fill_value=0.0)).sum(axis=1))

    # Combine daily returns
    ret_te = pd.concat(ret_te).sort_index() if ret_te else pd.Series(dtype=float)
    ret_ew = pd.concat(ret_ew).sort_index() if ret_ew else pd.Series(dtype=float)
    ret_iv = pd.concat(ret_iv).sort_index() if ret_iv else pd.Series(dtype=float)

    # Align with index
    idx_ret = r_idx.reindex(ret_te.index.union(ret_ew.index).union(ret_iv.index)).dropna()
    te_al = ret_te.reindex(idx_ret.index)
    ew_al = ret_ew.reindex(idx_ret.index)
    iv_al = ret_iv.reindex(idx_ret.index)

    # Report
    def report(name: str, pr: pd.Series):
        stats = perf_stats(pr)
        active = pr - idx_ret
        te = tracking_error(active)
        ir = info_ratio(active)
        print(f"{name:>18}: CAGR {stats['CAGR']:.2%} | Vol {stats['Vol']:.2%} | "
              f"TE {te:.2%} | IR {ir:.2f} | MaxDD {stats['MaxDD']:.2%}")

    print("\n=== Performance vs. RGUSTSC (daily) ===")
    report("TE-minimizer", te_al)
    report("Equal-weight", ew_al)
    report("Inv-Vol (1/σ)", iv_al)

    # Save outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(os.path.join(cfg.out_dir, "members_hardcoded.csv"), index=False)
    te_al.to_csv(os.path.join(cfg.out_dir, "returns_tracking_te.csv"))
    ew_al.to_csv(os.path.join(cfg.out_dir, "returns_equal_weight.csv"))
    iv_al.to_csv(os.path.join(cfg.out_dir, "returns_inverse_vol.csv"))
    idx_ret.to_csv(os.path.join(cfg.out_dir, "returns_index.csv"))
    pd.DataFrame(te_w_hist).T.to_csv(os.path.join(cfg.out_dir, "weights_te_by_rebalance.csv"))
    pd.DataFrame(ew_w_hist).T.to_csv(os.path.join(cfg.out_dir, "weights_ew_by_rebalance.csv"))
    pd.DataFrame(iv_w_hist).T.to_csv(os.path.join(cfg.out_dir, "weights_iv_by_rebalance.csv"))

    # Plots
    if cfg.save_plots and HAVE_PLT:
        try:
            eq_idx = (1 + idx_ret).cumprod()
            eq_te  = (1 + te_al.fillna(0)).cumprod()
            eq_ew  = (1 + ew_al.fillna(0)).cumprod()
            eq_iv  = (1 + iv_al.fillna(0)).cumprod()

            plt.figure(figsize=(10,6))
            eq_idx.plot(label=cfg.index_ticker)
            eq_te.plot(label="TE-minimizer")
            eq_ew.plot(label="Equal-weight")
            eq_iv.plot(label="Inv-Vol (1/σ)")
            plt.title(f"RGUSTSC Tracking / Smart Beta")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "equity_curves.png"), dpi=140)
            plt.close()

            plt.figure(figsize=(9,5))
            (te_al - idx_ret).hist(bins=60, alpha=0.5, label="TE-min - Index")
            (ew_al - idx_ret).hist(bins=60, alpha=0.5, label="EW - Index")
            (iv_al - idx_ret).hist(bins=60, alpha=0.5, label="IVOL - Index")
            plt.title("Active Return Distributions vs. RGUSTSC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "active_return_hist.png"), dpi=140)
            plt.close()

            print(f"Saved plots to: {os.path.abspath(cfg.out_dir)}")
        except Exception as e:
            print("Plotting skipped:", repr(e))

    # Done
    bbg.stop()
    print(f"Files saved to: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
