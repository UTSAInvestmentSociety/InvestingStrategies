#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Investing (Fama–French/Carhart style) — Demonstration Script
STRICTLY via Bloomberg blpapi (no pdblp)

What this does
--------------
1) Pulls Bloomberg history (prices + fundamentals) for a user universe.
2) Builds monthly factor-mimicking portfolios for Size (SMB), Value (HML), Momentum (UMD), Quality (QLTY).
   - Size  : long Small (low mkt cap) vs short Big (high mkt cap)
   - Value : long High B/M (low P/B) vs short Low B/M
   - Momentum: long winners (12-1) vs short losers
   - Quality : long high ROE vs short low ROE
3) Computes monthly factor returns (equal-weight long minus short).
4) Runs rolling time-series regressions of asset returns on factor returns (factor loadings).
5) Tests simple long–short strategies sorted by estimated factor betas (e.g., long high-UMD beta vs short low-UMD beta).
6) Saves CSV outputs and prints basic performance stats.

Notes
-----
- Uses 1-month lag for fundamentals to avoid look-ahead.
- Uses total-return index if available; otherwise falls back to PX_LAST.
- Risk-free assumed 0 for simplicity (you can add T-bill later).
- Educational demonstration; not investment advice.

Requirements
------------
- Bloomberg Terminal running (Desktop API enabled)
- Python packages: blpapi, pandas, numpy, matplotlib (optional)

Install:
    pip install blpapi pandas numpy matplotlib
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
    # Bloomberg session
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Universe (replace with your own or historical index members)
    universe: List[str] = None

    # Dates
    start_date: str = "2014-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Bloomberg fields
    tri_field: str = "TOT_RETURN_INDEX_GROSS_DVDS"  # preferred
    px_field: str  = "PX_LAST"                      # fallback for prices
    mktcap_field: str = "CUR_MKT_CAP"               # market cap (historical)
    pb_field: str = "PX_TO_BOOK_RATIO"              # for value (we use 1/PB)
    roe_field: str = "RETURN_COM_EQY"             # for quality (ROE %)

    # Historical request options
    periodicity: str = "DAILY"
    fill_option: str = "ALL_CALENDAR_DAYS"
    fill_method: str = "PREVIOUS_VALUE"

    # Factor construction
    quantiles: int = 5           # quintiles
    long_bucket: int = 5         # long top quintile (cheapest/smallest/best)
    short_bucket: int = 1        # short bottom quintile
    fundamental_lag_months: int = 1

    # Rolling regressions
    beta_window_months: int = 36 # regression lookback window

    # Beta-sorted L/S demo
    build_beta_sorted_ls: bool = True
    beta_sorted_factor: str = "UMD" # which factor's beta to sort on
    beta_sorted_quantiles: int = 5

    # Output
    out_dir: str = "./out_factor_investing_blpapi"
    save_plots: bool = True

cfg = Config()
if cfg.universe is None:
    # diversified set of US large caps for the demo
    cfg.universe = [
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
                    # Try float then string
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

def winsorize_cs(x: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    if x.dropna().empty:
        return x
    lo, hi = x.quantile(p_low), x.quantile(p_high)
    return x.clip(lower=lo, upper=hi)

def bucketize(s: pd.Series, q: int) -> pd.Series:
    """Return quantile buckets (1..q)."""
    x = s.dropna()
    if len(x) < q:
        return pd.Series(index=s.index, dtype=float)
    ranks = x.rank(method="first")
    buckets = pd.qcut(ranks, q, labels=False) + 1
    out = pd.Series(index=s.index, dtype=float)
    out.loc[x.index] = buckets.values
    return out

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
    print(f"{name:>8}: CAGR {d['CAGR']:.2%} | Vol {d['Vol']:.2%} | Sharpe {d['Sharpe']:.2f} | MaxDD {d['MaxDD']:.2%}")


# =========================
# Factor construction
# =========================
def build_factors(
    rets_m: pd.DataFrame,
    mktcap_m: pd.DataFrame,
    pb_m: pd.DataFrame,
    roe_m: pd.DataFrame,
    lag_m: int,
    q: int,
    long_bucket: int,
    short_bucket: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[pd.Timestamp, pd.Series]]]:
    """
    Build SMB, HML, UMD, QLTY factor returns (monthly) using equal-weight long-short.
    - Size signal: -log(MarketCap)
    - Value signal: 1 / P/B
    - Momentum signal: 12-1 return
    - Quality signal: ROE
    All signals are lagged by 'lag_m' months.
    Returns factor_returns DF and dict of long/short members per factor/date.
    """
    dates = rets_m.index
    tickers = rets_m.columns

    # Signals (monthly, lagged)
    size_signal = (-np.log(mktcap_m.replace(0, np.nan))).shift(lag_m)
    value_signal = (1.0 / pb_m.replace(0, np.nan)).shift(lag_m)
    # 12-1 momentum computed from price returns
    mom_level = (1 + rets_m).cumprod().replace(0, np.nan)
    mom_signal = (mom_level.shift(1) / mom_level.shift(12)) - 1.0  # as of t: (t-1)/(t-12)
    mom_signal = mom_signal.shift(lag_m)  # add reporting lag
    quality_signal = roe_m.shift(lag_m)

    # Containers
    fact_rets = []
    membership: Dict[str, Dict[pd.Timestamp, pd.Series]] = {
        "SMB": {}, "HML": {}, "UMD": {}, "QLTY": {}
    }

    for i in range(1, len(dates)):
        t0 = dates[i-1]
        t1 = dates[i]
        r1 = rets_m.loc[t1]

        # Build buckets at t0 for each factor
        for fac, sig_df in [
            ("SMB", size_signal),
            ("HML", value_signal),
            ("UMD", mom_signal),
            ("QLTY", quality_signal),
        ]:
            s = sig_df.loc[t0]
            b = bucketize(s, q)
            longs = b.index[b == long_bucket]
            shorts = b.index[b == short_bucket]

            # Equal-weight long/short returns over t0->t1
            long_ret = float(r1.reindex(longs).mean()) if len(longs) > 0 else np.nan
            short_ret = float(r1.reindex(shorts).mean()) if len(shorts) > 0 else np.nan
            ls = long_ret - short_ret if (np.isfinite(long_ret) and np.isfinite(short_ret)) else np.nan

            fact_rets.append(pd.Series({("SMB" if fac=="SMB" else fac): ls}, index=[fac], name=t1))

            membership[fac][t0] = pd.Series(1, index=longs).reindex(tickers, fill_value=0) - \
                                  pd.Series(1, index=shorts).reindex(tickers, fill_value=0)

    # Combine factor returns
    fac_df = pd.concat(fact_rets, axis=1).T
    fac_df.index = [r.Index for r in fac_df.itertuples(index=True)]

    # Build proper DataFrame with columns
    out = pd.DataFrame(index=rets_m.index, columns=["SMB","HML","UMD","QLTY"], dtype=float)
    for idx, row in fac_df.iterrows():
        # idx is timestamp (t1); row has a single non-null value in its factor label
        for col in out.columns:
            if not pd.isna(row.get(col, np.nan)):
                out.loc[idx, col] = row[col]
    out = out.sort_index()
    return out, membership


# =========================
# Rolling factor loadings (time-series regression)
# =========================
def rolling_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: int
) -> Dict[str, pd.DataFrame]:
    """
    For each asset, run rolling OLS: r_i = a + b_SMB*SMB + b_HML*HML + b_UMD*UMD + b_QLTY*QLTY
    Returns dict of betas {factor: DataFrame dates x tickers}.
    """
    common_dates = asset_returns.index.intersection(factor_returns.index)
    R = asset_returns.reindex(common_dates).copy()
    F = factor_returns.reindex(common_dates).copy()

    betas = {fac: pd.DataFrame(index=common_dates, columns=R.columns, dtype=float)
             for fac in F.columns}

    X_full = F.assign(const=1.0).values  # we'll slice per window
    cols = list(F.columns) + ["const"]

    for i in range(window, len(common_dates)):
        idx_slice = slice(i-window, i)  # [i-window, i)
        X = X_full[idx_slice, :]        # T x (k+1)
        # Drop rows with any NaN in factors
        valid_rows = ~np.isnan(X).any(axis=1)
        if valid_rows.sum() < window * 0.8:
            continue
        X = X[valid_rows, :]
        # Factors without const
        Xfac = X[:, :-1]  # T x k
        for j, tkr in enumerate(R.columns):
            y = R.iloc[idx_slice, j].values[valid_rows]
            if np.isnan(y).any() or len(y) != Xfac.shape[0]:
                continue
            # OLS via lstsq on [const + factors] then read factor coefs
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                # coef order matches cols: [factors..., const]
                for k, fac in enumerate(F.columns):
                    betas[fac].iloc[i, j] = coef[k]
            except Exception:
                continue

    return betas


# =========================
# Main
# =========================
def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Connect & fetch data
    bb = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bb.start()
    print("Connected to Bloomberg.")

    print("Fetching prices (TRI preferred) ...")
    tri = bb.bdh_single_field(cfg.universe, cfg.tri_field, cfg.start_date, cfg.end_date,
                              periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    px  = bb.bdh_single_field(cfg.universe, cfg.px_field,  cfg.start_date, cfg.end_date,
                              periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)

    tri_cov = tri.notna().mean().mean() if not tri.empty else 0.0
    prices = tri if tri_cov > 0.5 else px
    if tri_cov <= 0.5:
        print("Warning: TRI coverage low; using PX_LAST.")

    print("Fetching fundamentals (MktCap, P/B, ROE) ...")
    mktcap_d = bb.bdh_single_field(cfg.universe, cfg.mktcap_field, cfg.start_date, cfg.end_date,
                                   periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    pb_d     = bb.bdh_single_field(cfg.universe, cfg.pb_field, cfg.start_date, cfg.end_date,
                                   periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)
    roe_d    = bb.bdh_single_field(cfg.universe, cfg.roe_field, cfg.start_date, cfg.end_date,
                                   periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method)

    # 2) Monthly resampling
    rets_m   = monthly_returns_from_prices(prices).iloc[1:]  # drop first NaN
    mktcap_m = to_month_end(mktcap_d).ffill()
    pb_m     = to_month_end(pb_d).ffill()
    # ROE often in %; convert to decimal if needed
    roe_m    = to_month_end(roe_d).ffill()
    if roe_m.abs().median(numeric_only=True).median() > 1.5:  # crude check: likely % units
        roe_m = roe_m / 100.0

    # 3) Build factor portfolios
    print("Constructing factor-mimicking portfolios (SMB, HML, UMD, QLTY) ...")
    factors, members = build_factors(
        rets_m=rets_m,
        mktcap_m=mktcap_m,
        pb_m=pb_m,
        roe_m=roe_m,
        lag_m=cfg.fundamental_lag_months,
        q=cfg.quantiles,
        long_bucket=cfg.long_bucket,
        short_bucket=cfg.short_bucket,
    )

    # Align
    factors = factors.dropna(how="all").fillna(0.0)
    rets_m  = rets_m.reindex(factors.index).fillna(0.0)

    # 4) Factor performance
    print("\n=== Factor return stats (monthly) ===")
    for fac in ["SMB","HML","UMD","QLTY"]:
        print_stats(fac, perf_stats(factors[fac]))

    # 5) Rolling factor loadings (betas)
    print("\nEstimating rolling factor loadings (betas) ...")
    betas = rolling_factor_betas(asset_returns=rets_m, factor_returns=factors, window=cfg.beta_window_months)
    # Save last-available betas snapshot
    last_date = betas["SMB"].index.max()
    beta_snapshot = pd.concat({
        fac: df.loc[last_date] for fac, df in betas.items() if last_date in df.index
    }, axis=1)
    beta_snapshot.to_csv(os.path.join(cfg.out_dir, "factor_betas_last_snapshot.csv"))

    # 6) Beta-sorted long–short demo (e.g., by UMD beta)
    beta_ls_series = pd.Series(dtype=float)
    if cfg.build_beta_sorted_ls:
        fac = cfg.beta_sorted_factor
        print(f"\nBuilding beta-sorted L/S strategy on {fac} betas ...")
        # For each month t (after we have a beta estimate), sort assets by beta and hold next month
        bdf = betas[fac]
        dates = bdf.index
        for i in range(cfg.beta_window_months, len(dates)-1):
            t0 = dates[i]
            t1 = dates[i+1]
            b = bdf.loc[t0].dropna()
            if len(b) < cfg.beta_sorted_quantiles:
                continue
            # buckets
            ranks = b.rank(method="first")
            q = cfg.beta_sorted_quantiles
            labels = pd.qcut(ranks, q, labels=False) + 1
            longs = b.index[labels == q]
            shorts = b.index[labels == 1]
            # next-month return
            r1 = rets_m.loc[t1]
            long_ret = float(r1.reindex(longs).mean()) if len(longs) else np.nan
            short_ret = float(r1.reindex(shorts).mean()) if len(shorts) else np.nan
            if np.isfinite(long_ret) and np.isfinite(short_ret):
                beta_ls_series.loc[t1] = long_ret - short_ret

        if not beta_ls_series.empty:
            print_stats(f"{fac}-β L/S", perf_stats(beta_ls_series))
            beta_ls_series.to_csv(os.path.join(cfg.out_dir, f"beta_sorted_ls_{fac}.csv"))

    # 7) Persist core outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    rets_m.to_csv(os.path.join(cfg.out_dir, "asset_returns_monthly.csv"))
    factors.to_csv(os.path.join(cfg.out_dir, "factor_returns_monthly.csv"))
    for fac, hist in members.items():
        # Save long-minus-short membership (1 for long names, -1 for short) per rebalance date
        mem_df = pd.DataFrame(hist).T.sort_index()
        mem_df.to_csv(os.path.join(cfg.out_dir, f"{fac}_memberships.csv"))

    # 8) Optional plots
    if cfg.save_plots and HAVE_PLT:
        try:
            # Factor equity curves
            plt.figure(figsize=(10,6))
            for fac in ["SMB","HML","UMD","QLTY"]:
                ((1 + factors[fac].fillna(0)).cumprod()).plot(label=fac)
            plt.title("Factor Equity Curves (EW long–short)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "factor_equity_curves.png"), dpi=140)
            plt.close()

            if not beta_ls_series.empty:
                plt.figure(figsize=(9,5))
                (1 + beta_ls_series.fillna(0)).cumprod().plot(label=f"{cfg.beta_sorted_factor}-β L/S")
                plt.title(f"Beta-Sorted L/S on {cfg.beta_sorted_factor} Betas")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.out_dir, f"beta_sorted_ls_{cfg.beta_sorted_factor}_equity.png"), dpi=140)
                plt.close()

            print(f"Saved plots to: {os.path.abspath(cfg.out_dir)}")
        except Exception as e:
            print("Plotting skipped:", repr(e))

    # Close session
    bb.stop()
    print(f"\nFiles saved to: {os.path.abspath(cfg.out_dir)}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
