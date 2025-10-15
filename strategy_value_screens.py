#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value Investing Screens — Long Cheap / Short Expensive (Monthly Rebalance)
Data Source: Bloomberg Desktop API via blpapi (STRICTLY no pdblp)

Strategy:
- Composite value score = sector-relative z-score average of [E/P, B/M, EBITDA/EV]
  where E/P = 1/PE_RATIO, B/M = 1/PX_TO_BOOK_RATIO, EBITDA/EV = 1/EV_TO_EBITDA
- Long top decile (cheapest), short bottom decile (most expensive)
- Monthly rebalance, 1-month lag on fundamentals to avoid look-ahead
- Dollar-neutral, equal-weight within sides, turnover costs applied

Outputs:
- CSV files: value scores, deciles, weights, monthly returns (gross/net), benchmark returns

Author: (Your Name)
Date: 2025-09-24
"""

import os
import sys
import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Bloomberg API
import blpapi

# =========================
# Config
# =========================
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Universe (replace with your research universe or historical index members)
    universe: List[str] = None

    # Date range
    start_date: str = "2018-01-01"
    end_date: str = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Fundamental fields (ratios)
    field_pe: str = "PE_RATIO"
    field_pb: str = "PX_TO_BOOK_RATIO"
    field_evebitda: str = "BEST_EV_TO_BEST_EBITDA"

    # Sector field (reference)
    sector_field: str = "GICS_SECTOR_NAME"

    # Pricing (prefer total return; fallback to price)
    tri_field: str = "TOT_RETURN_INDEX_GROSS_DVDS"
    px_field: str = "PX_LAST"

    # Historical request options
    periodicity: str = "DAILY"                    # DAILY | WEEKLY | MONTHLY (we'll resample monthly ourselves)
    fill_option: str = "ALL_CALENDAR_DAYS"        # or "ACTIVE_DAYS_ONLY"
    fill_method: str = "PREVIOUS_VALUE"           # or "NIL_VALUE"

    # Rebalance & lags
    rebalance_freq: str = "M"
    fundamental_lag_months: int = 1

    # Portfolio construction
    deciles: int = 10
    long_bucket: int = 10
    short_bucket: int = 1
    max_weight_per_name: float = 0.03
    transaction_cost_bps: float = 10.0

    # Output
    out_dir: str = "./out_value_screens_blpapi"
    save_plots: bool = False

cfg = Config()
if cfg.universe is None:
    # Demo: large, liquid US names (replace with your own or SPX historical members)
    cfg.universe = [
        "AAPL US Equity","MSFT US Equity","AMZN US Equity","GOOGL US Equity","META US Equity",
        "NVDA US Equity","BRK/B US Equity","JPM US Equity","XOM US Equity","JNJ US Equity",
        "V US Equity","PG US Equity","UNH US Equity","HD US Equity","MA US Equity",
        "PFE US Equity","CVX US Equity","KO US Equity","PEP US Equity","CSCO US Equity",
        "ABBV US Equity","MRK US Equity","BAC US Equity","WMT US Equity","DIS US Equity",
        "NFLX US Equity","ADBE US Equity","INTC US Equity","T US Equity","VZ US Equity",
        "ORCL US Equity","CRM US Equity","MCD US Equity","NKE US Equity","WFC US Equity",
        "COST US Equity","TMO US Equity","AVGO US Equity","TXN US Equity","LIN UN Equity",
        "QCOM US Equity","PM US Equity","ACN UN Equity","IBM US Equity","AMD US Equity",
        "AMGN US Equity","HON US Equity","CAT US Equity","GS US Equity","BLK US Equity",
    ]

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)


# =========================
# Bloomberg Session Helpers
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
        """Send request and synchronously collect all messages for this request."""
        cid = self.session.sendRequest(request)
        msgs = []
        while True:
            ev = self.session.nextEvent(self.timeout_ms)
            for msg in ev:
                # Filter by correlationId
                for rcid in msg.correlationIds():
                    if rcid == cid:
                        msgs.append(msg)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        return msgs

    # ---------- Reference Data ----------
    def refdata(self, securities: List[str], fields: List[str]) -> pd.DataFrame:
        req = self.ref_service.createRequest("ReferenceDataRequest")
        sec_el = req.getElement("securities")
        for s in securities:
            sec_el.appendValue(s)
        fld_el = req.getElement("fields")
        for f in fields:
            fld_el.appendValue(f)

        msgs = self._send_request(req)
        rows = []
        for msg in msgs:
            if msg.hasElement("responseError"):
                raise RuntimeError(f"ReferenceDataRequest error: {msg.getElement('responseError')}")
            if not msg.hasElement("securityData"):
                continue
            sec_data_array = msg.getElement("securityData")
            for i in range(sec_data_array.numValues()):
                sdata = sec_data_array.getValueAsElement(i)
                if sdata.hasElement("securityError"):
                    rows.append({"ticker": sdata.getElementAsString("security"),
                                 "error": sdata.getElement("securityError").getElementAsString("message")})
                    continue
                sec = sdata.getElementAsString("security")
                fdata = sdata.getElement("fieldData")
                row = {"ticker": sec}
                for f in fields:
                    if fdata.hasElement(f):
                        try:
                            row[f] = fdata.getElementAsString(f)
                        except Exception:
                            try:
                                row[f] = fdata.getElementAsFloat(f)
                            except Exception:
                                row[f] = None
                    else:
                        row[f] = None
                rows.append(row)
        df = pd.DataFrame(rows).set_index("ticker")
        return df

    # ---------- Historical Data (single field) ----------
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
        Fetch historical data for a SINGLE field across multiple securities.
        Returns a wide DataFrame: index=date, columns=tickers.
        """
        # Bloomberg requires YYYYMMDD format
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

        # Collect per-security series
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
                # date is required
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
            print("HistoricalDataRequest failed tickers (first few):", bad[:5])

        if not frames:
            raise RuntimeError(f"No historical data returned for field {field}")

        # Merge all series by index (date)
        df = None
        for k, sdf in frames.items():
            df = sdf if df is None else df.join(sdf, how="outer")
        # columns currently are securities; we want columns=tickers
        df.columns = [c for c in df.columns]
        return df.sort_index()


# =========================
# Utilities
# =========================
def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("M").last()

def winsorize_cs(x: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    if x.dropna().empty:
        return x
    lo, hi = x.quantile(p_low), x.quantile(p_high)
    return x.clip(lower=lo, upper=hi)

def zscore_cs(x: pd.Series) -> pd.Series:
    m, s = x.mean(), x.std(ddof=1)
    if s is None or s == 0 or np.isnan(s):
        return x * 0.0
    return (x - m) / s

def equal_weight_sides(longs: List[str], shorts: List[str], per_name_cap: float) -> pd.Series:
    w = {}
    if longs:
        wl = min(1.0 / len(longs), per_name_cap)
        for t in longs: w[t] = wl
        s = sum(w[t] for t in longs)
        for t in longs: w[t] /= s
    if shorts:
        ws = min(1.0 / len(shorts), per_name_cap)
        for t in shorts: w[t] = -ws
        s = -sum(w[t] for t in shorts)
        for t in shorts: w[t] /= s
    return pd.Series(w, dtype=float)

def portfolio_turnover_cost(w_prev: pd.Series, w_new: pd.Series, tc_bps: float) -> float:
    aligned = pd.concat([w_prev.fillna(0), w_new.fillna(0)], axis=1)
    aligned.columns = ["prev", "new"]
    turnover = (aligned["new"] - aligned["prev"]).abs().sum()
    return float((tc_bps / 10000.0) * turnover)

def compute_perf_stats(series: pd.Series) -> Dict[str, float]:
    r = series.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    periods = 12
    cum = (1 + r).prod()
    years = len(r) / periods
    cagr = cum ** (1 / max(years, 1e-9)) - 1
    vol = r.std(ddof=1) * math.sqrt(periods)
    sharpe = (r.mean() * periods) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(dd.min())}

def decile_buckets(scores_row: pd.Series, deciles: int) -> pd.Series:
    s = scores_row.dropna()
    if len(s) < deciles:
        return pd.Series(index=scores_row.index, dtype=float)
    ranks = s.rank(method="first", ascending=True)
    buckets = pd.qcut(ranks, deciles, labels=False) + 1
    out = pd.Series(index=scores_row.index, dtype=float)
    out.loc[s.index] = buckets.values
    return out


# =========================
# Value Signal Engine
# =========================
def build_value_scores(
    pe_m: pd.DataFrame,
    pb_m: pd.DataFrame,
    evebitda_m: pd.DataFrame,
    sectors: pd.Series,
    lag_months: int
) -> pd.DataFrame:
    # Yields (cheaper = larger)
    ep = 1.0 / pe_m.replace(0, np.nan)
    bm = 1.0 / pb_m.replace(0, np.nan)
    ebitda_over_ev = 1.0 / evebitda_m.replace(0, np.nan)

    # Lag to avoid look-ahead
    ep = ep.shift(lag_months)
    bm = bm.shift(lag_months)
    ebitda_over_ev = ebitda_over_ev.shift(lag_months)

    # Composite per month with sector-relative z-scores
    idx = ep.index
    tickers = ep.columns
    sectors = sectors.reindex(tickers)

    scores = pd.DataFrame(index=idx, columns=tickers, dtype=float)

    for dt in idx:
        ep_cs = ep.loc[dt]
        bm_cs = bm.loc[dt]
        eoe_cs = ebitda_over_ev.loc[dt]

        # Winsorize within sector
        ep_w = ep_cs.groupby(sectors).apply(winsorize_cs)
        bm_w = bm_cs.groupby(sectors).apply(winsorize_cs)
        eoe_w = eoe_cs.groupby(sectors).apply(winsorize_cs)

        # Z-scores within sector
        z_ep = ep_w.groupby(sectors).apply(zscore_cs)
        z_bm = bm_w.groupby(sectors).apply(zscore_cs)
        z_eoe = eoe_w.groupby(sectors).apply(zscore_cs)

        comp = pd.concat([z_ep, z_bm, z_eoe], axis=1)
        comp.columns = ["z_ep", "z_bm", "z_eoe"]
        scores.loc[dt, comp.index] = comp.mean(axis=1, skipna=True).values

    return scores


# =========================
# Main Backtest
# =========================
def run_backtest():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Start Bloomberg
    bb = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bb.start()
    print("Connected to Bloomberg.")

    # 2) Reference sectors
    print("Fetching GICS sectors (reference)...")
    ref = bb.refdata(cfg.universe, [cfg.sector_field])
    sectors = ref[cfg.sector_field].fillna("Unknown")
    # normalize to strings
    sectors = sectors.astype(str)

    # 3) Prices — TR index preferred, fallback to PX_LAST
    print("Fetching price/total-return series...")
    tri = bb.bdh_single_field(cfg.universe, cfg.tri_field, cfg.start_date, cfg.end_date,
                              periodicity=cfg.periodicity,
                              fill_option=cfg.fill_option,
                              fill_method=cfg.fill_method)
    px = bb.bdh_single_field(cfg.universe, cfg.px_field, cfg.start_date, cfg.end_date,
                             periodicity=cfg.periodicity,
                             fill_option=cfg.fill_option,
                             fill_method=cfg.fill_method)

    # Heuristic fallback if TR coverage poor
    tri_coverage = tri.notna().mean().mean() if not tri.empty else 0.0
    prices = tri if tri_coverage > 0.5 else px
    if tri_coverage <= 0.5:
        print("Warning: TRI coverage low; using PX_LAST for returns.")

    # Monthly price levels & returns
    mpx = to_month_end(prices).ffill()
    rets = mpx.pct_change()

    # 4) Fundamentals (ratios) — daily to monthly last, then lag
    print("Fetching valuation ratios (PE, PB, EV/EBITDA)...")
    pe_d = bb.bdh_single_field(cfg.universe, cfg.field_pe, cfg.start_date, cfg.end_date,
                               periodicity=cfg.periodicity,
                               fill_option=cfg.fill_option,
                               fill_method=cfg.fill_method)
    pb_d = bb.bdh_single_field(cfg.universe, cfg.field_pb, cfg.start_date, cfg.end_date,
                               periodicity=cfg.periodicity,
                               fill_option=cfg.fill_option,
                               fill_method=cfg.fill_method)
    evebitda_d = bb.bdh_single_field(cfg.universe, cfg.field_evebitda, cfg.start_date, cfg.end_date,
                                     periodicity=cfg.periodicity,
                                     fill_option=cfg.fill_option,
                                     fill_method=cfg.fill_method)

    pe_m = to_month_end(pe_d).ffill()
    pb_m = to_month_end(pb_d).ffill()
    evebitda_m = to_month_end(evebitda_d).ffill()

    # 5) Value scores (sector-relative)
    print("Building composite value scores...")
    scores = build_value_scores(pe_m, pb_m, evebitda_m, sectors, cfg.fundamental_lag_months)

    # 6) Rebalance dates (intersection and drop first NaN diff)
    dates = scores.index.intersection(rets.index).sort_values()
    if len(dates) < 24:
        print("Warning: very short sample; extend date range.")
    dates = dates[1:]  # first return is NaN

    # 7) Backtest loop (monthly)
    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    deciles_hist: Dict[pd.Timestamp, pd.Series] = {}
    ls_gross, ls_net = [], []
    w_prev = pd.Series(dtype=float)

    for i in range(1, len(dates)):
        t0 = dates[i-1]
        t1 = dates[i]

        # Deciles at t0
        decs = decile_buckets(scores.loc[t0], cfg.deciles)
        deciles_hist[t0] = decs

        longs = decs.index[decs == cfg.long_bucket].tolist()
        shorts = decs.index[decs == cfg.short_bucket].tolist()

        # Equal-weight long/short with per-name cap
        w_new = equal_weight_sides(longs, shorts, cfg.max_weight_per_name)

        # Turnover cost at t0
        tc = portfolio_turnover_cost(w_prev, w_new, cfg.transaction_cost_bps)

        # Realized return over t0->t1
        next_r = rets.loc[t1].reindex(w_new.index).fillna(0.0)
        gross = float((w_new * next_r).sum())
        net = gross - tc

        ls_gross.append(pd.Series({t1: gross}))
        ls_net.append(pd.Series({t1: net}))
        weights_hist[t0] = w_new
        w_prev = w_new

    ls_gross = pd.concat(ls_gross).sort_index()
    ls_net = pd.concat(ls_net).sort_index()

    # 8) Benchmark (SPY)
    print("Fetching benchmark SPY...")
    spy_px = bb.bdh_single_field(["SPY US Equity"], cfg.px_field, cfg.start_date, cfg.end_date,
                                 periodicity=cfg.periodicity,
                                 fill_option=cfg.fill_option,
                                 fill_method=cfg.fill_method)
    spy_mpx = to_month_end(spy_px).ffill()
    spy_ret = spy_mpx.pct_change().iloc[:, 0].reindex(ls_net.index).fillna(0.0)

    # 9) Performance stats
    stats_gross = compute_perf_stats(ls_gross)
    stats_net = compute_perf_stats(ls_net)
    stats_spy = compute_perf_stats(spy_ret)

    # 10) Save outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    scores.to_csv(os.path.join(cfg.out_dir, "value_scores_composite.csv"))
    pd.DataFrame({k: v for k, v in deciles_hist.items()}).T.to_csv(os.path.join(cfg.out_dir, "deciles.csv"))
    pd.DataFrame({k: v for k, v in weights_hist.items()}).T.to_csv(os.path.join(cfg.out_dir, "weights_longshort.csv"))
    ls_gross.to_csv(os.path.join(cfg.out_dir, "ls_returns_gross_monthly.csv"))
    ls_net.to_csv(os.path.join(cfg.out_dir, "ls_returns_net_monthly.csv"))
    spy_ret.to_csv(os.path.join(cfg.out_dir, "spy_returns_monthly.csv"))
    mpx.to_csv(os.path.join(cfg.out_dir, "prices_monthly_used.csv"))

    # 11) Print summary
    def fmt(d):
        return f"CAGR {d['CAGR']:.2%} | Vol {d['Vol']:.2%} | Sharpe {d['Sharpe']:.2f} | MaxDD {d['MaxDD']:.2%}"

    print("\n=== Value Screens L/S (Monthly) — blpapi ===")
    print(f"Universe: {len(cfg.universe)} names | Deciles: {cfg.deciles} | Long D{cfg.long_bucket} vs Short D{cfg.short_bucket}")
    print(f"Lag (months): {cfg.fundamental_lag_months} | TC: {cfg.transaction_cost_bps:.1f} bps | Cap/name: {cfg.max_weight_per_name:.2%}")
    print("Gross:", fmt(stats_gross))
    print("Net:  ", fmt(stats_net))
    print("\n=== Benchmark (SPY) ===")
    print("SPY:  ", fmt(stats_spy))
    print(f"\nFiles saved to: {os.path.abspath(cfg.out_dir)}")

    # 12) Optional plot
    if cfg.save_plots:
        try:
            import matplotlib.pyplot as plt
            eq_ls = (1 + ls_net).cumprod()
            eq_spy = (1 + spy_ret).cumprod()
            plt.figure(figsize=(10,6))
            eq_ls.plot(label="Value L/S (net)")
            eq_spy.plot(label="SPY")
            plt.title("Equity Curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "equity_curves.png"), dpi=140)
            plt.close()
            print("Saved plot: equity_curves.png")
        except Exception as e:
            print("Plot skipped:", e)

    # 13) Clean shutdown
    bb.stop()


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    try:
        run_backtest()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        # Surface useful error details
        print("Error:", repr(e))
        sys.exit(1)
