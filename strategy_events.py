#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Study Strategies — Dividend Ex-Date Demo (strictly Bloomberg blpapi)

Idea
-----
Measure abnormal returns around corporate events using Bloomberg data.
This demo implements an event study around **dividend ex-dates** pulled from
Bloomberg's corporate action bulk field `DVD_HIST_ALL`. It:
  1) fetches daily prices for a ticker universe and a market proxy (SPY),
  2) fetches each ticker's dividend history (ex-dates),
  3) builds event windows ([-10, +10] trading days),
  4) estimates expected returns via a pre-event CAPM (alpha/beta) on [-250, -20],
  5) computes Abnormal Returns (AR_t) and Cumulative Abnormal Returns (CAR),
  6) aggregates Average AR (AAR) and Average CAR (ACAR) across events.

You can adapt the same scaffolding for other event types (earnings, M&A) by
replacing the event fetcher with the appropriate Bloomberg bulk/history fields.

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

# Optional plotting
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
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 120000

    # Study setup
    universe: List[str] = None         # tickers to study
    market_proxy: str = "SPY US Equity"

    start_date: str = "2015-01-01"
    end_date: str   = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Price fields / BDH options
    price_field: str = "PX_LAST"
    periodicity: str = "DAILY"              # DAILY | WEEKLY | MONTHLY
    fill_option: str = "ACTIVE_DAYS_ONLY"   # do not fabricate weekends
    fill_method: str = "NIL_VALUE"          # don't forward-fill price for returns

    # Event window & estimation window (trading days)
    event_L: int = 10       # days before event (negative side)
    event_R: int = 10       # days after event (positive side)
    est_pre: int = 250      # size of estimation window BEFORE event (e.g., 250 trading days)
    est_gap: int = 20       # gap between estimation window end and event (avoid leakage)

    # Dividend filter
    min_div_amt: float = 0.0   # filter tiny micro-dividends if desired

    # Output
    out_dir: str = "./out_event_study_blpapi"
    save_plots: bool = True

cfg = Config()
if cfg.universe is None:
    # Large, liquid US names for demonstration
    cfg.universe = [
        "AAPL US Equity","MSFT US Equity","JPM US Equity","XOM US Equity","JNJ US Equity",
        "PG US Equity","KO US Equity","PEP US Equity","CVX US Equity","PFE US Equity",
        "WMT US Equity","VZ US Equity","MCD US Equity","IBM US Equity","T US Equity",
    ]


# =========================
# Bloomberg (strict blpapi)
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
        fill_option: str = "ACTIVE_DAYS_ONLY",
        fill_method: str = "NIL_VALUE",
    ) -> pd.DataFrame:
        """HistoricalDataRequest for a SINGLE field across multiple securities -> wide DataFrame."""
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
                frames[sec] = pd.DataFrame(recs, columns=["date", sec]).set_index("date").sort_index()

        if bad:
            print("BDH failed tickers (first few):", bad[:5])
        if not frames:
            raise RuntimeError(f"No historical data returned for field {field}")

        df = None
        for _, sdf in frames.items():
            df = sdf if df is None else df.join(sdf, how="outer")
        return df.sort_index()

    def ref_bulk(self, tickers: List[str], field: str) -> Dict[str, List[Dict[str, str]]]:
        """
        ReferenceDataRequest for a single **bulk** field (e.g., 'DVD_HIST_ALL').
        Returns dict: {ticker: [row_dict, ...]}
        """
        req = self.ref_service.createRequest("ReferenceDataRequest")
        sec_el = req.getElement("securities")
        for s in tickers:
            sec_el.appendValue(s)
        fld_el = req.getElement("fields")
        fld_el.appendValue(field)

        msgs = self._send_request(req)
        out: Dict[str, List[Dict[str, str]]] = {}

        for msg in msgs:
            if msg.hasElement("responseError"):
                raise RuntimeError(f"ReferenceDataRequest error: {msg.getElement('responseError')}")
            if not msg.hasElement("securityData"):
                continue
            sdata_arr = msg.getElement("securityData")
            for i in range(sdata_arr.numValues()):
                sdata = sdata_arr.getValueAsElement(i)
                sec = sdata.getElementAsString("security")
                rows: List[Dict[str, str]] = []
                if sdata.hasElement("securityError"):
                    out[sec] = rows
                    continue
                fdata = sdata.getElement("fieldData")
                if fdata.hasElement(field):
                    bulk = fdata.getElement(field)
                    for j in range(bulk.numValues()):
                        row_el = bulk.getValueAsElement(j)
                        # Convert row to dict of {col_name: value_as_string}
                        row_dict = {}
                        for k in range(row_el.numElements()):
                            col_el = row_el.getElement(k)
                            name = col_el.name()
                            try:
                                val = col_el.getValueAsString()
                            except Exception:
                                try:
                                    val = str(col_el.getValue())
                                except Exception:
                                    val = ""
                            key = str(col_el.name())             # convert Name -> 'Ex-Date', etc.
                            row_dict[key] = val
                        rows.append(row_dict)
                out[sec] = rows
        return out


# =========================
# Event fetchers (Dividends)
# =========================
def fetch_dividend_events(bbg: BBG, tickers: List[str], start: str, end: str, min_amt: float = 0.0) -> Dict[str, List[pd.Timestamp]]:
    """
    Use bulk field 'DVD_HIST_ALL' to get dividend history and extract Ex-Dates.
    Returns {ticker: [ex_date_1, ex_date_2, ...]} filtered to [start, end].
    """
    raw = bbg.ref_bulk(tickers, "DVD_HIST_ALL")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    events: Dict[str, List[pd.Timestamp]] = {}
    for sec, rows in raw.items():
        ex_dates: List[pd.Timestamp] = []
        for row in rows:
            # The bulk columns vary slightly by instrument; be defensive:
            # Common keys: 'Ex-Date', 'Amount', 'Dividend Type', 'Pay Date', 'Record Date', 'Currency'
            keys = {k.lower(): k for k in row.keys()}
            ex_key = keys.get("ex-date") or keys.get("ex_date") or keys.get("ex date")
            amt_key = keys.get("amount") or keys.get("dividend amount")
            try:
                ex_dt = pd.to_datetime(row.get(ex_key)) if ex_key else None
            except Exception:
                ex_dt = None
            try:
                amt = float(row.get(amt_key)) if amt_key and row.get(amt_key) not in (None, "", "None") else np.nan
            except Exception:
                amt = np.nan
            if ex_dt is not None and start_dt <= ex_dt <= end_dt:
                if np.isnan(min_amt) or (not np.isnan(amt) and amt >= min_amt):
                    ex_dates.append(pd.Timestamp(ex_dt.normalize()))
        ex_dates = sorted(list(set(ex_dates)))
        events[sec] = ex_dates
    return events


# =========================
# Event study core
# =========================
def daily_returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.sort_index().pct_change()

def nearest_trading_index(dates: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    """
    Return the index location of the most recent trading date <= target.
    If all dates are > target, return -1 (invalid).
    """
    pos = dates.get_indexer([target], method="ffill")[0]
    return int(pos)

def estimate_capm(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    OLS: y = alpha + beta * x. Returns (alpha, beta).
    """
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    return alpha, beta

def event_AR_CAR_for_one(
    ticker: str,
    ex_dates: List[pd.Timestamp],
    r_asset: pd.DataFrame,
    r_mkt: pd.Series,
    L: int,
    R: int,
    est_pre: int,
    est_gap: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute AR and CAR for a single ticker across all events.
    Returns (AR_df, CAR_df) with rows=event_id, cols=tau in [-L..R].
    """
    dates = r_asset.index
    ar_rows = []
    car_rows = []
    for ev_idx, ev_dt in enumerate(ex_dates):
        ix = nearest_trading_index(dates, ev_dt)
        if ix <= 0:
            continue
        # Build estimation window [ix - est_gap - est_pre, ix - est_gap)
        est_end = ix - est_gap
        est_beg = est_end - est_pre
        if est_beg < 0 or est_end <= est_beg:
            continue
        y = r_asset.iloc[est_beg:est_end][ticker].values
        x = r_mkt.iloc[est_beg:est_end].values
        if np.isnan(y).any() or np.isnan(x).any() or len(y) < max(40, int(0.8*est_pre)):
            continue

        alpha, beta = estimate_capm(y, x)

        # Event window [-L, +R]
        w_beg = ix - L
        w_end = ix + R
        if w_beg < 0 or w_end >= len(dates):
            continue

        r_i = r_asset.iloc[w_beg:w_end+1][ticker].values
        r_m = r_mkt.iloc[w_beg:w_end+1].values
        exp = alpha + beta * r_m
        ar = r_i - exp
        # Index by tau
        tau = np.arange(-L, R+1)
        ar_s = pd.Series(ar, index=tau, name=f"{ticker}|{ev_dt.date()}")
        car_s = ar_s.cumsum()
        ar_rows.append(ar_s)
        car_rows.append(car_s)

    AR = pd.DataFrame(ar_rows) if ar_rows else pd.DataFrame(columns=np.arange(-L, R+1))
    CAR = pd.DataFrame(car_rows) if car_rows else pd.DataFrame(columns=np.arange(-L, R+1))
    return AR, CAR

def aggregate_AAR_ACAR(AR: pd.DataFrame, CAR: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute AAR(t), t-AAR(t), and ACAR(t), t-ACAR(t) across events (simple cross-sectional mean and t-stats).
    """
    aar = AR.mean(axis=0)
    acar = CAR.mean(axis=0)
    # t-stats (cross-sectional): mean / (std / sqrt(N))
    n_ar = AR.count(axis=0)
    n_car = CAR.count(axis=0)
    aar_t = aar / (AR.std(axis=0, ddof=1) / np.sqrt(n_ar))
    acar_t = acar / (CAR.std(axis=0, ddof=1) / np.sqrt(n_car))
    return aar, aar_t, acar, acar_t


# =========================
# Main
# =========================
def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Connect to Bloomberg
    bbg = BBG(cfg.host, cfg.port, cfg.timeout_ms)
    bbg.start()
    print("Connected to Bloomberg.")

    # 1) Prices
    print("Fetching prices for universe and market proxy ...")
    px_univ = bbg.bdh_single_field(
        cfg.universe, cfg.price_field, cfg.start_date, cfg.end_date,
        periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method
    )
    px_mkt = bbg.bdh_single_field(
        [cfg.market_proxy], cfg.price_field, cfg.start_date, cfg.end_date,
        periodicity=cfg.periodicity, fill_option=cfg.fill_option, fill_method=cfg.fill_method
    )
    px_univ = px_univ.ffill().dropna(how="all")
    px_mkt = px_mkt.ffill().dropna(how="all")

    # Align on common trading days
    common_dates = px_univ.index.intersection(px_mkt.index)
    px_univ = px_univ.reindex(common_dates)
    px_mkt = px_mkt.reindex(common_dates)

    # Daily returns
    r_univ = daily_returns_from_prices(px_univ).iloc[1:]
    r_mkt = daily_returns_from_prices(px_mkt).iloc[1:, 0]

    # 2) Events (dividend ex-dates)
    print("Fetching dividend histories (DVD_HIST_ALL) ...")
    div_events = fetch_dividend_events(bbg, cfg.universe, cfg.start_date, cfg.end_date, min_amt=cfg.min_div_amt)

    # 3) Event study: compute AR and CAR per ticker/event
    print("Computing AR and CAR for event windows ...")
    all_AR = []
    all_CAR = []
    per_ticker_counts = {}
    for tkr, ev_list in div_events.items():
        if not ev_list:
            continue
        AR, CAR = event_AR_CAR_for_one(
            ticker=tkr,
            ex_dates=ev_list,
            r_asset=r_univ,
            r_mkt=r_mkt,
            L=cfg.event_L,
            R=cfg.event_R,
            est_pre=cfg.est_pre,
            est_gap=cfg.est_gap
        )
        if not AR.empty:
            all_AR.append(AR)
            all_CAR.append(CAR)
            per_ticker_counts[tkr] = AR.shape[0]

    if not all_AR:
        print("No eligible events after filtering and data checks. Try expanding dates/universe.")
        bbg.stop()
        return

    AR_all = pd.concat(all_AR, axis=0)
    CAR_all = pd.concat(all_CAR, axis=0)

    # 4) Aggregate across events
    aar, aar_t, acar, acar_t = aggregate_AAR_ACAR(AR_all, CAR_all)

    # 5) Print quick stats
    total_events = AR_all.shape[0]
    print(f"\n=== Dividend Event Study: {total_events} events across {len(per_ticker_counts)} tickers ===")
    print(f"Window τ ∈ [{-cfg.event_L}, +{cfg.event_R}], estimation [{cfg.est_pre} pre, gap {cfg.est_gap}]")
    for w in [(1,1), (5,5), (10,10)]:
        left, right = w
        car_w = CAR_all[[c for c in CAR_all.columns if -left <= c <= right]].iloc[:, -1]  # CAR at +right if window starts at -left
        print(f"CAR[-{left}, +{right}] mean = {car_w.mean():+.3%} (median {car_w.median():+.3%})")

    print("\nSelected AAR (mean %) around event:")
    for tau in [-10, -5, -1, 0, +1, +5, +10]:
        if tau in aar.index:
            print(f"  τ={tau:>3}: AAR={aar[tau]*100:+.2f}% (t={aar_t[tau]:+.2f})")

    # 6) Save outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    AR_all.to_csv(os.path.join(cfg.out_dir, "AR_all_events.csv"))
    CAR_all.to_csv(os.path.join(cfg.out_dir, "CAR_all_events.csv"))
    pd.Series(per_ticker_counts).to_csv(os.path.join(cfg.out_dir, "events_per_ticker.csv"))
    pd.DataFrame({"AAR": aar, "AAR_t": aar_t}).to_csv(os.path.join(cfg.out_dir, "AAR.csv"))
    pd.DataFrame({"ACAR": acar, "ACAR_t": acar_t}).to_csv(os.path.join(cfg.out_dir, "ACAR.csv"))

    # 7) Optional plots
    if cfg.save_plots and HAVE_PLT:
        try:
            plt.figure(figsize=(9,5))
            aar.plot(label="AAR")
            (acar).plot(label="ACAR", linestyle="--")
            plt.axvline(0, color="k", linewidth=1)
            plt.title("Average Abnormal Return (AAR) and Average CAR around Dividend Ex-Date")
            plt.xlabel("Event time τ (trading days)")
            plt.ylabel("Return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "AAR_ACAR.png"), dpi=140)
            plt.close()
            print(f"Saved plots to: {os.path.abspath(cfg.out_dir)}")
        except Exception as e:
            print("Plotting skipped:", repr(e))

    # Done
    bbg.stop()
    print(f"Files saved to: {os.path.abspath(cfg.out_dir)}")
    print("Event study complete.")


# =========================
# Notes for other events
# =========================
"""
EARNINGS & M&A (how to adapt)
-----------------------------
- Earnings:
  If entitled, request the appropriate earnings announcement bulk/history field via
  ReferenceDataRequest (e.g., fields that contain per-date announcement rows).
  Replace `fetch_dividend_events()` with an analogous `fetch_earnings_events()`
  that returns a list of announcement timestamps per ticker, then run the same
  AR/CAR pipeline.

- M&A:
  If you have access to Bloomberg M&A event datasets, parse the bulk field that includes
  announcement dates (and optionally completion dates). As above, return event dates
  and reuse the AR/CAR computation.

- Fama-French model:
  This demo uses CAPM (alpha/beta vs market). If you wish to use Fama–French or Carhart,
  fetch factor-mimicking returns (e.g., via your own construction or proxies) and replace
  `estimate_capm()` with a multivariate OLS that predicts expected returns with those factors.
"""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", repr(e))
        sys.exit(1)
