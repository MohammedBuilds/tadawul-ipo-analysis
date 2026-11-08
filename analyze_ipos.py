"""
Tadawul IPO analysis
How did Saudi main market IPOs trade after listing, and does oversubscription tell you anything?

Usage (run from the folder that holds ipo_dataset.csv and tasi_manual.csv):
    pip install yfinance pandas scipy matplotlib
    python analyze_ipos.py

Outputs (written to ./output):
    prices/            daily closes per ticker, cached so reruns are fast
    ipo_returns.csv    returns at each checkpoint, TASI returns, excess returns
    data_checks.csv    every adjustment and data issue found
    group_summary.csv  low / medium / high oversubscription comparison
    summary.txt        headline results
    chart1_scatter.png, chart2_checkpoints.png, chart3_sectors.png, chart4_post_day1_buyers_1y.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATASET = "ipo_dataset.csv"
OUT = "output"
PRICE_DIR = os.path.join(OUT, "prices")
CHECKPOINTS = {"day1": 0, "3m": 63, "6m": 126, "1y": 252}   # trading-day index, 0 = listing day
TASI_TICKERS = ["^TASI.SR", "^TASI"]
MIN_HISTORY = 126 + 1

# chart style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})

flags = []
def flag(company, ticker, issue):
    flags.append({"company_name": company, "ticker": ticker, "issue": issue})


# Prices

def fetch(symbol, start):
    """Daily Close (split-adjusted by Yahoo) plus split events. Cached to disk."""
    os.makedirs(PRICE_DIR, exist_ok=True)
    cache = os.path.join(PRICE_DIR, symbol.replace("^", "IDX_") + ".csv")
    split_cache = cache.replace(".csv", "_splits.csv")
    if os.path.exists(cache):
        px = pd.read_csv(cache, index_col=0, parse_dates=True)["Close"]
        splits = (pd.read_csv(split_cache, index_col=0, parse_dates=True).iloc[:, 0]
                  if os.path.exists(split_cache) else pd.Series(dtype=float))
        return px, splits
    import yfinance as yf
    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)
            hist = t.history(start=start, auto_adjust=False, actions=True)
            break
        except Exception as e:
            print(f"  retry {symbol}: {e}")
            time.sleep(2)
    else:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    hist.index = hist.index.tz_localize(None)
    px = hist["Close"].dropna()
    splits = hist.get("Stock Splits", pd.Series(dtype=float))
    splits = splits[splits > 0]
    px.to_frame("Close").to_csv(cache)
    splits.to_frame("ratio").to_csv(split_cache)
    return px, splits


EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
MANUAL_TASI = "tasi_manual.csv"


def load_tasi():
    # 1) a TASI file you downloaded yourself (columns: Date, Close)
    if os.path.exists(MANUAL_TASI):
        m = pd.read_csv(MANUAL_TASI)
        cols = {c: str(c).strip().lower() for c in m.columns}
        date_col = next((c for c, l in cols.items() if "date" in l), m.columns[0])
        close_col = next((c for c, l in cols.items() if l in ("close", "price", "last", "adj close")),
                         [c for c in m.columns if c != date_col][0])
        dates = pd.to_datetime(m[date_col], errors="coerce")
        closes = pd.to_numeric(m[close_col].astype(str).str.replace(",", ""), errors="coerce")
        px = pd.Series(closes.values, index=dates).dropna()
        px = px[px.index.notna()].sort_index()
        print(f"TASI loaded from {MANUAL_TASI} ({len(px)} days)")
        return px
    # 2) Yahoo
    for sym in TASI_TICKERS:
        px, _ = fetch(sym, "2018-12-01")
        print(f"  {sym}: {len(px)} days returned")
        if len(px) > 500:
            print(f"TASI loaded from {sym}")
            return px
    print("WARNING: could not load TASI from Yahoo. Excess returns will be blank, everything else still runs.")
    print(f"  Fix: save TASI daily history as {MANUAL_TASI} (columns Date, Close) next to this script and rerun.")
    return EMPTY


def last_on_or_before(series, date, strict=False):
    if series.empty:
        return np.nan
    s = series[series.index < date] if strict else series[series.index <= date]
    return s.iloc[-1] if len(s) else np.nan


# Main

def main():
    if not os.path.exists(DATASET):
        sys.exit(f"Can't find {DATASET} in this folder.")
    os.makedirs(OUT, exist_ok=True)
    ipo = pd.read_csv(DATASET, dtype={"ticker": str})
    ipo["listing_date"] = pd.to_datetime(ipo["listing_date"], errors="coerce")
    tasi = load_tasi()
    today = pd.Timestamp.today().normalize()

    rows = []
    for _, r in ipo.iterrows():
        name, tk = r["company_name"], r["ticker"]
        sym = f"{tk}.SR"
        start = (r["listing_date"] - pd.Timedelta(days=10)).strftime("%Y-%m-%d") \
            if pd.notna(r["listing_date"]) else "2019-01-01"
        print(f"{sym:<9} {name}")
        px, splits = fetch(sym, start)
        if px.empty:
            flag(name, tk, "No price data from Yahoo. Check ticker.")
            continue

        # Listing date: sourced if we have it, otherwise Yahoo's first bar
        yahoo_first = px.index[0]
        if pd.notna(r["listing_date"]):
            listing = r["listing_date"]
            gap = (yahoo_first - listing).days
            if abs(gap) > 5:
                flag(name, tk, f"Yahoo history starts {yahoo_first.date()}, {gap} days after the real listing "
                               f"{listing.date()}. Checkpoints are counted from the real listing date.")
            px = px[px.index >= listing]
        else:
            listing = yahoo_first
            flag(name, tk, f"No sourced listing date. Using Yahoo's first trading day {listing.date()}, unverified.")

        # trading calendar from TASI (complete), so checkpoints don't drift when Yahoo is missing days
        if not tasi.empty:
            cal = tasi.index[tasi.index >= listing]
        else:
            cal = px.index
        if len(cal) < MIN_HISTORY or (today - listing).days < 183:
            flag(name, tk, f"Less than 6 months of trading history. Excluded from 6m analysis.")

        # corporate actions: Yahoo's Close is split-adjusted, the offer price is not
        post = splits[splits.index > listing] if len(splits) else splits
        factor = float(np.prod(post.values)) if len(post) else 1.0
        offer = float(r["offer_price_sar"])
        adj_offer = offer / factor
        if factor != 1.0:
            flag(name, tk, f"Split/bonus factor {factor:.4g} after listing "
                           f"({', '.join(f'{d.date()}: {v:g}' for d, v in post.items())}). "
                           f"Offer price adjusted {offer} -> {adj_offer:.4f}.")

        # Day-1 close: sourced value first, otherwise Yahoo's close on the listing day
        src_day1 = pd.to_numeric(r.get("first_day_close_sar"), errors="coerce")
        if pd.notna(src_day1):
            day1_adj = float(src_day1) / factor
            day1_from = "source"
        elif len(px) and px.index[0] == listing:
            day1_adj = float(px.iloc[0])
            day1_from = "yahoo"
        else:
            day1_adj = np.nan
            day1_from = "missing"
            flag(name, tk, "Yahoo has no bar on the listing day and no sourced day-1 close. Day-1 left blank.")

        if pd.notna(day1_adj):
            pop = day1_adj / adj_offer - 1
            if not -0.31 <= pop <= 0.31:
                flag(name, tk, f"Day-1 close implies a {pop:+.0%} move from offer price. Outside the 30% debut "
                               f"limit, so the ticker, offer price or split adjustment is probably wrong.")

        # unhandled corporate actions: big single-day drops
        daily = px.pct_change()
        for d, v in daily[daily < -0.20].items():
            flag(name, tk, f"{v:.0%} single-day move on {d.date()}. Check for an unadjusted bonus issue or split.")

        rec = {"company_name": name, "ticker": tk, "listing_date_used": listing.date(),
               "sector": r["sector"], "offer_price_sar": offer, "adj_factor": factor,
               "adj_offer_price": round(adj_offer, 4),
               "institutional_oversubscription": r["institutional_oversubscription"],
               "retail_oversubscription": r["retail_oversubscription"],
               "ipo_size_sar_mln": r["ipo_size_sar_mln"],
               "first_day_close_sar": round(day1_adj * factor, 4) if pd.notna(day1_adj) else np.nan,
               "day1_source": day1_from}

        # TASI base = last close before listing
        tasi_base = last_on_or_before(tasi, listing, strict=True)
        for label, k in CHECKPOINTS.items():
            for p in ("ret_", "post_", "tasi_ret_", "excess_", "date_"):
                rec[f"{p}{label}"] = np.nan
            if k == 0:
                d, price = listing, day1_adj
            else:
                if len(cal) <= k:
                    continue
                d = cal[k]
                if px.empty or px.index[0] > d:      # Yahoo has no data yet at this checkpoint
                    continue
                price = last_on_or_before(px, d)
            if pd.isna(price):
                continue
            ret = price / adj_offer - 1
            t_now = last_on_or_before(tasi, d)
            t_ret = t_now / tasi_base - 1 if pd.notna(tasi_base) else np.nan
            rec[f"ret_{label}"] = ret
            # return for someone who bought at the day-1 close instead of getting an allocation
            rec[f"post_{label}"] = price / day1_adj - 1 if (k and pd.notna(day1_adj)) else np.nan
            rec[f"tasi_ret_{label}"] = t_ret
            rec[f"excess_{label}"] = ret - t_ret if pd.notna(t_ret) else np.nan
            rec[f"date_{label}"] = pd.Timestamp(d).date()
        rec["ret_latest"] = px.iloc[-1] / adj_offer - 1
        rows.append(rec)

    res = pd.DataFrame(rows)
    # extra columns for Power BI
    res["listing_year"] = pd.to_datetime(res["listing_date_used"]).dt.year
    ok = res["ret_6m"].notna() & res["institutional_oversubscription"].notna()
    res.loc[ok, "oversub_group"] = pd.qcut(res.loc[ok, "institutional_oversubscription"], 3,
                                           labels=["Low", "Medium", "High"]).astype(str)
    res["above_offer_6m"] = np.where(res["ret_6m"].notna(), res["ret_6m"] > 0, np.nan)
    res = res.drop(columns=["post_day1"], errors="ignore")
    num = res.select_dtypes("number").columns.drop(["listing_year"], errors="ignore")
    res.round({c: 4 for c in num}).to_csv(os.path.join(OUT, "ipo_returns.csv"), index=False)
    pd.DataFrame(flags).to_csv(os.path.join(OUT, "data_checks.csv"), index=False)

    # fill first_day_close back into the dataset copy
    ds = ipo.copy()
    ds["first_day_close_sar"] = ds["ticker"].map(res.set_index("ticker")["first_day_close_sar"])
    ds["listing_date"] = ds["ticker"].map(res.set_index("ticker")["listing_date_used"]).fillna(ds["listing_date"])
    ds.to_csv(os.path.join(OUT, "ipo_dataset_filled.csv"), index=False)

    analyse(res)


# Analysis

def analyse(res):
    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    six = res.dropna(subset=["ret_6m"])
    test = six.dropna(subset=["institutional_oversubscription"]).copy()
    say(f"IPOs in dataset with prices: {len(res)}")
    say(f"IPOs with 6+ months of history: {len(six)}")
    say(f"Of those, with institutional oversubscription: {len(test)}")
    say("")

    say("== Whole sample, median return vs offer price ==")
    for c in CHECKPOINTS:
        s = res[f"ret_{c}"].dropna()
        e = res[f"excess_{c}"].dropna()
        say(f"  {c:>4}: median {s.median():+.1%} (n={len(s)}), share above offer {(s > 0).mean():.0%}, "
            f"median vs TASI {e.median():+.1%}" if len(e) else f"  {c:>4}: median {s.median():+.1%} (n={len(s)})")
    say("")

    # groups
    test["group"] = pd.qcut(test["institutional_oversubscription"], 3, labels=["Low", "Medium", "High"])
    agg = {}
    for c in CHECKPOINTS:
        agg[f"median_ret_{c}"] = (f"ret_{c}", "median")
        agg[f"mean_ret_{c}"] = (f"ret_{c}", "mean")
        agg[f"median_excess_{c}"] = (f"excess_{c}", "median")
        if c != "day1":
            agg[f"median_post_{c}"] = (f"post_{c}", "median")
    agg["count"] = ("ticker", "count")
    agg["oversub_min"] = ("institutional_oversubscription", "min")
    agg["oversub_max"] = ("institutional_oversubscription", "max")
    g = test.groupby("group", observed=True).agg(**agg)
    g.to_csv(os.path.join(OUT, "group_summary.csv"))
    say("== By institutional oversubscription (equal-size thirds) ==")
    for grp, row in g.iterrows():
        say(f"  {grp:<6} {row.oversub_min:.0f}x-{row.oversub_max:.0f}x (n={row['count']:.0f}): "
            f"day1 {row.median_ret_day1:+.1%} | 3m {row.median_ret_3m:+.1%} | 6m {row.median_ret_6m:+.1%} | "
            f"6m vs TASI {row.median_excess_6m:+.1%}")
    say("")
    say("== Bought at the day-1 close instead of the offer price (median) ==")
    for grp, row in g.iterrows():
        say(f"  {grp:<6} 3m {row.median_post_3m:+.1%} | 6m {row.median_post_6m:+.1%} | 1y {row.median_post_1y:+.1%}")
    s = res["post_6m"].dropna()
    say(f"  All IPOs: 6m {s.median():+.1%}, share above day-1 close after 6m {(s > 0).mean():.0%} (n={len(s)})")
    say("")

    rho, p = spearmanr(test["institutional_oversubscription"], test["ret_6m"])
    say(f"Spearman oversubscription vs 6m return: rho={rho:+.2f}, p={p:.3f}, n={len(test)}")
    ex = test.dropna(subset=["excess_6m"])
    if len(ex) > 5:
        rho2, p2 = spearmanr(ex["institutional_oversubscription"], ex["excess_6m"])
        say(f"Spearman oversubscription vs 6m excess return: rho={rho2:+.2f}, p={p2:.3f}, n={len(ex)}")
    pop = six.dropna(subset=["ret_day1"])
    rho3, p3 = spearmanr(pop["ret_day1"], pop["ret_6m"])
    say(f"Spearman day-1 pop vs 6m return from offer: rho={rho3:+.2f}, p={p3:.3f}, n={len(pop)} "
        "(partly mechanical, the 6m return includes the day-1 pop)")
    pp = six.dropna(subset=["ret_day1", "post_6m"])
    rho4, p4 = spearmanr(pp["ret_day1"], pp["post_6m"])
    say(f"Spearman day-1 pop vs return from day-1 close to 6m: rho={rho4:+.2f}, p={p4:.3f}, n={len(pp)}")
    pi = test.dropna(subset=["post_6m"])
    rho5, p5 = spearmanr(pi["institutional_oversubscription"], pi["post_6m"])
    say(f"Spearman oversubscription vs return from day-1 close to 6m: rho={rho5:+.2f}, p={p5:.3f}, n={len(pi)}")
    say("Read these as patterns, not proof. With ~50-60 IPOs, p-values above 0.05 mean the "
        "relationship could easily be noise.")
    say("")

    # survivors
    six = six.copy()
    six["survivor"] = six["ret_6m"] > 0
    say(f"== Above offer price at 6 months: {six.survivor.sum()} of {len(six)} ==")
    comp = six.groupby("survivor").agg(
        n=("ticker", "count"),
        median_oversub=("institutional_oversubscription", "median"),
        median_size_sar_mln=("ipo_size_sar_mln", "median"),
        median_day1_pop=("ret_day1", "median"))
    say(comp.rename(index={True: "Above offer", False: "Below offer"}).to_string())
    say("")
    say("Survivors: " + ", ".join(six[six.survivor].sort_values("ret_6m", ascending=False)
                                  .apply(lambda r: f"{r.company_name} ({r.ret_6m:+.0%})", axis=1)))
    say("")
    sec = six.groupby("sector").agg(n=("ticker", "count"), above=("survivor", "mean"),
                                    median_6m=("ret_6m", "median")).sort_values("above", ascending=False)
    say("By sector (all sizes):")
    say(sec.to_string(formatters={"above": "{:.0%}".format, "median_6m": "{:+.1%}".format}))
    say("")
    say(f"Flags raised: {len(flags)}. Review output/data_checks.csv before trusting any number above.")

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    charts(test, g, six, sec)


# Charts

def pct(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:+.0%}" if v else "0%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def charts(test, g, six, sec):
    # 1. scatter
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.scatter(test["institutional_oversubscription"], test["ret_6m"], s=60, color=BLUE,
               edgecolor=BG, linewidth=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Institutional book oversubscribed (times, log scale)")
    ax.set_ylabel("Return vs offer price after 6 months")
    pct(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Saudi main market IPOs: demand at listing vs six months later",
                 loc="left", color=INK, fontsize=13, pad=14)
    # label the best and worst only
    for _, r in pd.concat([test.nlargest(1, "ret_6m"), test.nsmallest(1, "ret_6m")]).iterrows():
        ax.annotate(r.company_name.split(" (")[0], (r.institutional_oversubscription, r.ret_6m),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.01, "Source: offer data from Argaam, Saudi Exchange and company announcements; prices from Yahoo Finance",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "chart1_scatter.png"))
    plt.close(fig)

    # 2. median return by checkpoint per group
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    x = list(CHECKPOINTS)
    labels = ["Day 1", "3 months", "6 months", "1 year"]
    for grp, col in zip(["Low", "Medium", "High"], [BLUE, ORANGE, AQUA]):
        if grp not in g.index:
            continue
        y = [g.loc[grp, f"median_ret_{c}"] for c in x]
        ax.plot(labels, y, color=col, lw=2, marker="o", markersize=8,
                markeredgecolor=BG, markeredgewidth=2, label=f"{grp} demand")
        ax.annotate(f"{grp} demand", (labels[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    pct(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Median return vs offer price")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("Median return after listing, by oversubscription group", loc="left",
                 color=INK, fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart2_checkpoints.png"))
    plt.close(fig)

    # 4. Median 1-year return after buying at the day-1 close, by demand group
    if "median_post_1y" in g.columns:
        order = [x for x in ["Low", "Medium", "High"] if x in g.index]
        vals = [g.loc[x, "median_post_1y"] for x in order]
        ns = [int(test[(test["group"] == x) & test["post_1y"].notna()].shape[0]) for x in order]
        ranges = [f"{g.loc[x, 'oversub_min']:.0f}x to {g.loc[x, 'oversub_max']:.0f}x" for x in order]
        fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
        cols = [ORANGE if x == "High" else "#b9b8b2" for x in order]
        bars = ax.bar(range(len(order)), vals, color=cols, width=0.55, zorder=3)
        ax.axhline(0, color=INK2, lw=1, zorder=4)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + (0.012 if v >= 0 else -0.012), f"{v:+.1%}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=13, color=INK, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{x} demand\n{r} oversubscribed\n(n={n})" for x, r, n in zip(order, ranges, ns)],
                           fontsize=9.5)
        lo = min(min(vals), 0)
        hi = max(max(vals), 0)
        ax.set_ylim(lo * 1.35 - 0.02, hi * 1.35 + 0.04)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_title("Bought at the day-one close, one year later\nMedian return, Tadawul main market IPOs 2019 to 2026",
                     loc="left", color=INK, fontsize=13, pad=14)
        fig.text(0.01, 0.01, "Data: Argaam, Saudi Exchange, company announcements, Yahoo Finance. Price returns, "
                 "excludes dividends.", fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(OUT, "chart4_post_day1_buyers_1y.png"))
        plt.close(fig)

    # 3. Share above offer price at 6m by sector, sectors with 3+ IPOs
    s3 = sec[sec["n"] >= 3].sort_values("above")
    if len(s3):
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(s3) + 1.8), dpi=200)
        ax.barh(s3.index, s3["above"], color=BLUE, height=0.55)
        for i, (sct, r) in enumerate(s3.iterrows()):
            ax.text(r.above + 0.01, i, f"{r.above:.0%} of {r.n:.0f}", va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, 1.25)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle("Share of IPOs above offer price after 6 months, by sector (sectors with 3+ IPOs)",
                     x=0.02, ha="left", color=INK, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(OUT, "chart3_sectors.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
