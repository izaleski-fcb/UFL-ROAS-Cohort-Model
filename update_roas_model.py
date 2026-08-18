"""
Пересчёт UFL ROAS Cohort Model.

Тянет недельные когорты (D0-D90) по iOS и Android с 15 апреля 2026 по сегодня,
группировка по media source (сетка), пересчитывает мультипликаторы D0-D90
взвешенно по revenue и степенной коэффициент b по регрессии D30/D60/D90 (по
всем неделям, где D90 уже созрел), считает прогноз D180/D360 для агрегата
(все сетки) и для каждой сетки отдельно.

Пишет результат в roas_data_output.json (DATA, NETWORK_DATA, MULT, meta) —
дальше отдельным шагом эти блоки нужно вставить в roas_weekly.html вместо
текущих `const DATA = {...};` и `const NETWORK_DATA = {...};`.

Требует переменные окружения: AF_API_TOKEN, AF_APP_ID_IOS, AF_APP_ID_ANDROID.
"""
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from appsflyer.client import get_ios_client, get_android_client

NETWORKS_FOR_BREAKDOWN = [
    "googleadwords_int", "Facebook Ads", "applovin_int",
    "tiktokglobal_int", "unityads_int", "moloco_int", "Apple Search Ads",
]

START_DATE = date(2026, 4, 15)


def week_windows(start: date, end: date):
    windows = []
    cur = start
    while cur <= end:
        w_end = min(cur + timedelta(days=6), end)
        windows.append((cur.isoformat(), w_end.isoformat()))
        cur = w_end + timedelta(days=1)
    return windows


def week_label(week_from: str) -> str:
    d = pd.to_datetime(week_from)
    d2 = d + pd.Timedelta(days=6)
    return f"{d.strftime('%d.%m')}–{d2.strftime('%d.%m')}"


def pull_all(today: date):
    windows = week_windows(START_DATE, today)
    frames = []
    for platform, client in [("iOS", get_ios_client()), ("Android", get_android_client())]:
        for wf, wt in windows:
            try:
                df = client.get_cohort_revenue(wf, wt, groupings=["pid"])
            except Exception as e:
                print(f"{platform} {wf}-{wt}: ERROR {e}")
                continue
            if df.empty:
                print(f"{platform} {wf}-{wt}: EMPTY")
                continue
            df["platform"] = platform
            df["week_from"] = wf
            df["week_to"] = wt
            frames.append(df)
            time.sleep(0.15)
    return pd.concat(frames, ignore_index=True)


def weighted_avg_mult(d, num_col, den_col):
    sub = d[(d[num_col] > 0) & (d[den_col] > 0)]
    if len(sub) == 0:
        return None, 0
    ratios = sub[num_col] / sub[den_col]
    w = sub[den_col]
    return (ratios * w).sum() / w.sum(), len(sub)


def compute_multipliers(agg: pd.DataFrame) -> dict:
    mult = {}
    for platform in ["iOS", "Android"]:
        d = agg[agg["platform"] == platform]
        m = {}
        for num, den, label in [
            ("rev_d1", "rev_d0", "D0_D1"),
            ("rev_d7", "rev_d1", "D1_D7"),
            ("rev_d30", "rev_d7", "D7_D30"),
            ("rev_d60", "rev_d30", "D30_D60"),
            ("rev_d90", "rev_d60", "D60_D90"),
        ]:
            val, n = weighted_avg_mult(d, num, den)
            m[label] = round(val, 3) if val is not None else None
            print(f"{platform} {label} = {m[label]} (n={n})")

        d90 = d[d["rev_d90"] > 0]
        bs = []
        for _, r in d90.iterrows():
            ts = np.array([30, 60, 90])
            rs = np.array([r["rev_d30"], r["rev_d60"], r["rev_d90"]])
            if (rs <= 0).any():
                continue
            b, _ = np.polyfit(np.log(ts), np.log(rs), 1)
            bs.append(b)
        b = float(np.mean(bs)) if bs else 0.25
        m["b"] = round(b, 3)
        m["D60_D180"] = round((180 / 60) ** b, 3)
        m["D180_D360"] = round((360 / 180) ** b, 3)
        print(f"{platform} b={m['b']} n_weeks_d90={len(bs)} D60_D180={m['D60_D180']} D180_D360={m['D180_D360']}")
        mult[platform] = m
    return mult


def forecast_row(m: dict, d0, d1, d7, d30, d60, d90):
    def has(x):
        return x is not None and not pd.isna(x) and x > 0

    if has(d90):
        last = "D90"
    elif has(d60):
        last = "D60"
    elif has(d30):
        last = "D30"
    elif has(d7):
        last = "D7"
    elif has(d1):
        last = "D1"
    else:
        last = "D0"

    if has(d60):
        f180 = d60 * m["D60_D180"]
        f360 = f180 * m["D180_D360"]
    elif has(d30):
        f60 = d30 * m["D30_D60"]
        f180 = f60 * m["D60_D180"]
        f360 = f180 * m["D180_D360"]
    elif has(d7):
        f30 = d7 * m["D7_D30"]
        f60 = f30 * m["D30_D60"]
        f180 = f60 * m["D60_D180"]
        f360 = f180 * m["D180_D360"]
    elif has(d1):
        f7 = d1 * m["D1_D7"]
        f30 = f7 * m["D7_D30"]
        f60 = f30 * m["D30_D60"]
        f180 = f60 * m["D60_D180"]
        f360 = f180 * m["D180_D360"]
    elif has(d0):
        f1 = d0 * m["D0_D1"]
        f7 = f1 * m["D1_D7"]
        f30 = f7 * m["D7_D30"]
        f60 = f30 * m["D30_D60"]
        f180 = f60 * m["D60_D180"]
        f360 = f180 * m["D180_D360"]
    else:
        f180 = f360 = None
    return last, f180, f360


def build_aggregate(full: pd.DataFrame, mult: dict) -> dict:
    day_cols = {d: f"revenue_sum_day_{d}" for d in [0, 1, 3, 7, 14, 30, 60, 90]}
    agg = full.groupby(["platform", "week_from", "week_to"]).agg(
        cost=("cost", "sum"),
        **{f"rev_d{d}": (col, "sum") for d, col in day_cols.items()},
    ).reset_index()

    out = {"Android": [], "iOS": []}
    for _, r in agg.iterrows():
        m = mult[r["platform"]]
        cost = r["cost"]
        if not cost or cost <= 0:
            continue

        def pct(x):
            return round(x / cost * 100, 1) if x and x > 0 else (0.0 if x == 0 else None)

        last, f180, f360 = forecast_row(
            m, r["rev_d0"], r["rev_d1"], r["rev_d7"], r["rev_d30"], r["rev_d60"], r["rev_d90"]
        )
        row = {
            "w": week_label(r["week_from"]),
            "spend": round(cost),
            "last": last,
            "D0": pct(r["rev_d0"]), "D1": pct(r["rev_d1"]), "D7": pct(r["rev_d7"]),
            "D30": pct(r["rev_d30"]), "D60": pct(r["rev_d60"]), "D90": pct(r["rev_d90"]),
            "D180": round(f180 / cost * 100, 1) if f180 else None,
            "D360": round(f360 / cost * 100, 1) if f360 else None,
        }
        out[r["platform"]].append(row)
    return out, agg


def build_network_breakdown(full: pd.DataFrame, mult: dict) -> dict:
    result = {"iOS": {}, "Android": {}}
    for platform in ["iOS", "Android"]:
        m = mult[platform]
        for net in NETWORKS_FOR_BREAKDOWN:
            sub = full[(full["platform"] == platform) & (full["pid"] == net)]
            if sub.empty:
                continue
            rows = []
            for _, r in sub.iterrows():
                cost = r["cost"]
                if pd.isna(cost) or cost <= 0:
                    continue

                def pct(x):
                    return round(x / cost * 100, 1) if x and not pd.isna(x) and x > 0 else None

                last, f180, f360 = forecast_row(
                    m,
                    r["revenue_sum_day_0"], r["revenue_sum_day_1"], r["revenue_sum_day_7"],
                    r["revenue_sum_day_30"], r["revenue_sum_day_60"], r["revenue_sum_day_90"],
                )
                rows.append({
                    "w": week_label(r["week_from"]),
                    "spend": round(cost),
                    "users": int(r["users"]) if not pd.isna(r["users"]) else None,
                    "last": last,
                    "D0": pct(r["revenue_sum_day_0"]), "D1": pct(r["revenue_sum_day_1"]),
                    "D7": pct(r["revenue_sum_day_7"]), "D30": pct(r["revenue_sum_day_30"]),
                    "D60": pct(r["revenue_sum_day_60"]), "D90": pct(r["revenue_sum_day_90"]),
                    "D180": round(f180 / cost * 100, 1) if f180 else None,
                    "D360": round(f360 / cost * 100, 1) if f360 else None,
                })
            result[platform][net] = rows
    return result


def main():
    today = date.today()
    print(f"Pulling cohorts {START_DATE.isoformat()} .. {today.isoformat()}")
    full = pull_all(today)

    day_cols = {d: f"revenue_sum_day_{d}" for d in [0, 1, 3, 7, 14, 30, 60, 90]}
    agg = full.groupby(["platform", "week_from", "week_to"]).agg(
        cost=("cost", "sum"), **{f"rev_d{d}": (col, "sum") for d, col in day_cols.items()}
    ).reset_index()

    mult = compute_multipliers(agg)
    data, _ = build_aggregate(full, mult)
    network_data = build_network_breakdown(full, mult)

    out = {
        "meta": {"updated": today.isoformat(), "n_weeks": len(data["Android"])},
        "MULT": mult,
        "DATA": data,
        "NETWORK_DATA": network_data,
    }
    with open("roas_data_output.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("Wrote roas_data_output.json")
    print(json.dumps(mult, indent=2))


if __name__ == "__main__":
    main()
