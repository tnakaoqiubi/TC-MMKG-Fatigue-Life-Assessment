# -*- coding: utf-8 -*-
"""Rainflow counting used by the original TC-MMKG prototype.

The implementation intentionally keeps the original processing path that was used
for the system figures and damage results:
1) build a load-percentage history from moment percentage (falling back to
   actual/rated load),
2) extract peak/valley turning points with the original 2% minimum change,
3) apply the original three-point rainflow closure rule,
4) write cycle IDs back to the row-level history and save a separate cycle-detail
   workbook.

Stress values used for fatigue damage are retrieved later by ``try6_use.py`` for
each detected cycle. Keeping cycle extraction and stress lookup separate avoids a
vision-extraction outlier changing the rainflow cycle count itself.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

# 0-based positions matching the 23-field dataset described in the manuscript.
COL_VEHICLE_CODE = 0
COL_MODEL = 1
COL_TIME = 2
COL_ACTUAL_LOAD = 10
COL_RATED_LOAD = 11
COL_LENGTH = 12
COL_ANGLE = 13
COL_RADIUS = 14
COL_TORQUE_PERCENT = 17

MIN_PEAK_VALLEY_DIFF = 2.0
USE_ACTUAL_RATED = True

ALIASES = {
    "vehicle": ["车辆代码", "车辆型号", "crane code", "vehicle code", "model"],
    "time": ["时间", "时间点", "timestamp", "time"],
    "actual": ["实际吊重量", "actual load"],
    "rated": ["额定吊重量", "rated load"],
    "torque": ["力矩百分比", "moment percentage", "torque percentage", "载荷百分比"],
    "length": ["主臂长度", "长度", "boom length"],
    "angle": ["角度", "boom angle"],
    "radius": ["工作幅度", "working radius"],
}


def _find_col(df: pd.DataFrame, aliases: List[str], fallback: int):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return df.columns[fallback] if fallback < len(df.columns) else None


def extract_turning_points(time_series, value_series, min_diff=MIN_PEAK_VALLEY_DIFF):
    """Original peak/valley extraction logic."""
    turning_points = []
    value_series = pd.to_numeric(value_series, errors="coerce").reset_index(drop=True)
    time_series = pd.Series(time_series).reset_index(drop=True)
    for i in range(1, len(value_series) - 1):
        prev_val = value_series.iloc[i - 1]
        curr_val = value_series.iloc[i]
        next_val = value_series.iloc[i + 1]
        if pd.isna(prev_val) or pd.isna(curr_val) or pd.isna(next_val):
            continue
        is_peak = (curr_val > prev_val) and (curr_val > next_val)
        is_valley = (curr_val < prev_val) and (curr_val < next_val)
        if is_peak or is_valley:
            if abs(curr_val - prev_val) >= min_diff or abs(curr_val - next_val) >= min_diff:
                turning_points.append({
                    "index": i,
                    "time": time_series.iloc[i],
                    "value": float(curr_val),
                    "type": "peak" if is_peak else "valley",
                })
    return turning_points


def rainflow_counting_strict(turning_points):
    """Original three-point rainflow closure rule used by the prototype."""
    if len(turning_points) < 3:
        return []
    points = [(tp["index"], tp["time"], tp["value"], tp["index"]) for tp in turning_points]
    cycles = []
    remaining = deque(range(len(points)))
    while len(remaining) >= 3:
        idx1, idx2, idx3 = remaining[0], remaining[1], remaining[2]
        range_1_2 = abs(points[idx2][2] - points[idx1][2])
        range_2_3 = abs(points[idx3][2] - points[idx2][2])
        if range_1_2 <= range_2_3:
            cycles.append({
                "range": range_1_2,
                "amplitude": range_1_2 / 2.0,
                "mean": (points[idx1][2] + points[idx2][2]) / 2.0,
                "start_idx_in_tp": points[idx1][0],
                "end_idx_in_tp": points[idx2][0],
                "start_time": points[idx1][1],
                "end_time": points[idx2][1],
                "start_val": points[idx1][2],
                "end_val": points[idx2][2],
                "start_global_idx": points[idx1][3],
                "end_global_idx": points[idx2][3],
                "count": 1.0,
            })
            remaining.popleft()
            remaining.popleft()
        else:
            remaining.popleft()
    return cycles


def run_rainflow(input_file: str, output_file: str | None = None) -> str:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(input_file)
    df_raw = pd.read_excel(path, header=0)
    if df_raw.empty:
        raise ValueError("Input Excel contains no operating records.")

    c_vehicle = _find_col(df_raw, ALIASES["vehicle"], COL_VEHICLE_CODE)
    c_time = _find_col(df_raw, ALIASES["time"], COL_TIME)
    c_actual = _find_col(df_raw, ALIASES["actual"], COL_ACTUAL_LOAD)
    c_rated = _find_col(df_raw, ALIASES["rated"], COL_RATED_LOAD)
    c_torque = _find_col(df_raw, ALIASES["torque"], COL_TORQUE_PERCENT)
    c_length = _find_col(df_raw, ALIASES["length"], COL_LENGTH)
    c_angle = _find_col(df_raw, ALIASES["angle"], COL_ANGLE)
    c_radius = _find_col(df_raw, ALIASES["radius"], COL_RADIUS)
    if c_time is None:
        raise ValueError("Time column not found.")

    df = pd.DataFrame(index=df_raw.index)
    df["原始索引"] = df_raw.index
    df["车辆型号"] = df_raw[c_vehicle].astype(str) if c_vehicle is not None else "UNKNOWN"
    df["时间"] = pd.to_datetime(df_raw[c_time], errors="coerce")
    for col, src in [
        ("实际吊重量", c_actual), ("额定吊重量", c_rated), ("力矩百分比", c_torque),
        ("主臂长度", c_length), ("角度", c_angle), ("工作幅度", c_radius),
    ]:
        df[col] = pd.to_numeric(df_raw[src], errors="coerce") if src is not None else np.nan

    load_pct = pd.to_numeric(df["力矩百分比"], errors="coerce")
    if USE_ACTUAL_RATED:
        ratio = pd.to_numeric(df["实际吊重量"], errors="coerce") / pd.to_numeric(
            df["额定吊重量"], errors="coerce"
        ).replace(0, np.nan) * 100.0
        load_pct = load_pct.where(load_pct.notna(), ratio)
    df["载荷百分比"] = load_pct

    all_cycles_temp: List[Dict] = []
    for vehicle, group in df.groupby("车辆型号", dropna=False):
        if group["时间"].isna().all():
            continue
        group = group.sort_values("时间").reset_index(drop=True)
        turning_points = extract_turning_points(group["时间"], group["载荷百分比"])
        if len(turning_points) < 3:
            continue
        cycles = rainflow_counting_strict(turning_points)
        for cycle in cycles:
            start_global = int(group.iloc[cycle["start_idx_in_tp"]]["原始索引"])
            end_global = int(group.iloc[cycle["end_idx_in_tp"]]["原始索引"])
            all_cycles_temp.append({
                "车辆": vehicle,
                "起始时间": cycle["start_time"],
                "结束时间": cycle["end_time"],
                "起始原始索引": start_global,
                "结束原始索引": end_global,
                "range": cycle["range"],
                "amplitude": cycle["amplitude"],
                "mean": cycle["mean"],
                "start_val": cycle["start_val"],
                "end_val": cycle["end_val"],
            })

    if output_file is None:
        output_file = str(Path(OUTPUT_DIR) / f"{path.stem}_雨流结果.xlsx")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df_out = df_raw.copy()
    df_out["载荷百分比"] = df["载荷百分比"]
    df_out["循环ID"] = np.nan
    df_out["时间"] = df["时间"]
    for col in ["实际吊重量", "额定吊重量", "主臂长度", "角度", "工作幅度"]:
        df_out[col] = df[col]

    if not all_cycles_temp:
        df_out.to_excel(output_file, index=False)
        # Keep a valid empty cycle-detail file so the damage stage can emit a clear error.
        pd.DataFrame(columns=["循环ID", "起始时间", "结束时间"]).to_excel(
            output_file.replace(".xlsx", "_循环明细.xlsx"), index=False
        )
        return output_file

    df_cycles = pd.DataFrame(all_cycles_temp).sort_values("起始时间").reset_index(drop=True)
    df_cycles["循环ID"] = range(1, len(df_cycles) + 1)

    # Write cycle IDs back to the original row positions. Use positional bounds from
    # the original dataset rather than the sorted group index to avoid misalignment.
    for _, cycle in df_cycles.iterrows():
        start_idx, end_idx = sorted((int(cycle["起始原始索引"]), int(cycle["结束原始索引"])))
        mask = (df_out.index >= start_idx) & (df_out.index <= end_idx)
        df_out.loc[mask, "循环ID"] = int(cycle["循环ID"])

    df_out.to_excel(output_file, index=False)
    cycle_detail_file = output_file.replace(".xlsx", "_循环明细.xlsx")
    df_cycles.to_excel(cycle_detail_file, index=False)
    return output_file


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python rainflow2_use.py <operational-data.xlsx>")
    print(run_rainflow(sys.argv[1]))
