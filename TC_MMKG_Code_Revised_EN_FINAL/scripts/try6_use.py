# -*- coding: utf-8 -*-
"""汽车起重机伸缩臂疲劳损伤与剩余寿命计算。

本文件只负责损伤与寿命计算，不改变雨流计数、三元组提取、跨模态融合、
Neo4j 图谱结构或前端流程。

本版针对“实测记录时间不连续、并非逐日连续工况记录”的实际数据特点，
修正了原来的年化方法：

原方法：
    D_annual = D_total * (365 / 有效工作日期数)

该方法会把零散抽样日期误当成连续观测天数，从而对年损伤进行不合理放大。
例如只有 73 个出现有效载荷的日期时，会把样本损伤约放大 5 倍。

修正方法：
1. 当前 257 条离散记录及其雨流循环被视为一个“代表性年度载荷谱样本块”；
2. D_total 是该代表性样本块对应的 Miner 累积损伤；
3. 默认代表周期为 1 年，因此 D_annual = D_total；
4. 剩余寿命按 Miner 剩余损伤容量计算：
       N_remaining_eq = N_sample * (1 - D_current) / D_total
       Life_remaining  = N_remaining_eq / (N_sample / T_rep)
                       = T_rep * (1 - D_current) / D_total
   其中：
       N_sample  为当前样本雨流等效循环数；
       T_rep     为代表性样本对应的时间周期，默认 1 年；
       D_current 为设备在本次评估前已经真实累计的历史损伤，默认 0。

重要说明：
- 当前样本的 D_total 只用于描述代表性载荷谱的损伤强度，不再同时被当作
  “已经消耗掉的历史损伤”，避免重复扣减寿命。
- 如果今后掌握了设备真实历史累计损伤，可修改 CURRENT_ACCUMULATED_DAMAGE。
- 如果当前样本代表的不是 1 年，可修改 REPRESENTATIVE_PERIOD_YEARS。
- 其余应力查询、Haigh/Goodman 转换、S-N 曲线、Miner 损伤、图表方法均保持不变。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
try:
    from neo4j import GraphDatabase
except ImportError:  # core numeric functions remain testable without Neo4j installed
    GraphDatabase = None

from config import (
    FALLBACK_STRESS_AT_100_PERCENT_MPA,
    FALLBACK_STRESS_RATIO,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OUTPUT_DIR,
    Q690D_FAT_MPA,
    Q690D_GAMMA_M,
    Q690D_NREF,
    Q690D_RM_MPA,
    Q690D_SN_SLOPE,
)
from scripts.common import normalize_time, parse_float

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 寿命外推参数
# =============================================================================
# 当前 257 条非连续实测记录跨多个时间点，用作代表性工况样本，而不是把
# “出现数据的日期数”当成连续工作天数。因此默认将这一整批代表性样本
# 视为 1 个年度载荷谱样本块。
REPRESENTATIVE_PERIOD_YEARS = 1.0

# 设备在本次代表性载荷谱评估之前已经真实累计的历史 Miner 损伤。
# 当前没有完整连续历史载荷谱可以可靠计算该值，因此默认设为 0，不能把
# 本次代表性样本损伤 D_total 再当作历史损伤重复扣除。
CURRENT_ACCUMULATED_DAMAGE = 0.0
# =============================================================================


@dataclass(frozen=True)
class FatigueParams:
    delta_sigma_C: float = Q690D_FAT_MPA
    N_R: float = Q690D_NREF
    m: float = Q690D_SN_SLOPE
    gamma_m: float = Q690D_GAMMA_M
    Rm: float = Q690D_RM_MPA
    max_stress_at_100_percent: float = FALLBACK_STRESS_AT_100_PERCENT_MPA
    stress_ratio_R: float = FALLBACK_STRESS_RATIO
    max_stress_ratio: float = 0.8
    goodman_denom_min: float = 0.05
    max_goodman_factor: float = 100.0


def safe_float(value, default=0.0):
    v = parse_float(value)
    return default if v is None else float(v)


def _to_mpa(value, params: FatigueParams) -> Optional[float]:
    """将 Neo4j 中的应力尾实体统一转换为 MPa，并过滤明显不合理数值。"""
    if value is None:
        return None
    text = str(value).strip()
    v = parse_float(text)
    if v is None or not np.isfinite(v):
        return None
    lower = text.lower().replace(' ', '')
    if 'gpa' in lower:
        v *= 1000.0
    elif 'mpa' in lower:
        pass
    elif 'kpa' in lower:
        v /= 1000.0
    elif 'pa' in lower:
        v /= 1e6
    elif abs(v) > 1e6:
        # 有限元云图科学计数法通常以 Pa 表示。
        v /= 1e6

    # 不自动“猜测”小数点错误。若结果明显不合理，则拒绝该值。
    sanity_limit = max(1000.0, params.Rm * 1.2)
    if abs(v) > sanity_limit:
        return None
    return float(v)


def query_stress_from_graph(
    start_time: str,
    end_time: str,
    driver,
    params: FatigueParams,
) -> Tuple[Optional[float], Optional[float]]:
    if driver is None:
        return None, None
    cypher = """
    MATCH (t:TimePoint)-[r]->(e:Entity)
    WHERE t.time >= $start AND t.time <= $end
      AND (type(r) IN ['SMX','SMN'] OR coalesce(r.relation,'') IN
           ['SMX','SMN','最大应力','最小应力','最大应力 (SMX)','最小应力 (SMN)'])
    RETURN type(r) AS rtype, r.relation AS relation, e.value AS value
    """
    smx: List[float] = []
    smn: List[float] = []
    try:
        with driver.session() as session:
            for rec in session.run(
                cypher,
                start=normalize_time(start_time),
                end=normalize_time(end_time),
            ):
                rel = str(rec.get('relation') or rec.get('rtype') or '')
                val = _to_mpa(rec.get('value'), params)
                if val is None:
                    continue
                if 'SMX' in rel or '最大' in rel:
                    smx.append(val)
                if 'SMN' in rel or '最小' in rel:
                    smn.append(val)
    except Exception:
        return None, None
    if not smx or not smn:
        return None, None
    smax, smin = max(smx), min(smn)
    if not np.isfinite(smax) or not np.isfinite(smin) or smax <= 0 or smax <= smin:
        return None, None
    return float(smax), float(smin)


def load_to_stress_from_load_pct(
    pct: float,
    params: FatigueParams,
) -> Tuple[float, float]:
    """保留原型代码的力矩/载荷百分比到应力的回退映射。"""
    pct = max(0.0, float(pct))
    sigma_max = (pct / 100.0) * params.max_stress_at_100_percent
    max_allowed = params.Rm * params.max_stress_ratio
    if sigma_max > max_allowed:
        sigma_max = max_allowed
    sigma_min = params.stress_ratio_R * sigma_max
    return float(sigma_max), float(sigma_min)


def equivalent_stress_range(
    sigma_max: float,
    sigma_min: float,
    params: FatigueParams,
) -> float:
    """保留原有线性 Haigh/Goodman 平均应力修正。"""
    if not np.isfinite(sigma_max) or not np.isfinite(sigma_min) or sigma_max <= 0:
        return 0.0
    sigma_a = (sigma_max - sigma_min) / 2.0
    sigma_m = (sigma_max + sigma_min) / 2.0
    if sigma_a <= 0:
        return 0.0
    if sigma_m <= 0:
        sigma_a_eq = sigma_a
    else:
        if sigma_m >= params.Rm:
            return 2.0 * params.Rm
        denom = 1.0 - sigma_m / params.Rm
        denom = max(params.goodman_denom_min, denom)
        factor = min(params.max_goodman_factor, 1.0 / denom)
        sigma_a_eq = sigma_a * factor
    delta_sigma_eq = 2.0 * sigma_a_eq
    return min(delta_sigma_eq, 2.0 * params.Rm)


def cycles_to_failure(delta_sigma: float, params: FatigueParams) -> float:
    if not np.isfinite(delta_sigma) or delta_sigma <= 0:
        return float('inf')
    return params.N_R * ((params.delta_sigma_C / delta_sigma) ** params.m)


def cumulative_damage(
    stress_ranges: List[float],
    frequencies: List[int],
    params: FatigueParams,
) -> Tuple[float, List[Dict]]:
    """保留原型代码的应力谱 Miner 累积损伤计算。"""
    if not stress_ranges or not frequencies:
        return 0.0, []
    valid = [
        (float(ds), int(n))
        for ds, n in zip(stress_ranges, frequencies)
        if np.isfinite(ds) and ds > 0 and n > 0
    ]
    if not valid:
        return 0.0, []
    stress_ranges, frequencies = zip(*valid)
    ds_max = max(stress_ranges)
    if ds_max <= 0:
        return 0.0, []

    A = (
        params.gamma_m
        * (ds_max ** params.m)
        / ((params.delta_sigma_C ** params.m) * params.N_R)
    )
    sum_term = 0.0
    details = []
    for ds, n in zip(stress_ranges, frequencies):
        ratio = (ds / ds_max) ** params.m
        contrib = n * ratio
        sum_term += contrib
        details.append({
            'Δσ_i (MPa)': ds,
            '频次 n_i': n,
            '(Δσ_i/Δσ_max)^m': ratio,
            'n_i*(...)^m': contrib,
        })
    D = A * sum_term
    for d in details:
        d['单级损伤贡献'] = A * d['n_i*(...)^m']
    return float(D), details


def calculate_representative_spectrum_life(
    sample_damage: float,
    sample_cycles: int,
    representative_period_years: float = REPRESENTATIVE_PERIOD_YEARS,
    current_accumulated_damage: float = CURRENT_ACCUMULATED_DAMAGE,
) -> Dict[str, float]:
    """由代表性应力谱计算年损伤率、等效剩余循环和剩余寿命。

    该函数专门替代原代码中：
        D_annual = D_total * (365 / working_days)

    对非连续抽样数据，不再统计“出现数据的日期数”来进行年化。

    参数
    ----
    sample_damage:
        当前代表性样本块计算得到的 Miner 累积损伤 D_total。
    sample_cycles:
        当前样本块中识别出的雨流等效循环数。
    representative_period_years:
        该代表性载荷谱样本块所代表的周期长度，当前默认 1 年。
    current_accumulated_damage:
        本次评估之前已经真实累计的历史 Miner 损伤，默认 0。

    返回
    ----
    annual_damage_rate:
        年等效损伤率。
    damage_per_equivalent_cycle:
        单位等效循环对应的平均 Miner 损伤。
    annual_equivalent_cycles:
        由代表样本周期得到的年等效循环数。
    predicted_total_equivalent_cycles:
        从 D=0 到 D=1，在当前代表性应力谱下的理论总等效循环数。
    remaining_equivalent_cycles:
        扣除真实历史累计损伤后剩余的等效循环数。
    predicted_total_life_years:
        当前代表性工况保持不变时，从新机 D=0 到 D=1 的理论总寿命。
    remaining_life_years:
        当前真实历史损伤状态下的剩余寿命。
    """
    if not np.isfinite(sample_damage) or sample_damage < 0:
        raise ValueError('Representative-sample cumulative damage must be a finite non-negative value.')
    if sample_cycles <= 0:
        raise ValueError('Representative-sample cycle count must be greater than 0.')
    if not np.isfinite(representative_period_years) or representative_period_years <= 0:
        raise ValueError('Representative-sample period must be greater than 0 years.')
    if (
        not np.isfinite(current_accumulated_damage)
        or current_accumulated_damage < 0
        or current_accumulated_damage >= 1
    ):
        raise ValueError('Historical cumulative damage must satisfy 0 <= D_current < 1.')

    if sample_damage == 0:
        return {
            'annual_damage_rate': 0.0,
            'damage_per_equivalent_cycle': 0.0,
            'annual_equivalent_cycles': sample_cycles / representative_period_years,
            'predicted_total_equivalent_cycles': float('inf'),
            'remaining_equivalent_cycles': float('inf'),
            'predicted_total_life_years': float('inf'),
            'remaining_life_years': float('inf'),
            'remaining_life_days': float('inf'),
        }

    # 当前样本作为一个代表性载荷谱块，不依据离散日期数量进行放大。
    annual_damage_rate = sample_damage / representative_period_years

    # 将整个代表性谱块等价成平均每个雨流循环的损伤，仅用于寿命尺度转换。
    damage_per_equivalent_cycle = sample_damage / float(sample_cycles)
    annual_equivalent_cycles = sample_cycles / representative_period_years

    predicted_total_equivalent_cycles = 1.0 / damage_per_equivalent_cycle
    remaining_capacity = max(0.0, 1.0 - current_accumulated_damage)
    remaining_equivalent_cycles = (
        remaining_capacity / damage_per_equivalent_cycle
        if damage_per_equivalent_cycle > 0
        else float('inf')
    )

    predicted_total_life_years = (
        1.0 / annual_damage_rate
        if annual_damage_rate > 0
        else float('inf')
    )
    remaining_life_years = (
        remaining_capacity / annual_damage_rate
        if annual_damage_rate > 0
        else float('inf')
    )
    remaining_life_days = remaining_life_years * 365.0

    return {
        'annual_damage_rate': float(annual_damage_rate),
        'damage_per_equivalent_cycle': float(damage_per_equivalent_cycle),
        'annual_equivalent_cycles': float(annual_equivalent_cycles),
        'predicted_total_equivalent_cycles': float(predicted_total_equivalent_cycles),
        'remaining_equivalent_cycles': float(remaining_equivalent_cycles),
        'predicted_total_life_years': float(predicted_total_life_years),
        'remaining_life_years': float(remaining_life_years),
        'remaining_life_days': float(remaining_life_days),
    }


def _open_driver():
    if GraphDatabase is None or not NEO4J_PASSWORD:
        return None
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        with driver.session() as session:
            session.run('RETURN 1').consume()
        return driver
    except Exception:
        return None


def run_damage_analysis(
    rainflow_excel: str,
    output_prefix: str = '主臂疲劳分析',
    use_graph_stress: bool = True,
    allow_stress_fallback: bool = False,
    params: FatigueParams | None = None,
    representative_period_years: float = REPRESENTATIVE_PERIOD_YEARS,
    current_accumulated_damage: float = CURRENT_ACCUMULATED_DAMAGE,
) -> Tuple[str, str, str]:
    params = params or FatigueParams()
    rainflow_path = Path(rainflow_excel)
    if not rainflow_path.exists():
        raise FileNotFoundError(rainflow_excel)

    driver = _open_driver() if use_graph_stress else None
    df_rf = pd.read_excel(rainflow_path)
    required_cols = ['循环ID', '载荷百分比', '时间', '实际吊重量', '额定吊重量']
    missing = [c for c in required_cols if c not in df_rf.columns]
    if missing:
        raise ValueError(f"Rainflow result missing required columns: {missing}")

    cycle_detail_file = str(rainflow_path).replace('.xlsx', '_循环明细.xlsx')
    if Path(cycle_detail_file).exists():
        df_cycles = pd.read_excel(cycle_detail_file)
        if '起始时间' not in df_cycles.columns or '结束时间' not in df_cycles.columns:
            df_cycles = pd.DataFrame()
    else:
        df_cycles = pd.DataFrame()

    if df_cycles.empty:
        valid = df_rf[df_rf['循环ID'].notna()].copy()
        if valid.empty:
            if driver is not None:
                driver.close()
            raise ValueError('No rainflow cycles are available for fatigue calculation.')
        df_cycles = valid.groupby('循环ID').agg(
            起始时间=('时间', 'first'),
            结束时间=('时间', 'last'),
        ).reset_index()

    load_pct_map = df_rf.groupby('循环ID')['载荷百分比'].mean().to_dict()
    df_cycles['载荷百分比'] = df_cycles['循环ID'].map(load_pct_map)

    # -------------------------------------------------------------------------
    # 注意：这里不再根据“有数据的日期数量”计算 working_days。
    # 数据时间不连续，日期数量不能代表连续工作天数，也不能用于 365/working_days
    # 的线性年化。寿命外推统一在 calculate_representative_spectrum_life()
    # 中按代表性载荷谱周期处理。
    # -------------------------------------------------------------------------

    cycles_data = []
    graph_cycles = 0
    fallback_cycles = 0

    for _, row in df_cycles.iterrows():
        cycle_id = row['循环ID']
        start_time = str(row['起始时间'])
        end_time = str(row['结束时间'])
        smax = smin = None
        source = 'load-percentage fallback'

        if driver is not None:
            smax, smin = query_stress_from_graph(
                start_time,
                end_time,
                driver,
                params,
            )
            if smax is not None and smin is not None:
                source = 'TC-MMKG SMX/SMN'
                graph_cycles += 1

        if smax is None or smin is None:
            if not allow_stress_fallback:
                if driver is not None:
                    driver.close()
                raise ValueError(
                    f'TC-MMKG stress is missing for cycle {cycle_id} '
                    f'({start_time} -> {end_time}). '
                    'The current dataset must complete text/image triple extraction, '
                    'cross-modal fusion, and Neo4j ingestion before fatigue damage '
                    'is calculated.'
                )
            pct = safe_float(row.get('载荷百分比', 0.0))
            smax, smin = load_to_stress_from_load_pct(pct, params)
            fallback_cycles += 1

        cycles_data.append({
            '循环ID': cycle_id,
            '起始时间': start_time,
            '结束时间': end_time,
            '载荷百分比': safe_float(row.get('载荷百分比', 0.0)),
            '应力最大值 (MPa)': smax,
            '应力最小值 (MPa)': smin,
            '应力来源': source,
        })

    cycles_df = pd.DataFrame(cycles_data)
    if cycles_df.empty:
        if driver is not None:
            driver.close()
        raise ValueError('No cycles extracted, please check rainflow result data.')

    cycles_df['Δσ_i (MPa)'] = cycles_df.apply(
        lambda r: equivalent_stress_range(
            r['应力最大值 (MPa)'],
            r['应力最小值 (MPa)'],
            params,
        ),
        axis=1,
    )
    cycles_df['单循环损伤'] = cycles_df['Δσ_i (MPa)'].apply(
        lambda ds: 1.0 / cycles_to_failure(ds, params) if ds > 0 else 0.0
    )

    grouped = cycles_df.groupby(
        cycles_df['Δσ_i (MPa)'].round(0)
    ).agg(
        Δσ_i=('Δσ_i (MPa)', 'first'),
        频次=('循环ID', 'count'),
    ).reset_index(drop=True).sort_values('Δσ_i', ascending=False)

    stress_ranges = grouped['Δσ_i'].tolist()
    frequencies = grouped['频次'].tolist()
    D_total, damage_details = cumulative_damage(
        stress_ranges,
        frequencies,
        params,
    )
    total_cycles = int(sum(frequencies))

    # -------------------------------------------------------------------------
    # 修正后的寿命外推
    # -------------------------------------------------------------------------
    # 不再使用：D_total * (365 / working_days)
    # 当前零散、不连续的工况记录作为一个代表性年度载荷谱样本块。
    life_result = calculate_representative_spectrum_life(
        sample_damage=D_total,
        sample_cycles=total_cycles,
        representative_period_years=representative_period_years,
        current_accumulated_damage=current_accumulated_damage,
    )

    D_annual = life_result['annual_damage_rate']
    damage_per_cycle = life_result['damage_per_equivalent_cycle']
    annual_equivalent_cycles = life_result['annual_equivalent_cycles']
    predicted_total_equivalent_cycles = life_result['predicted_total_equivalent_cycles']
    remaining_equivalent_cycles = life_result['remaining_equivalent_cycles']
    predicted_total_life_years = life_result['predicted_total_life_years']
    remaining_years = life_result['remaining_life_years']
    remaining_days = life_result['remaining_life_days']

    def _fmt_number(value: float, digits: int = 1) -> str:
        if not np.isfinite(value):
            return '∞'
        return f'{value:.{digits}f}'

    report_text = (
        f"Total cycles: {total_cycles}\n"
        f"Cumulative damage: {D_total:.6f}\n"
        f"Annual damage rate: {D_annual:.4f}\n"
        f"Remaining life (years): {_fmt_number(remaining_years, 1)}\n"
        f"Remaining life (days): {_fmt_number(remaining_days, 0)}\n"
        f"Stress source: TC-MMKG cycles={graph_cycles}, fallback cycles={fallback_cycles}\n"
        f"Life extrapolation: representative annual load spectrum; no scaling by discontinuous sampling dates"
    )

    result_excel = Path(OUTPUT_DIR) / f"{output_prefix}_损伤结果.xlsx"
    with pd.ExcelWriter(result_excel, engine='openpyxl') as writer:
        cycles_df.to_excel(writer, sheet_name='各循环Δσ', index=False)
        pd.DataFrame(damage_details).to_excel(
            writer,
            sheet_name='应力级统计',
            index=False,
        )
        pd.DataFrame([{
            '代表性样本累积损伤': D_total,
            '年损伤率': D_annual,
            '代表性样本周期(年)': representative_period_years,
            '样本等效循环数': total_cycles,
            '单位等效循环平均损伤': damage_per_cycle,
            '年等效循环数': annual_equivalent_cycles,
            '理论总等效循环数': predicted_total_equivalent_cycles,
            '剩余等效循环数': remaining_equivalent_cycles,
            '理论总寿命(年)': predicted_total_life_years,
            '历史累计损伤': current_accumulated_damage,
            '剩余寿命(年)': remaining_years,
            '剩余寿命(天)': remaining_days,
            '寿命外推方法': '代表性载荷谱法（不按非连续采样日期数年化）',
            'Δσ_max(MPa)': max(stress_ranges) if stress_ranges else 0,
            'Δσ_C': params.delta_sigma_C,
            'N_R': params.N_R,
            'm': params.m,
            'γ_m': params.gamma_m,
            '应力比R': params.stress_ratio_R,
            '图谱应力循环数': graph_cycles,
            '回退应力循环数': fallback_cycles,
        }]).to_excel(writer, sheet_name='寿命结果', index=False)

    # -------------------------------------------------------------------------
    # 每行瞬时损伤：保留原有输出结构，供 damage_to_neo4j.py 使用。
    # -------------------------------------------------------------------------
    df_original = pd.read_excel(rainflow_path)

    def compute_row_damage(row):
        time_str = str(row['时间'])
        smax = smin = None
        if driver is not None:
            smax, smin = query_stress_from_graph(
                time_str,
                time_str,
                driver,
                params,
            )
        if smax is None or smin is None:
            pct = safe_float(row.get('载荷百分比', 0.0))
            smax, smin = load_to_stress_from_load_pct(pct, params)
        ds_eq = equivalent_stress_range(smax, smin, params)
        if ds_eq <= 0:
            return 0.0
        nf = cycles_to_failure(ds_eq, params)
        return 1.0 / nf if np.isfinite(nf) and nf > 0 else 0.0

    df_original['瞬时损伤'] = df_original.apply(compute_row_damage, axis=1)
    per_row_excel = Path(OUTPUT_DIR) / f"{output_prefix}_每行损伤.xlsx"
    df_original.to_excel(per_row_excel, index=False)

    if driver is not None:
        driver.close()

    # -------------------------------------------------------------------------
    # 原有绘图方法保持不变：布局、曲线、bins、坐标范围、颜色和分辨率均不改。
    # -------------------------------------------------------------------------
    monthly_damage = None
    if Path(cycle_detail_file).exists():
        try:
            df_cycles_detail = pd.read_excel(cycle_detail_file)
            df_cycles_detail = df_cycles_detail.merge(
                cycles_df[['循环ID', 'Δσ_i (MPa)']],
                on='循环ID',
                how='left',
            )
            df_cycles_detail['起始时间'] = pd.to_datetime(
                df_cycles_detail['起始时间'],
                errors='coerce',
            )
            df_cycles_detail['月份'] = (
                df_cycles_detail['起始时间']
                .dt.to_period('M')
                .dt.start_time
            )
            df_cycles_detail = df_cycles_detail.dropna(
                subset=['月份', 'Δσ_i (MPa)']
            )
            df_cycles_detail['循环损伤'] = df_cycles_detail[
                'Δσ_i (MPa)'
            ].apply(
                lambda ds: 1.0 / cycles_to_failure(ds, params)
                if ds > 0
                else 0.0
            )
            monthly_damage = (
                df_cycles_detail
                .groupby('月份')['循环损伤']
                .sum()
                .reset_index()
            )
            monthly_damage = monthly_damage.sort_values('月份')
            monthly_damage['累积损伤'] = monthly_damage['循环损伤'].cumsum()
        except Exception:
            monthly_damage = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(
        'Truck-Crane Boom Fatigue Damage Analysis (Miner + Goodman)',
        fontsize=14,
        fontweight='bold',
    )

    if monthly_damage is not None and not monthly_damage.empty:
        ax1.plot(
            monthly_damage['月份'],
            monthly_damage['累积损伤'],
            'b-o',
            markersize=4,
        )
        for x, y in zip(
            monthly_damage['月份'],
            monthly_damage['累积损伤'],
        ):
            ax1.annotate(
                f'{y:.4f}',
                (x, y),
                textcoords='offset points',
                xytext=(0, 8),
                ha='center',
                fontsize=7,
            )
        month_positions = monthly_damage['月份']
        month_labels = [m.strftime('%b') for m in month_positions]
        ax1.set_xticks(month_positions)
        ax1.set_xticklabels(month_labels, rotation=45)
    else:
        ax1.text(
            0.5,
            0.5,
            'No valid cycle data',
            ha='center',
            va='center',
        )

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Cumulative Damage D')
    ax1.set_title('Monthly Miner Cumulative Damage')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(direction='in')

    bins = np.arange(0, 475, 25)
    ax2.hist(
        cycles_df['Δσ_i (MPa)'],
        bins=bins,
        edgecolor='black',
        color='skyblue',
        align='mid',
        rwidth=0.8,
    )
    ax2.set_xlabel('Equivalent Stress Range Δσ (MPa)')
    ax2.set_ylabel('Cycle Count')
    ax2.set_title('Equivalent Stress Range Distribution')
    ax2.set_xlim(0, 450)
    ax2.set_xticks(np.arange(0, 451, 50))
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(direction='in')

    plt.tight_layout()
    chart_path = Path(OUTPUT_DIR) / f"{output_prefix}_综合图表.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return report_text, str(chart_path), str(result_excel)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        raise SystemExit('Usage: python try6_use.py <rainflow-results.xlsx>')
    print(run_damage_analysis(sys.argv[1]))
