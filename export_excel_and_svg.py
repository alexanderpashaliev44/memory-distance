#!/usr/bin/env python3
"""
Generates from the raw hardware CSV telemetry:
1. Native Excel Workbook (`fermi_memory_distance.xlsx`) with live formulas,
   conditional formatting data bars, and native interactive Excel charts.
2. Clean IEEE/ACM-style vector `.svg` charts (`charts/*.svg`) suitable for GitHub.
"""

import csv
import os
import shutil
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.formatting.rule import DataBarRule
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_range(ws, cell_range, font=None, fill=None, border=None, alignment=None):
    for row in ws[cell_range]:
        for cell in row:
            if font:
                cell.font = font
            if fill:
                cell.fill = fill
            if border:
                cell.border = border
            if alignment:
                cell.alignment = alignment


def auto_fit_columns(ws, min_width=12, max_width=42):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if val_str.startswith("="):
                val_str = "123456.78"
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 4))


def build_excel_workbook(lat_rows, bw_rows, roof_rows):
    wb = openpyxl.Workbook()

    # Shared styles
    font_title = Font(name="Segoe UI", size=14, bold=True, color="FFFFFF")
    font_subtitle = Font(name="Segoe UI", size=10, italic=True, color="E2E8F0")
    font_sec = Font(name="Segoe UI", size=11, bold=True, color="1E293B")
    font_th = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    font_td = Font(name="Segoe UI", size=10, bold=False, color="0F172A")
    font_td_bold = Font(name="Segoe UI", size=10, bold=True, color="0F172A")

    fill_navy = PatternFill("solid", fgColor="0F172A")
    fill_slate = PatternFill("solid", fgColor="1E293B")
    fill_th = PatternFill("solid", fgColor="2563EB")
    fill_zebra = PatternFill("solid", fgColor="F8FAFC")
    fill_white = PatternFill("solid", fgColor="FFFFFF")
    fill_callout = PatternFill("solid", fgColor="EFF6FF")

    thin_side = Side(border_style="thin", color="CBD5E1")
    border_grid = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    align_right = Alignment(horizontal="right", vertical="center")
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # ==========================================================================
    # SHEET 1: OVERVIEW & EXECUTIVE DASHBOARD
    # ==========================================================================
    ws_dash = wb.active
    ws_dash.title = "1_Dashboard"
    ws_dash.views.sheetView[0].showGridLines = True

    ws_dash.merge_cells("A1:H1")
    ws_dash["A1"] = "GPU MEMORY-DISTANCE & ROOFLINE BENCHMARK (NVIDIA QUADRO 2000)"
    style_range(ws_dash, "A1:H1", font=font_title, fill=fill_navy, alignment=align_left)
    ws_dash.row_dimensions[1].height = 30

    ws_dash.merge_cells("A2:H2")
    ws_dash["A2"] = (
        'Core Thesis: "Memory proximity buys BANDWIDTH and ENERGY-PER-BIT; parallelism hides LATENCY."'
    )
    style_range(ws_dash, "A2:H2", font=font_subtitle, fill=fill_slate, alignment=align_left)
    ws_dash.row_dimensions[2].height = 22

    # Key Findings Summary Table
    ws_dash["A4"] = "EMPIRICAL PROOF SUMMARY (NVIDIA QUADRO 2000: 192 CORES @ 1.25 GHz, 1 GB GDDR5 @ 41.6 GB/s)"
    ws_dash["A4"].font = font_sec

    headers_kpi = ["GPU Memory Tier", "Working Set", "Measured Latency (ns)", "Measured Cycles (@1.25G)", "Measured Bandwidth (GB/s)", "Theoretical Peak (GB/s)", "Efficiency (%)", "Architectural Proof"]
    for c_idx, h in enumerate(headers_kpi, 1):
        cell = ws_dash.cell(row=5, column=c_idx, value=h)
        cell.font = font_th
        cell.fill = fill_th
        cell.alignment = align_center
        cell.border = border_grid
    ws_dash.row_dimensions[5].height = 24

    reg_lat = float(lat_rows[0]["gpu_register_reference_ns"])
    sram_lat = float(lat_rows[0]["gpu_sram_reference_ns"])
    l1_lat = float(next(r["gpu_latency_ns"] for r in lat_rows if int(r["footprint_kb"]) == 16))
    l2_lat = float(next(r["gpu_latency_ns"] for r in lat_rows if int(r["footprint_kb"]) == 128))
    vram_lat = float(lat_rows[-1]["gpu_latency_ns"])

    bw_map = {r["tier"]: r for r in bw_rows}
    sram_bw = float(bw_map["GPU Shared SRAM (48 KB/SM)"]["measured_gbs"])
    l2_bw = float(bw_map["GPU L2 Cache (256 KB)"]["measured_gbs"])
    vram_bw = float(bw_map["GPU GDDR5 VRAM (128-bit)"]["measured_gbs"])
    pcie_bw = float(bw_map["PCIe 2.0 x16 Host<->GPU"]["measured_gbs"])

    summary_rows = [
        ["GPU Register / ALU", "Scalar Regs", reg_lat, round(reg_lat * 1.25, 1), "—", "—", "—", "9-stage core ALU dependency baseline"],
        ["GPU Shared SRAM (48 KB/SM)", "16 KB (groupshared)", sram_lat, round(sram_lat * 1.25, 1), sram_bw, 160.0, round(sram_bw / 160.0, 3), f"Deterministic on-chip scratchpad: {sram_bw / vram_bw:.2f}x BW vs GDDR5"],
        ["GPU L1 Cache (PreferL1 CS)", "4 KB - 48 KB", l1_lat, round(l1_lat * 1.25, 1), "—", "—", "—", "Driver auto-enables 48 KB L1 when groupshared = 0"],
        ["GPU L2 Cache (256 KB Unified)", "64 KB - 256 KB", l2_lat, round(l2_lat * 1.25, 1), l2_bw, 80.0, round(l2_bw / 80.0, 3), "On-die L2 cushion before off-chip DRAM cliff"],
        ["GPU GDDR5 VRAM (128-bit Bus)", "384 KB - 32 MB", vram_lat, round(vram_lat * 1.25, 1), vram_bw, 41.6, round(vram_bw / 41.6, 3), f"Off-chip DRAM latency cliff ({vram_lat:.1f} ns; 11.7x slower than Shared SRAM)"],
        ["PCIe 2.0 x16 Interconnect", "4 MB Staging", "—", "—", pcie_bw, 8.0, round(pcie_bw / 8.0, 3), "Off-die interconnect bottleneck for low-reuse data"],
    ]

    for r_offset, r_data in enumerate(summary_rows):
        r_num = 6 + r_offset
        ws_dash.row_dimensions[r_num].height = 20
        fill_row = fill_zebra if r_offset % 2 == 1 else fill_white
        for c_idx, val in enumerate(r_data, 1):
            cell = ws_dash.cell(row=r_num, column=c_idx, value=val)
            cell.font = font_td_bold if c_idx == 1 else font_td
            cell.fill = fill_row
            cell.border = border_grid
            if isinstance(val, float):
                if c_idx == 7:
                    cell.number_format = "0.0%"
                else:
                    cell.number_format = "0.00" if c_idx in (3, 5) else "0.0"
                cell.alignment = align_right
            elif c_idx in (1, 2, 8):
                cell.alignment = align_left
            else:
                cell.alignment = align_center

    auto_fit_columns(ws_dash)
    ws_dash.column_dimensions["A"].width = 28
    ws_dash.column_dimensions["H"].width = 46

    # ==========================================================================
    # SHEET 2: LATENCY LADDER
    # ==========================================================================
    ws_lat = wb.create_sheet(title="2_Latency_Ladder")
    ws_lat.views.sheetView[0].showGridLines = True

    ws_lat.merge_cells("A1:G1")
    ws_lat["A1"] = "POINTER-CHASING LATENCY LADDER (GPU REGISTERS -> SHARED SRAM -> L1 -> L2 -> GDDR5)"
    style_range(ws_lat, "A1:G1", font=font_title, fill=fill_navy, alignment=align_left)
    ws_lat.row_dimensions[1].height = 28

    lat_headers = [
        "Footprint (KB)",
        "MEASURED GPU Global Latency (ns)",
        "MEASURED GPU Shared SRAM Ref (ns)",
        "MEASURED GPU Register Ref (ns)",
        "GPU Shader Cycles (@1.25 GHz)",
        "GPU Core Cycles (@625 MHz)",
        "Silicon Memory Regime",
    ]
    for c_idx, h in enumerate(lat_headers, 1):
        cell = ws_lat.cell(row=3, column=c_idx, value=h)
        cell.font = font_th
        cell.fill = fill_th
        cell.alignment = align_center
        cell.border = border_grid
    ws_lat.row_dimensions[3].height = 24

    for idx, r in enumerate(lat_rows):
        row_num = 4 + idx
        ws_lat.row_dimensions[row_num].height = 19
        fill_row = fill_zebra if idx % 2 == 1 else fill_white
        vals = [
            int(r["footprint_kb"]),
            float(r["gpu_latency_ns"]),
            float(r["gpu_sram_reference_ns"]),
            float(r["gpu_register_reference_ns"]),
            float(r["gpu_shader_cycles_1250mhz"]),
            float(r["gpu_core_cycles_625mhz"]),
            r["regime"],
        ]
        for c_idx, v in enumerate(vals, 1):
            cell = ws_lat.cell(row=row_num, column=c_idx, value=v)
            cell.font = font_td
            cell.fill = fill_row
            cell.border = border_grid
            if c_idx == 1:
                cell.number_format = "#,##0"
                cell.alignment = align_right
            elif c_idx in (2, 3, 4):
                cell.number_format = "0.00"
                cell.alignment = align_right
            elif c_idx in (5, 6):
                cell.number_format = "0.0"
                cell.alignment = align_right
            else:
                cell.alignment = align_left

    last_lat_row = 3 + len(lat_rows)
    ws_lat.conditional_formatting.add(
        f"B4:B{last_lat_row}",
        DataBarRule(start_type="num", start_value=0, end_type="num", end_value=550, color="3B82F6")
    )
    auto_fit_columns(ws_lat)

    # Create Native Excel LineChart for Latency Ladder
    def make_latency_chart():
        chart = LineChart()
        chart.title = "Pointer-Chasing Latency Ladder (L1 -> L2 -> GDDR5 vs. On-Chip SRAM)"
        chart.style = 10
        chart.y_axis.title = "Dependent Access Latency (ns)"
        chart.x_axis.title = "Working Set Footprint (KB)"
        chart.width = 22
        chart.height = 12
        data_ref = Reference(ws_lat, min_col=2, min_row=3, max_col=4, max_row=last_lat_row)
        cats_ref = Reference(ws_lat, min_col=1, min_row=4, max_row=last_lat_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        for s in chart.series:
            s.graphicalProperties.line.width = 25000
        return chart

    ws_lat.add_chart(make_latency_chart(), "J3")
    ws_dash.add_chart(make_latency_chart(), "A15")

    # ==========================================================================
    # SHEET 3: EXPERIMENT B — BANDWIDTH LADDER
    # ==========================================================================
    ws_bw = wb.create_sheet(title="3_Bandwidth_Ladder")
    ws_bw.views.sheetView[0].showGridLines = True

    ws_bw.merge_cells("A1:D1")
    ws_bw["A1"] = "EXPERIMENT B: MEMORY PROXIMITY BANDWIDTH LADDER (MEASURED vs. THEORETICAL PEAK)"
    style_range(ws_bw, "A1:D1", font=font_title, fill=fill_navy, alignment=align_left)
    ws_bw.row_dimensions[1].height = 28

    bw_headers = ["Memory Tier", "MEASURED Bandwidth (GB/s)", "THEORETICAL PEAK (GB/s)", "Sustained Efficiency (%)"]
    for c_idx, h in enumerate(bw_headers, 1):
        cell = ws_bw.cell(row=3, column=c_idx, value=h)
        cell.font = font_th
        cell.fill = fill_th
        cell.alignment = align_center
        cell.border = border_grid
    ws_bw.row_dimensions[3].height = 24

    for idx, r in enumerate(bw_rows):
        row_num = 4 + idx
        ws_bw.row_dimensions[row_num].height = 20
        fill_row = fill_zebra if idx % 2 == 1 else fill_white
        c1 = ws_bw.cell(row=row_num, column=1, value=r["tier"])
        c2 = ws_bw.cell(row=row_num, column=2, value=float(r["measured_gbs"]))
        c3 = ws_bw.cell(row=row_num, column=3, value=float(r["theoretical_peak_gbs"]))
        c4 = ws_bw.cell(row=row_num, column=4, value=f"=B{row_num}/C{row_num}")

        for c in (c1, c2, c3, c4):
            c.font = font_td_bold if c == c1 else font_td
            c.fill = fill_row
            c.border = border_grid
        c1.alignment = align_left
        c2.number_format = "0.00"; c2.alignment = align_right
        c3.number_format = "0.0"; c3.alignment = align_right
        c4.number_format = "0.0%"; c4.alignment = align_right

    last_bw_row = 3 + len(bw_rows)
    ws_bw.conditional_formatting.add(
        f"B4:B{last_bw_row}",
        DataBarRule(start_type="num", start_value=0, end_type="num", end_value=160, color="10B981")
    )
    auto_fit_columns(ws_bw)

    def make_bw_chart():
        chart = BarChart()
        chart.type = "col"
        chart.style = 11
        chart.grouping = "clustered"
        chart.overlap = -15
        chart.title = "Memory Proximity Bandwidth Ladder: MEASURED vs. THEORETICAL PEAK (GB/s)"
        chart.y_axis.title = "Bandwidth (GB/s)"
        chart.x_axis.title = "Memory Hierarchy Tier"
        chart.width = 20
        chart.height = 12
        data_ref = Reference(ws_bw, min_col=2, min_row=3, max_col=3, max_row=last_bw_row)
        cats_ref = Reference(ws_bw, min_col=1, min_row=4, max_row=last_bw_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        return chart

    ws_bw.add_chart(make_bw_chart(), "F3")
    ws_dash.add_chart(make_bw_chart(), "F15")

    # ==========================================================================
    # SHEET 4: EXPERIMENT C — DISTANCE-ROOFLINE
    # ==========================================================================
    ws_rf = wb.create_sheet(title="4_Distance_Roofline")
    ws_rf.views.sheetView[0].showGridLines = True

    ws_rf.merge_cells("A1:F1")
    ws_rf["A1"] = "EXPERIMENT C: EMPIRICAL DISTANCE-ROOFLINE (QUADRO 2000: 192 CORES, 480 GFLOPS PEAK)"
    style_range(ws_rf, "A1:F1", font=font_title, fill=fill_navy, alignment=align_left)
    ws_rf.row_dimensions[1].height = 28

    rf_headers = [
        "Arithmetic Intensity (FLOP/Byte)",
        "MEASURED Throughput (GFLOPS)",
        "THEORETICAL Roofline (GFLOPS)",
        "Sustained GDDR5 BW (GB/s)",
        "Roofline Efficiency (%)",
        "Bound Regime",
    ]
    for c_idx, h in enumerate(rf_headers, 1):
        cell = ws_rf.cell(row=3, column=c_idx, value=h)
        cell.font = font_th
        cell.fill = fill_th
        cell.alignment = align_center
        cell.border = border_grid
    ws_rf.row_dimensions[3].height = 24

    for idx, r in enumerate(roof_rows):
        row_num = 4 + idx
        ws_rf.row_dimensions[row_num].height = 20
        fill_row = fill_zebra if idx % 2 == 1 else fill_white
        c1 = ws_rf.cell(row=row_num, column=1, value=float(r["arithmetic_intensity_flop_per_byte"]))
        c2 = ws_rf.cell(row=row_num, column=2, value=float(r["measured_gflops"]))
        c3 = ws_rf.cell(row=row_num, column=3, value=f"=MIN(480.0, A{row_num}*41.6)")
        c4 = ws_rf.cell(row=row_num, column=4, value=float(r["sustained_bandwidth_gbs"]))
        c5 = ws_rf.cell(row=row_num, column=5, value=f"=B{row_num}/C{row_num}")
        c6 = ws_rf.cell(row=row_num, column=6, value=r["regime"])

        for c in (c1, c2, c3, c4, c5, c6):
            c.font = font_td
            c.fill = fill_row
            c.border = border_grid
        c1.number_format = "0.00"; c1.alignment = align_right
        c2.number_format = "0.00"; c2.alignment = align_right
        c3.number_format = "0.00"; c3.alignment = align_right
        c4.number_format = "0.00"; c4.alignment = align_right
        c5.number_format = "0.0%"; c5.alignment = align_right
        c6.alignment = align_center

    last_rf_row = 3 + len(roof_rows)
    ws_rf.conditional_formatting.add(
        f"B4:B{last_rf_row}",
        DataBarRule(start_type="num", start_value=0, end_type="num", end_value=480, color="2563EB")
    )
    auto_fit_columns(ws_rf)

    def make_rf_chart():
        chart = LineChart()
        chart.title = "Empirical Roofline Model: MEASURED vs. THEORETICAL PEAK (NVIDIA Quadro 2000)"
        chart.style = 12
        chart.y_axis.title = "Throughput (GFLOPS)"
        chart.x_axis.title = "Arithmetic Intensity (FLOP/Byte)"
        chart.width = 22
        chart.height = 12
        data_ref = Reference(ws_rf, min_col=2, min_row=3, max_col=3, max_row=last_rf_row)
        cats_ref = Reference(ws_rf, min_col=1, min_row=4, max_row=last_rf_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        for s in chart.series:
            s.graphicalProperties.line.width = 25000
        return chart

    ws_rf.add_chart(make_rf_chart(), "H3")
    ws_dash.add_chart(make_rf_chart(), "A39")

    xlsx_repo_path = os.path.join(BASE_DIR, "memory_distance.xlsx")
    wb.save(xlsx_repo_path)
    xlsx_desktop_path = os.path.join(os.path.dirname(BASE_DIR), "memory_distance.xlsx")
    shutil.copyfile(xlsx_repo_path, xlsx_desktop_path)
    print(f"[SAVED] Native Excel Workbook -> {xlsx_repo_path}")
    print(f"[SAVED] Desktop Copy          -> {xlsx_desktop_path}")


def build_clean_github_svgs(lat_rows, bw_rows, roof_rows):
    """Generates clean, white-background IEEE/GitHub-style vector SVG charts with centered titles and layout."""
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 9.5

    # 1. Latency Ladder SVG
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=120)
    fp = [float(r["footprint_kb"]) for r in lat_rows]
    gpu_ns = [float(r["gpu_latency_ns"]) for r in lat_rows]
    sram_ns = float(lat_rows[0]["gpu_sram_reference_ns"])
    reg_ns = float(lat_rows[0]["gpu_register_reference_ns"])

    ax.plot(fp, gpu_ns, "o-", color="#1d4ed8", linewidth=2.2, markersize=4.8, label="MEASURED: GPU Global (L1 -> L2 -> GDDR5)")
    ax.axhline(sram_ns, color="#059669", linestyle="-.", linewidth=1.9, label=f"MEASURED: GPU Shared SRAM ({sram_ns:.2f} ns)")
    ax.axhline(reg_ns, color="#7c3aed", linestyle="--", linewidth=1.7, label=f"MEASURED: GPU Register / ALU ({reg_ns:.2f} ns)")
    ax.axvline(48, color="#dc2626", linestyle=":", linewidth=1.3)
    ax.text(48 * 0.55, 430, "48 KB L1", color="#dc2626", fontsize=8.5, fontweight="bold")
    ax.axvline(256, color="#dc2626", linestyle=":", linewidth=1.3)
    ax.text(256 * 1.12, 430, "256 KB L2", color="#dc2626", fontsize=8.5, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working Set Footprint (KB, Log2 Scale)", fontweight="bold")
    ax.set_ylabel("Dependent Access Latency (ns)", fontweight="bold")
    fig.suptitle("Pointer-Chasing Latency Ladder (NVIDIA Quadro 2000)", x=0.5, y=0.96, ha="center", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.86, bottom=0.14)
    svg_lat = os.path.join(CHARTS_DIR, "latency_ladder.svg")
    plt.savefig(svg_lat, format="svg")
    plt.close(fig)

    # 2. Bandwidth Ladder SVG (Vertical Column Chart so x/y margins and title are 100% symmetrically centered)
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=120)
    tiers = [r["tier"].replace(" (", "\n(") for r in bw_rows]
    meas = [float(r["measured_gbs"]) for r in bw_rows]
    theo = [float(r["theoretical_peak_gbs"]) for r in bw_rows]
    x = np.arange(len(tiers))
    w = 0.36

    b1 = ax.bar(x - w / 2, meas, width=w, color="#059669", label="MEASURED Sustained Bandwidth (GB/s)")
    b2 = ax.bar(x + w / 2, theo, width=w, color="#e2e8f0", edgecolor="#475569", hatch="//", label="THEORETICAL PEAK Limit (GB/s)")
    for bar in b1:
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + 3.0, f"{h_val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46")
    for bar in b2:
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + 3.0, f"{h_val:.1f}", ha="center", va="bottom", fontsize=8, color="#475569")

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontweight="bold", fontsize=8.5)
    ax.set_ylim(0, 185)
    ax.set_ylabel("Bandwidth (GB/s)", fontweight="bold")
    ax.set_xlabel("Memory Hierarchy Tier", fontweight="bold")
    fig.suptitle("Memory Proximity Bandwidth Ladder (MEASURED vs. THEORETICAL PEAK)", x=0.5, y=0.96, ha="center", fontweight="bold", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.86, bottom=0.16)
    svg_bw = os.path.join(CHARTS_DIR, "bandwidth_ladder.svg")
    plt.savefig(svg_bw, format="svg")
    plt.close(fig)

    # 3. Roofline Model SVG
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=120)
    ai = [float(r["arithmetic_intensity_flop_per_byte"]) for r in roof_rows]
    m_gf = [float(r["measured_gflops"]) for r in roof_rows]
    t_gf = [float(r["theoretical_roofline_gflops"]) for r in roof_rows]

    ax.plot(ai, t_gf, "--", color="#dc2626", linewidth=2.0, label="THEORETICAL PEAK Roofline (480 GFLOPS / 41.6 GB/s)")
    ax.plot(ai, m_gf, "o-", color="#1d4ed8", linewidth=2.2, markersize=5, label="MEASURED Quadro 2000 FP32 Throughput (GFLOPS)")
    ax.axvline(480.0 / 41.6, color="#d97706", linestyle=":", linewidth=1.4)
    ax.text(11.54 * 1.08, 35, "Ridge Point\n11.54 FLOP/B", color="#b45309", fontsize=8.5, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte, Log2 Scale)", fontweight="bold")
    ax.set_ylabel("FP32 Throughput (GFLOPS, Log10 Scale)", fontweight="bold")
    fig.suptitle("Empirical Distance-Roofline (NVIDIA Quadro 2000, 192 Cores @ 1.25 GHz)", x=0.5, y=0.96, ha="center", fontweight="bold", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.86, bottom=0.14)
    svg_rf = os.path.join(CHARTS_DIR, "roofline_model.svg")
    plt.savefig(svg_rf, format="svg")
    plt.close(fig)
    print(f"[SAVED] Clean Centered GitHub Vector SVGs -> {CHARTS_DIR}")


if __name__ == "__main__":
    lat = load_csv("latency_ladder_raw.csv")
    bw = load_csv("bandwidth_ladder_raw.csv")
    rf = load_csv("roofline_sweep_raw.csv")
    build_excel_workbook(lat, bw, rf)
    build_clean_github_svgs(lat, bw, rf)
