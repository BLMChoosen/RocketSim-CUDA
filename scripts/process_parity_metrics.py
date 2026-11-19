import re
import json

path = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\all_parity_report.md"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

sections = content.split("### Milestone 5 Phase 1 Component-Wise Parity Baseline (")

results = []

for sec in sections[1:]:
    scn_name = sec.split(",")[0].strip()
    
    # Extract Relative Error table
    rel_part = sec.split("#### Relative Error")
    if len(rel_part) < 2:
        continue
    rel_table_text = rel_part[1].split("#### Snapshot Detail")[0]
    
    # Parse rows: Component | Tick 1 (Med) | Tick 1 (P95) | Tick 10 (Med) | Tick 10 (P95) | Tick 60 (Med) | Tick 60 (P95)
    rows = []
    for line in rel_table_text.strip().splitlines():
        if not line.strip().startswith("|") or ":---" in line or "Component" in line:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 7:
            comp = parts[0]
            try:
                t1_med = float(parts[1])
                t1_p95 = float(parts[2])
                t10_med = float(parts[3])
                t10_p95 = float(parts[4])
                t60_med = float(parts[5])
                t60_p95 = float(parts[6])
                rows.append({
                    "comp": comp,
                    "t1_med": t1_med, "t1_p95": t1_p95,
                    "t10_med": t10_med, "t10_p95": t10_p95,
                    "t60_med": t60_med, "t60_p95": t60_p95
                })
            except ValueError:
                pass
                
    # Also extract absolute max error from the summary table
    # Regex find for this scenario in the summary
    summary_matches = re.findall(rf'\|\s+{re.escape(scn_name)}\s+\|\s+(1|10|60)\s+\([^\)]+\)\s+\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|', content)
    
    w_stats = {}
    for sm in summary_matches:
        w_ticks = int(sm[0])
        car_pos = float(sm[1].strip())
        car_vel = float(sm[2].strip())
        car_quat = float(sm[3].strip())
        ball_pos = float(sm[4].strip())
        ball_vel = float(sm[5].strip())
        status = sm[7].strip()
        w_stats[w_ticks] = {
            "car_pos": car_pos, "car_vel": car_vel, "car_quat": car_quat,
            "ball_pos": ball_pos, "ball_vel": ball_vel, "status": status
        }
        
    # Calculate max relative error at t1, t10, t60
    max_rel_t1 = max([r["t1_p95"] for r in rows]) if rows else 0.0
    max_rel_t10 = max([r["t10_p95"] for r in rows]) if rows else 0.0
    max_rel_t60 = max([r["t60_p95"] for r in rows]) if rows else 0.0
    
    # Calculate similarity percentage:
    sim_pct_t1 = max(0.0, (1.0 - max_rel_t1) * 100.0)
    sim_pct_t10 = max(0.0, (1.0 - max_rel_t10) * 100.0)
    sim_pct_t60 = max(0.0, (1.0 - max_rel_t60) * 100.0)
    
    # Also average similarity across components
    avg_sim_t1 = sum([max(0.0, (1.0 - r["t1_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    avg_sim_t10 = sum([max(0.0, (1.0 - r["t10_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    avg_sim_t60 = sum([max(0.0, (1.0 - r["t60_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    
    results.append({
        "name": scn_name,
        "max_rel_t1": max_rel_t1, "max_rel_t10": max_rel_t10, "max_rel_t60": max_rel_t60,
        "sim_pct_t1": sim_pct_t1, "sim_pct_t10": sim_pct_t10, "sim_pct_t60": sim_pct_t60,
        "avg_sim_t1": avg_sim_t1, "avg_sim_t10": avg_sim_t10, "avg_sim_t60": avg_sim_t60,
        "w_stats": w_stats,
        "rows": rows
    })

print(f"Processed {len(results)} scenario baseline results.")
for r in results[:15]:
    w60 = r["w_stats"].get(60, {})
    print(f"{r['name']:30s} | T1: {r['sim_pct_t1']:8.4f}% | T10: {r['sim_pct_t10']:8.4f}% | T60: {r['sim_pct_t60']:8.4f}% (Avg: {r['avg_sim_t60']:8.4f}%) | CarPosErr={w60.get('car_pos', 0.0):.2e} UU")
