import re
import json

path = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\all_parity_report.md"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

sections = content.split("### Milestone 5 Phase 1 Component-Wise Parity Baseline (")

results = {}

for sec in sections[1:]:
    scn_name = sec.split(",")[0].strip()
    
    rel_part = sec.split("#### Relative Error")
    if len(rel_part) < 2:
        continue
    rel_table_text = rel_part[1].split("#### Snapshot Detail")[0]
    
    rows = []
    for line in rel_table_text.strip().splitlines():
        if not line.strip().startswith("|") or ":---" in line or "Component" in line:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 7:
            comp = parts[0]
            try:
                t1_p95 = float(parts[2])
                t10_p95 = float(parts[4])
                t60_p95 = float(parts[6])
                rows.append({
                    "comp": comp,
                    "t1_p95": t1_p95,
                    "t10_p95": t10_p95,
                    "t60_p95": t60_p95
                })
            except ValueError:
                pass
                
    summary_matches = re.findall(rf'\|\s+{re.escape(scn_name)}\s+\|\s+(1|10|60)\s+\([^\)]+\)\s+\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|', content)
    
    w_stats = {}
    for sm in summary_matches:
        w_ticks = int(sm[0])
        w_stats[w_ticks] = {
            "car_pos": float(sm[1].strip()),
            "car_vel": float(sm[2].strip()),
            "car_quat": float(sm[3].strip()),
            "ball_pos": float(sm[4].strip()),
            "ball_vel": float(sm[5].strip()),
            "status": sm[7].strip()
        }
        
    max_rel_t1 = max([r["t1_p95"] for r in rows]) if rows else 0.0
    max_rel_t10 = max([r["t10_p95"] for r in rows]) if rows else 0.0
    max_rel_t60 = max([r["t60_p95"] for r in rows]) if rows else 0.0
    
    sim_pct_t1 = max(0.0, (1.0 - max_rel_t1) * 100.0)
    sim_pct_t10 = max(0.0, (1.0 - max_rel_t10) * 100.0)
    sim_pct_t60 = max(0.0, (1.0 - max_rel_t60) * 100.0)
    
    avg_sim_t1 = sum([max(0.0, (1.0 - r["t1_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    avg_sim_t10 = sum([max(0.0, (1.0 - r["t10_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    avg_sim_t60 = sum([max(0.0, (1.0 - r["t60_p95"]) * 100.0) for r in rows]) / len(rows) if rows else 100.0
    
    results[scn_name] = {
        "name": scn_name,
        "sim_t1": sim_pct_t1, "sim_t10": sim_pct_t10, "sim_t60": sim_pct_t60,
        "avg_t1": avg_sim_t1, "avg_t10": avg_sim_t10, "avg_t60": avg_sim_t60,
        "w_stats": w_stats,
        "rows": rows
    }

categories = [
    ("1. Dinâmica Básica, Repouso e Queda Livre", [
        "idle", "freefall"
    ]),
    ("2. Condução em Solo, Tração, Boost e Freio de Mão", [
        "throttle", "boost", "ablation_1_discrete_throttle_steer",
        "ablation_2_analog_throttle_steer", "ablation_3_plus_boost",
        "ablation_4_plus_handbrake"
    ]),
    ("3. Mecânica de Saltos, Duplo Salto & Flips/Dodges", [
        "jump_flip", "ablation_5_plus_single_jump",
        "ablation_6_plus_double_jump", "ablation_7_plus_dodge_flip",
        "ablation_5_flips"
    ]),
    ("4. Controle Aéreo, Downforce & Pousos", [
        "ablation_8_plus_air_control", "ablation_9_plus_landings"
    ]),
    ("5. Geometria da Arena, Rampas Curvas & Teto", [
        "car_side_wall", "car_corner_ramp", "car_ceiling",
        "car_jump_wall_land", "car_floor_wall_transition"
    ]),
    ("6. Física da Bola & Quiques em Superfícies", [
        "ball_flight", "ball_floor_drop", "ball_floor_angled",
        "ball_side_wall", "ball_back_wall", "ball_ceiling",
        "ball_corner_ramp", "ball_goal_post", "ball_crossbar"
    ]),
    ("7. Colisão Carro-Bola (Toques, Chutes & Defesas)", [
        "car_ball_hit", "kickoff_goalie", "wheels_on_ball"
    ]),
    ("8. Boost Pads (Coleta & Cooldowns)", [
        "boost_pad_pickup"
    ]),
    ("9. Presets Oficiais de Hitbox (6 Hitboxes)", [
        "hitbox_dominus", "hitbox_plank", "hitbox_breakout",
        "hitbox_hybrid", "hitbox_merc", "hitbox_psyclops"
    ]),
    ("10. Colisões Carro-Carro, Bumps & Demolições", [
        "car_car_front", "car_car_side", "car_car_rear",
        "car_car_air", "car_on_car", "wheels_on_car",
        "car_bump_supersonic", "car_demo", "car_demo_respawn"
    ]),
    ("11. Configurações de Mutator (Física Customizada)", [
        "config_low_gravity", "config_heavy_ball"
    ]),
    ("12. Partidas Multi-Carro, Kickoff Canônico & Caos", [
        "kickoff_multicar", "multicar_2v2_goal_demo",
        "ablation_10_random_full", "random"
    ])
]

seen = set()
for _, scns in categories:
    seen.update(scns)

diff = set(results.keys()) - seen
assert len(diff) == 0, f"Missing scenarios: {diff}"
assert len(seen) == 54, f"Expected 54, got {len(seen)}"

out_data = {
    "categories": [],
    "total_scenarios": len(results),
    "summary": {}
}

all_t1 = []
all_t10 = []
all_t60 = []
all_avg_t60 = []

for cat_title, scn_list in categories:
    cat_items = []
    for s in scn_list:
        r = results[s]
        w60 = r["w_stats"].get(60, {})
        pos_err = w60.get("car_pos", 0.0)
        vel_err = w60.get("car_vel", 0.0)
        quat_err = w60.get("car_quat", 0.0)
        bpos_err = w60.get("ball_pos", 0.0)
        bvel_err = w60.get("ball_vel", 0.0)
        cat_items.append({
            "scenario": s,
            "sim_t1": r["sim_t1"],
            "sim_t10": r["sim_t10"],
            "sim_t60": r["sim_t60"],
            "avg_t60": r["avg_t60"],
            "car_pos_err": pos_err,
            "car_vel_err": vel_err,
            "car_quat_err": quat_err,
            "ball_pos_err": bpos_err,
            "ball_vel_err": bvel_err,
            "status": w60.get("status", "PASS")
        })
        all_t1.append(r["sim_t1"])
        all_t10.append(r["sim_t10"])
        all_t60.append(r["sim_t60"])
        all_avg_t60.append(r["avg_t60"])
    out_data["categories"].append({
        "title": cat_title,
        "items": cat_items
    })

out_data["summary"] = {
    "mean_sim_t1": sum(all_t1) / len(all_t1),
    "mean_sim_t10": sum(all_t10) / len(all_t10),
    "mean_sim_t60_worst": sum(all_t60) / len(all_t60),
    "mean_sim_t60_avg_components": sum(all_avg_t60) / len(all_avg_t60),
}

with open(r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\scripts\parity_processed_data.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

print("Exported parity_processed_data.json successfully with all 54 scenarios!")
