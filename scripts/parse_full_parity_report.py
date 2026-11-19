import re
import json

path = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\all_parity_report.md"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

table_lines = []
in_table = False
for line in lines:
    if line.strip().startswith("| Scenario | Window"):
        in_table = True
        continue
    if in_table:
        if line.strip().startswith("| :---"):
            continue
        if not line.strip().startswith("|"):
            break
        table_lines.append(line.strip())

parsed = []
for tl in table_lines:
    parts = [p.strip() for p in tl.split("|")[1:-1]]
    if len(parts) >= 9:
        scenario = parts[0]
        window = parts[1]
        car_pos = float(parts[2])
        car_vel = float(parts[3])
        car_quat = float(parts[4])
        ball_pos = float(parts[5])
        ball_vel = float(parts[6])
        status = parts[8]
        parsed.append({
            "scenario": scenario,
            "window": window,
            "car_pos": car_pos,
            "car_vel": car_vel,
            "car_quat": car_quat,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "status": status
        })

print(f"Total parsed scenario window entries: {len(parsed)}")

# Also read snapshot details for relative errors
# Let's inspect unique scenarios
unique_scenarios = sorted(list(set(p["scenario"] for p in parsed)))
print(f"Total unique scenarios: {len(unique_scenarios)}")
for s in unique_scenarios:
    entries = [p for p in parsed if p["scenario"] == s]
    print(f"Scenario: {s}")
    for e in entries:
        print(f"  {e['window']:15s}: CarPos={e['car_pos']:9.3e} UU | CarVel={e['car_vel']:9.3e} UU/s | Quat={e['car_quat']:9.3e} | BallPos={e['ball_pos']:9.3e} | BallVel={e['ball_vel']:9.3e}")
