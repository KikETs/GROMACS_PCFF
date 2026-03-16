#!/usr/bin/env python3
import os
import json

def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi*yi for xi, yi in zip(x, y))
    sum_xx = sum(xi*xi for xi in x)
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0: return 0, 0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def main():
    output_path = "/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_charged_recovery/dense_salt_polymer"
    xvg_file = os.path.join(output_path, "energy_raw.xvg")
    
    data = []
    with open(xvg_file, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')): continue
            cols = line.split()
            if len(cols) >= 3:
                data.append([float(x) for x in cols])

    # 1ns blocks
    blocks = []
    for i in range(3): # We have 3ns
        start_t = i * 1000
        end_t = (i + 1) * 1000
        b_data = [row for row in data if start_t <= row[0] < end_t]
        if not b_data: continue
        
        b_time = [row[0] for row in b_data]
        b_pot = [row[1] for row in b_data]
        b_temp = [row[2] for row in b_data]
        
        slope_pot, _ = linear_regression(b_time, b_pot)
        
        blocks.append({
            "index": i,
            "window_ns": [start_t/1000, end_t/1000],
            "potential_energy_mean": sum(b_pot)/len(b_pot),
            "potential_energy_drift_per_100ps": slope_pot * 100,
            "temperature_mean": sum(b_temp)/len(b_temp)
        })

    # High temperature check
    last_block = blocks[-1]
    if last_block["temperature_mean"] > 400:
        overall_status = "unresolved / unstable (Thermal Runaway)"
    else:
        overall_status = "partial / extend equilibration"

    summary = {
        "milestone": "TP1.2",
        "system_id": "dense_salt_polymer",
        "equilibration_duration_ns": 3.017,
        "recovery_status": {
            "overall": overall_status,
            "note": "Rerun failed at 3ns due to thermal runaway (T > 500K). System is NOT ready for production."
        },
        "block_analysis": {
            "units": {"potential_energy": "kJ/mol", "temperature": "K"},
            "blocks": blocks
        }
    }
    
    with open(os.path.join(output_path, "recovery_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also update drift_analysis.csv
    with open(os.path.join(output_path, "drift_analysis.csv"), 'w') as f:
        f.write("block_idx,start_ps,end_ps,pot_eng_mean,pot_eng_drift_per_100ps,temp_mean\n")
        for b in blocks:
            f.write(f"{b['index']},{b['window_ns'][0]*1000},{b['window_ns'][1]*1000},{b['potential_energy_mean']},{b['potential_energy_drift_per_100ps']},{b['temperature_mean']}\n")

    print(f"Analysis complete. Status: {overall_status}")

if __name__ == "__main__":
    main()
