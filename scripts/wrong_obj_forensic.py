#!/usr/bin/env python3
"""r7 抓错取证：每个 WRONG_OBJECT 集到底抓了什么。"""
import json
import os
from collections import Counter

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting/eval_r2"
for f in ["diag_B1_r7press", "diag_B2_r7press", "diag_A_r7press"]:
    p = os.path.join(D, f + ".json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    agg = {}
    for e in d.get("results", []):
        if e.get("category") != "WRONG_OBJECT":
            continue
        b = str(e.get("bddl", e.get("bddl_file", "?")))
        b = b.replace("pick_up_the_", "").replace("_steel_plate_and_place_it_in_the_", "->")[:40]
        agg.setdefault(b, Counter()).update(e.get("wrong_plates_grasped", ["?"]))
    for b, c in agg.items():
        print(f, "|", b, "抓了:", dict(c))
