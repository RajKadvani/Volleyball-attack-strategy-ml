"""
Generate realistic benchmark volleyball attack rally dataset matching VREN schema.
Generates 1,632 rallies across 15 NCAA / international matches.
Uses only Python standard library.
"""

import os
import csv
import random
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def generate_dataset(output_path, num_rallies=1632, num_matches=15, seed=42):
    random.seed(seed)
    
    headers = [
        "match_id",
        "rally_id",
        "round",
        "team",
        "serve_type",
        "pass_rating",
        "set_type",
        "set_zone",
        "attack_type",
        "attack_zone",
        "num_blockers",
        "block_touch",
        "attack_success"
    ]
    
    match_ids = [f"M{i:02d}" for i in range(1, num_matches + 1)]
    serve_types = ["Float_Serve", "Jump_Serve", "Topspin"]
    teams = ["Team_A", "Team_B"]
    
    rows = []
    
    for r_idx in range(1, num_rallies + 1):
        # Assign match (roughly equal rallies per match)
        match_id = match_ids[(r_idx - 1) % num_matches]
        round_num = random.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.12, 0.05])[0]
        team = random.choice(teams)
        serve_type = random.choices(serve_types, weights=[0.50, 0.40, 0.10])[0]
        
        # Pass rating: 1 (Poor), 2 (Medium), 3 (Perfect)
        # Float serve often leads to medium/perfect passes; Jump serve creates more poor passes
        if serve_type == "Jump_Serve":
            pass_rating = random.choices([1, 2, 3], weights=[0.32, 0.40, 0.28])[0]
        else:
            pass_rating = random.choices([1, 2, 3], weights=[0.18, 0.42, 0.40])[0]
            
        # Setter options depend heavily on pass rating
        if pass_rating == 3: # In-system
            set_type = random.choices(
                ["Quick", "Outside", "Slide", "Pipe", "Opposite"],
                weights=[0.32, 0.30, 0.18, 0.12, 0.08]
            )[0]
        elif pass_rating == 2: # Medium system
            set_type = random.choices(
                ["Outside", "Opposite", "Pipe", "Quick", "High_Ball"],
                weights=[0.45, 0.25, 0.15, 0.08, 0.07]
            )[0]
        else: # Out-of-system (poor pass)
            set_type = random.choices(
                ["High_Ball", "Outside", "Pipe"],
                weights=[0.68, 0.24, 0.08]
            )[0]
            
        # Set zone corresponds to set type
        if set_type in ["Outside", "High_Ball"]:
            set_zone = "Zone_4"
            attack_zone = "Zone_4"
        elif set_type == "Quick":
            set_zone = "Zone_3"
            attack_zone = "Zone_3"
        elif set_type == "Slide":
            set_zone = "Zone_2"
            attack_zone = "Zone_2"
        elif set_type == "Opposite":
            set_zone = "Zone_2"
            attack_zone = "Zone_2"
        elif set_type == "Pipe":
            set_zone = "Zone_3"
            attack_zone = "Zone_6" # back row middle
        else:
            set_zone = "Zone_4"
            attack_zone = "Zone_4"
            
        # Number of blockers defending
        # In-system quicks often isolate 0 or 1 blocker. High balls allow 2 or 3 blockers to close.
        if set_type == "Quick":
            num_blockers = random.choices([0, 1, 2], weights=[0.25, 0.55, 0.20])[0]
        elif set_type == "Slide":
            num_blockers = random.choices([1, 2], weights=[0.60, 0.40])[0]
        elif set_type == "High_Ball":
            num_blockers = random.choices([2, 3], weights=[0.65, 0.35])[0]
        else:
            num_blockers = random.choices([1, 2, 3], weights=[0.20, 0.65, 0.15])[0]
            
        # Attack execution type
        if num_blockers >= 3:
            attack_type = random.choices(
                ["Hard_Spike", "Tool_Block", "Tip", "Roll_Shot", "Off_Speed"],
                weights=[0.35, 0.25, 0.20, 0.12, 0.08]
            )[0]
        elif num_blockers == 0:
            attack_type = random.choices(
                ["Hard_Spike", "Tip"],
                weights=[0.92, 0.08]
            )[0]
        else:
            attack_type = random.choices(
                ["Hard_Spike", "Tip", "Roll_Shot", "Tool_Block", "Off_Speed"],
                weights=[0.60, 0.16, 0.10, 0.08, 0.06]
            )[0]
            
        # Block touch probability
        if num_blockers == 0:
            block_touch = 0
        elif num_blockers == 1:
            block_touch = random.choices([0, 1], weights=[0.65, 0.35])[0]
        elif num_blockers == 2:
            block_touch = random.choices([0, 1], weights=[0.45, 0.55])[0]
        else:
            block_touch = random.choices([0, 1], weights=[0.30, 0.70])[0]
            
        # Latent outcome modeling (attack win probability)
        # Physical volleyball principles:
        # - High pass rating gives tactical initiative (+0.55 per level)
        # - Each additional blocker drastically drops scoring probability (-0.65 per blocker)
        # - Fast tempo attacks (Quick/Slide) against 0/1 blocker have high kill efficiency
        # - High balls into 3 blockers have low efficiency
        # - Tooling the block against multi-blockers has positive edge
        logit = 0.0
        logit += 0.55 * (pass_rating - 2.0)
        logit -= 0.75 * (num_blockers - 1.5)
        
        if set_type in ["Quick", "Slide"] and num_blockers <= 1:
            logit += 0.85
        elif set_type == "High_Ball":
            logit -= 0.50
            
        if attack_type == "Hard_Spike" and num_blockers <= 1:
            logit += 0.65
        elif attack_type == "Hard_Spike" and num_blockers >= 3:
            logit -= 0.55
        elif attack_type == "Tool_Block" and num_blockers >= 2:
            logit += 0.40
        elif attack_type == "Tip" and num_blockers >= 2:
            logit += 0.25 # tipping over big block is effective
            
        if block_touch == 1 and attack_type != "Tool_Block":
            logit -= 0.40
            
        # Stochastic component
        logit += random.gauss(0, 0.45)
        prob = sigmoid(logit)
        attack_success = 1 if random.random() < prob else 0
        
        rows.append([
            match_id,
            r_idx,
            round_num,
            team,
            serve_type,
            pass_rating,
            set_type,
            set_zone,
            attack_type,
            attack_zone,
            num_blockers,
            block_touch,
            attack_success
        ])
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} rally possessions in {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_csv = os.path.join(base_dir, "data", "sample_vren_dataset.csv")
    generate_dataset(target_csv)
