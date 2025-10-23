"""Quick diagnostic to see what features the model expects vs what the app generates"""

import json
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("model/Final_Transform")

# Load expected features
with open(MODEL_PATH / "artifacts" / "feature_columns.json", 'r') as f:
    expected_features = json.load(f)['feature_columns']

print(f"Model expects {len(expected_features)} features:\n")
print("\n".join(f"  {i+1}. {feat}" for i, feat in enumerate(expected_features)))

print("\n" + "="*70)
print("FEATURE GROUPS:")
print("="*70)

# Group by prefix
groups = {}
for feat in expected_features:
    if '__' in feat:
        prefix = feat.split('__')[0]
        groups.setdefault(prefix, []).append(feat)
    else:
        groups.setdefault('other', []).append(feat)

for group, feats in sorted(groups.items()):
    print(f"\n{group}: {len(feats)} features")
    if len(feats) <= 10:
        for f in feats:
            print(f"  - {f}")

print("\n" + "="*70)
print("Check if these features exist in your forecast data!")
