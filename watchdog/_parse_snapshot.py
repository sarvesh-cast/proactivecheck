#!/usr/bin/env python3
"""One-shot script to parse the MCP tool result and extract snapshot data."""
import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path) as f:
    raw = json.load(f)

# The outer JSON has schema {result: string} where result is itself JSON
inner = json.loads(raw["result"])

# inner has keys: success, description, result
# The actual data is in inner["result"]
data = inner["result"]

# Save the data JSON (has keys: counts, sample_pods, sample_nodes, etc.)
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Wrote {len(json.dumps(data))} chars to {output_path}")
print(f"Keys: {list(data.keys())}")
for k, v in data.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} items")
    elif isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    else:
        print(f"  {k}: {v}")
