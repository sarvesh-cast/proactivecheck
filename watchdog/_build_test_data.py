import json

# Part 1: pods + nodes (from first MCP call)
part1_raw = '''{"result":"{\\"success\\": true, \\"description\\": \\"Extract trimmed sample pods and nodes for testing\\", \\"result\\": PLACEHOLDER}"}'''

# We'll build part1 and part2 from the actual MCP results
# Instead, let's parse the persisted output file if available, or reconstruct manually

# Actually, let's just build the combined JSON directly from the data we know
# This script is called from the workspace where we'll pipe in the data

import sys

if len(sys.argv) < 3:
    print("Usage: python _build_test_data.py <part1.json> <part2.json> <output.json>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    p1_outer = json.load(f)
p1 = json.loads(p1_outer["result"])["result"]

with open(sys.argv[2]) as f:
    p2_outer = json.load(f)
p2 = json.loads(p2_outer["result"])["result"]

combined = {
    "counts": {
        "total_pods": p1["total_pods"],
        "total_nodes": p1["total_nodes"],
        "total_deps": p2["total_deps"],
        "total_recs": p2["total_recs"],
        "total_metrics": p2["total_metrics"],
        "oom_found": p1["oom_count"],
        "agent_found": p1["agent_count"],
    },
    "sample_pods": p1["sample_pods"],
    "sample_nodes": p1["sample_nodes"],
    "sample_deps": p2["sample_deps"],
    "sample_recs": p2["sample_recs"],
    "sample_metrics": p2["sample_metrics"],
}

with open(sys.argv[3], "w") as f:
    json.dump(combined, f, indent=2)

print(f"Written {len(json.dumps(combined))} chars")
print(f"Keys: {list(combined.keys())}")
for k, v in combined.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} items")
    elif isinstance(v, dict):
        print(f"  {k}: {json.dumps(v)}")
