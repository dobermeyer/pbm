# audit_objects.py
# Run from /workspaces/pbm:
#   python audit_objects.py

import json
from collections import Counter

with open("backend/output/metadata.json", "r") as f:
    metadata = json.load(f)

objects = metadata.get("objects", {})

total_frames = len(objects)
counts = {frame: len(items) for frame, items in objects.items()}

# Distribution of object counts per frame
distribution = Counter(counts.values())

print(f"\n=== Object Detection Audit ===")
print(f"Total frames with detections: {total_frames}")
print(f"\nFrames by object count:")
for n in sorted(distribution.keys()):
    bar = "█" * distribution[n]
    print(f"  {n} object(s): {distribution[n]:4d} frames  {bar}")

# Frames with multiple objects
multi = {f: v for f, v in counts.items() if v > 1}
print(f"\nFrames with 2+ objects: {len(multi)} ({len(multi)/total_frames*100:.1f}% of detected frames)")

# Top 10 busiest frames
print(f"\nTop 10 busiest frames:")
for frame, count in sorted(multi.items(), key=lambda x: -x[1])[:10]:
    labels = [obj["label"] for obj in objects[frame]]
    print(f"  {frame}: {count} objects — {labels}")

# Label frequency across all frames
label_counter = Counter()
for items in objects.values():
    for obj in items:
        label_counter[obj["label"]] += 1

print(f"\nLabel frequency (all detections):")
for label, count in label_counter.most_common():
    print(f"  {label:20s}: {count}")