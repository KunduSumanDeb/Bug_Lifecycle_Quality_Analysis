import pandas as pd

# Load JSON file
df = pd.read_json("bugzilla_2020_2026_200k.json")

# Count occurrences of each bug id
id_counts = df.groupby("id").size().reset_index(name="count")

# Show duplicates only
duplicates = id_counts[id_counts["count"] > 1]

print(duplicates)
print(f"Total duplicate IDs: {len(duplicates)}")