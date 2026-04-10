import requests
import json
import time
import os

BASE_URL = "https://bugzilla.redhat.com/rest/bug"

LIMIT = 100
MAX_BUGS = 200000   # 🔥 HARD LIMIT
SLEEP_TIME = 0.5
FILE_NAME = "bugzilla_2020_2026_200k.json"

FIELDS = "id,summary,severity,status,creation_time,product,component"

# Load existing data if resuming
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r", encoding="utf-8") as f:
        all_bugs = json.load(f)
else:
    all_bugs = []

total_collected = len(all_bugs)
print(f"Starting... Already have {total_collected} bugs")

for year in range(2026, 2019, -1):   # 🔥 latest first (important)
    print(f"\n===== YEAR {year} =====")

    offset = 0

    while True:
        if total_collected >= MAX_BUGS:
            print("\n✅ Reached 200,000 bugs limit")
            break

        params = {
            "limit": LIMIT,
            "offset": offset,
            "creation_time": f"{year}-01-01",
            "last_change_time": f"{year}-12-31",
            "include_fields": FIELDS
        }

        print(f"Fetching Year {year}, Offset {offset}")

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
            continue

        bugs = data.get("bugs", [])

        if not bugs:
            print(f"Finished year {year}")
            break

        # 🔥 Trim if exceeding limit
        remaining = MAX_BUGS - total_collected
        if len(bugs) > remaining:
            bugs = bugs[:remaining]

        all_bugs.extend(bugs)
        total_collected += len(bugs)

        # Save progress
        with open(FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(all_bugs, f)

        print(f"Saved {total_collected} bugs total")

        offset += LIMIT
        time.sleep(SLEEP_TIME)

    if total_collected >= MAX_BUGS:
        break

print("\n🎉 DONE!")
print(f"Total bugs collected: {total_collected}")