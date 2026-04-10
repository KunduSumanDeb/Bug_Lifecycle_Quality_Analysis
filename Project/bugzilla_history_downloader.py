import requests
import json
import csv
from datetime import datetime, timezone

# Bugzilla API endpoint (Mozilla as example)
url = "https://bugzilla.mozilla.org/rest/bug"

# Parameters for fetching bugs
params = {
    "product": "Firefox",
    "include_fields": "id,status,priority,assigned_to,creator,creation_time,last_change_time",
    "limit": 1000,
    "status": "RESOLVED"   # ✅ Only resolved bugs
}

# Comment filter window (make them timezone-aware UTC)
COMMENT_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
COMMENT_END   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

all_bugs_with_comments = []
flat_comments = []  # for CSV export
offset = 0

while True:
    print(f"Fetching RESOLVED bug batch with offset {offset}...")
    params["offset"] = offset
    res = requests.get(url, params=params)

    try:
        data = res.json()
    except Exception as e:
        print("Error decoding JSON:", e)
        break

    bugs = data.get("bugs", [])
    if not bugs:
        print("No more bugs found.")
        break

    for bug in bugs:
        bug_id = bug["id"]

        # Fetch comments with new_since filter
        comments_url = f"https://bugzilla.mozilla.org/rest/bug/{bug_id}/comment"
        try:
            c_res = requests.get(comments_url, params={"new_since": "2020-01-01T00:00:00Z"})
            c_data = c_res.json()
            comments = c_data.get("bugs", {}).get(str(bug_id), {}).get("comments", [])

            # Filter comments by 2020–2025
            filtered_comments = []
            for c in comments:
                try:
                    # Parse comment time as timezone-aware UTC
                    c_time = datetime.fromisoformat(c.get("creation_time").replace("Z", "+00:00"))
                    if COMMENT_START <= c_time <= COMMENT_END:
                        creator_email = c.get("creator")
                        developer_name = creator_email.split("@")[0] if creator_email else None
                        comment_obj = {
                            "comment_id": c.get("id"),
                            "comment_time": c.get("creation_time"),
                            "commenter_email": creator_email,
                            "developer_name": developer_name,
                            "commenter_id": c.get("creator_id"),
                            "bug_id": bug_id,
                            "bug_status": bug.get("status"),
                            "bug_priority": bug.get("priority"),
                            "bug_creator": bug.get("creator"),
                            "bug_creation_time": bug.get("creation_time"),
                        }
                        filtered_comments.append(comment_obj)
                        flat_comments.append(comment_obj)  # ✅ for CSV
                except Exception as e:
                    print(f"Error parsing time for comment {c.get('id')}: {e}")

            if filtered_comments:
                bug["comments"] = filtered_comments
                all_bugs_with_comments.append(bug)

        except Exception as e:
            print(f"Error fetching comments for bug {bug_id}: {e}")
            bug["comments"] = []

    offset += 1000

# Save JSON
with open("resolved_bugs_comments_2020_2025.json", "w", encoding="utf-8") as f:
    json.dump(all_bugs_with_comments, f, indent=4)

print(f"Saved {len(all_bugs_with_comments)} resolved bugs to resolved_bugs_comments_2020_2025.json")

# Save CSV (flattened comments)
if flat_comments:
    keys = flat_comments[0].keys()
    with open("resolved_bugs_comments_2020_2025.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_comments)

    print(f"Saved {len(flat_comments)} comments to resolved_bugs_comments_2020_2025.csv")
else:
    print("No comments found for RESOLVED bugs in 2020–2025.")