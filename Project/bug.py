import requests
import pandas as pd

url = "https://raw.githubusercontent.com/ansymo/msr2013-bug_dataset/refs/heads/master/data/v02/mozilla/bug_status.json"

# Fetch JSON
response = requests.get(url)
bug_data = response.json()

# Flatten data
rows = []

for bug_id, events in bug_data["bug_status"].items():
    for event in events:
        rows.append({
            "bug_id": int(bug_id),
            "timestamp": event["when"],
            "status": event["what"],
            "user_id": event["who"]
        })

df = pd.DataFrame(rows)


df["date"] = pd.to_datetime(df["timestamp"], unit="s")

df.to_csv("dataset.csv",index = False)
print(df.head())
print(df.head())
print(df.shape)