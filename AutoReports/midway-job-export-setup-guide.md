# Guide: Daily Midway Job Log Export to Globus

**Context:** You're setting up automated daily export of SLURM job logs from Midway to a Globus endpoint, where they'll be synced to a MacBook running the CHORUS system. These logs will become high-confidence `computation` hyperedges in CHORUS's knowledge graph.

---

## Overview

Create a daily cron job on Midway that:
1. Queries yesterday's SLURM jobs for Knowledge Lab users
2. Formats the data as JSON
3. Drops it in the Globus transfer directory
4. Handles incremental updates and error cases

---

## Step 1: Understand the Current Globus Setup

**First, locate the existing Globus endpoint directory:**

```bash
# Find where Globus Personal Connect is configured
echo $GLOBUS_ENDPOINT_PATH
# or check common locations:
ls -la ~/globus_endpoint
ls -la ~/.globus
```

**Verify you can write to this location and that files placed here sync to the MacBook.**

---

## Step 2: Query SLURM Job Data

**Test the `sacct` query interactively first:**

```bash
# Get yesterday's jobs for Knowledge Lab accounts
# Adjust the account pattern to match actual KL account names
sacct --starttime yesterday --endtime today \
      --accounts=pi-jamesevans,pi-evanslab \
      --format=JobID,User,Account,JobName,Partition,Start,End,State,NNodes,AllocTRES \
      --parsable2 --noheader
```

**Key flags explained:**
- `--starttime yesterday --endtime today`: Gets jobs from previous day
- `--accounts`: Filter to Knowledge Lab accounts (adjust pattern as needed)
- `--format`: Specific fields to extract
- `--parsable2`: Pipe-delimited output
- `--noheader`: Skip header row for easier parsing
- `AllocTRES`: Shows allocated resources including GPUs

**Identify the correct account names:**

```bash
# List all accounts you have access to
sacctmgr show associations where user=$USER format=account --noheader
```

---

## Step 3: Create the Export Script

**Create `~/bin/export_jobs_to_globus.sh`:**

```bash
#!/bin/bash

# Configuration
GLOBUS_DIR="$HOME/globus_endpoint/chorus_data"  # Adjust to actual path
OUTPUT_DIR="$GLOBUS_DIR/midway_jobs"
KL_ACCOUNTS="pi-jamesevans,pi-evanslab"  # Adjust to actual account names
DATE=$(date -d "yesterday" +%Y-%m-%d)
OUTPUT_FILE="$OUTPUT_DIR/jobs_$DATE.json"
LOG_FILE="$HOME/logs/job_export.log"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $LOG_FILE)"

# Log start
echo "[$(date)] Starting job export for $DATE" >> "$LOG_FILE"

# Check if file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "[$(date)] File $OUTPUT_FILE already exists, skipping" >> "$LOG_FILE"
    exit 0
fi

# Query SLURM
SACCT_OUTPUT=$(sacct --starttime "$DATE" --endtime "$(date -d "$DATE + 1 day" +%Y-%m-%d)" \
    --accounts="$KL_ACCOUNTS" \
    --format=JobID,User,Account,JobName,Partition,Start,End,State,NNodes,AllocTRES \
    --parsable2 --noheader 2>&1)

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: sacct failed: $SACCT_OUTPUT" >> "$LOG_FILE"
    exit 1
fi

# Count jobs
JOB_COUNT=$(echo "$SACCT_OUTPUT" | wc -l)
echo "[$(date)] Found $JOB_COUNT job records" >> "$LOG_FILE"

# Convert to JSON
# This Python script handles the conversion
python3 << 'PYTHON_SCRIPT' > "$OUTPUT_FILE"
import sys
import json
from datetime import datetime
import os

date = os.environ['DATE']
sacct_output = """$SACCT_OUTPUT"""

jobs = []
for line in sacct_output.strip().split('\n'):
    if not line:
        continue
    
    fields = line.split('|')
    if len(fields) < 10:
        continue
    
    job_id, user, account, job_name, partition, start, end, state, nnodes, alloc_tres = fields
    
    # Parse GPU count from AllocTRES (format: billing=X,cpu=Y,gres/gpu=Z,...)
    gpus = 0
    if 'gres/gpu=' in alloc_tres:
        try:
            gpu_part = [x for x in alloc_tres.split(',') if 'gres/gpu=' in x][0]
            gpus = int(gpu_part.split('=')[1])
        except:
            pass
    
    # Skip sub-jobs (those with dots in JobID)
    if '.' in job_id:
        continue
    
    job = {
        "job_id": job_id,
        "user": user,
        "account": account,
        "job_name": job_name,
        "partition": partition,
        "start": start if start != "Unknown" else None,
        "end": end if end != "Unknown" else None,
        "state": state,
        "nodes": int(nnodes) if nnodes.isdigit() else 0,
        "gpus": gpus
    }
    jobs.append(job)

output = {
    "date": date,
    "export_timestamp": datetime.now().isoformat(),
    "midway_cluster": os.environ.get('HOSTNAME', 'unknown'),
    "job_count": len(jobs),
    "jobs": jobs
}

print(json.dumps(output, indent=2))
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo "[$(date)] Successfully exported to $OUTPUT_FILE" >> "$LOG_FILE"
else
    echo "[$(date)] ERROR: JSON conversion failed" >> "$LOG_FILE"
    exit 1
fi

# Verify JSON is valid
python3 -m json.tool "$OUTPUT_FILE" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Generated invalid JSON" >> "$LOG_FILE"
    exit 1
fi

echo "[$(date)] Export complete" >> "$LOG_FILE"
```

**Make it executable:**

```bash
chmod +x ~/bin/export_jobs_to_globus.sh
```

---

## Step 4: Test the Script

**Run manually first:**

```bash
~/bin/export_jobs_to_globus.sh
```

**Check the output:**

```bash
# View the generated JSON
ls -lh ~/globus_endpoint/chorus_data/midway_jobs/

# Inspect content
cat ~/globus_endpoint/chorus_data/midway_jobs/jobs_$(date -d "yesterday" +%Y-%m-%d).json | head -50

# Check logs
tail ~/logs/job_export.log
```

**Verify the JSON structure matches what CHORUS expects.**

---

## Step 5: Set Up Daily Cron Job

**Edit your crontab:**

```bash
crontab -e
```

**Add this line (runs at 1 AM daily):**

```cron
0 1 * * * /home/YOUR_USERNAME/bin/export_jobs_to_globus.sh
```

**Or run at a different time if Globus sync happens at specific intervals.**

**Verify cron is set:**

```bash
crontab -l
```

---

## Step 6: Account Name Discovery

**If you're unsure which accounts to include, find all Knowledge Lab accounts:**

```bash
# List all accounts
sacctmgr show accounts format=account,description --noheader

# Find accounts with "evans" or "knowledge" in the name
sacctmgr show accounts format=account,description --noheader | grep -i evans
sacctmgr show accounts format=account,description --noheader | grep -i knowledge
```

**Update the `KL_ACCOUNTS` variable in the script accordingly.**

---

## Step 7: Backfill Historical Data

**Once the daily export is working, optionally backfill:**

```bash
#!/bin/bash
# Backfill script - run once to get historical data

GLOBUS_DIR="$HOME/globus_endpoint/chorus_data"
OUTPUT_DIR="$GLOBUS_DIR/midway_jobs"
KL_ACCOUNTS="pi-jamesevans,pi-evanslab"

# Backfill from January 1, 2024 to yesterday
START_DATE="2024-01-01"
END_DATE=$(date -d "yesterday" +%Y-%m-%d)

current_date="$START_DATE"
while [ "$current_date" != "$END_DATE" ]; do
    DATE="$current_date" ~/bin/export_jobs_to_globus.sh
    current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
    echo "Processed $current_date"
    sleep 1  # Be nice to SLURM
done
```

**Warning: This could generate many files. Test with a small date range first (e.g., last week).**

---

## Output Schema

**Each JSON file will look like:**

```json
{
  "date": "2026-01-15",
  "export_timestamp": "2026-01-16T01:00:23.123456",
  "midway_cluster": "midway3-login1",
  "job_count": 47,
  "jobs": [
    {
      "job_id": "12345678",
      "user": "revans",
      "account": "pi-jamesevans",
      "job_name": "bert_census_analysis",
      "partition": "gpu2",
      "start": "2026-01-15T08:23:11",
      "end": "2026-01-15T14:52:33",
      "state": "COMPLETED",
      "nodes": 1,
      "gpus": 2
    }
  ]
}
```

---

## Troubleshooting

**No jobs appearing:**
- Check account names are correct
- Verify date range (SLURM's "yesterday" might need explicit date format)
- Check if you have permission to query those accounts

**Globus not syncing:**
- Verify Globus Personal Connect is running on Midway
- Check directory permissions
- Test by placing a dummy file in the directory

**Cron not running:**
- Check cron logs: `grep CRON /var/log/syslog` (if accessible)
- Verify script has correct shebang and executable permissions
- Test with absolute paths (cron has minimal environment)

**JSON parsing errors:**
- Check for special characters in job names
- Verify Python 3 is available in cron environment
- Add error handling for malformed sacct output

---

## Next Steps for MacBook/CHORUS Integration

Once files are landing on the MacBook:

1. **Watcher script** to detect new JSON files
2. **Parser** that converts JSON into CHORUS hyperedge format:
   - User → Person node
   - Job name → potential Project/Dataset hints
   - Timestamp → exact event time
   - Create `computation` hyperedge with confidence=1.0
3. **Entity resolution** to map usernames to Person nodes in the graph
4. **Optional enrichment** by matching job names against known projects/datasets

---

## Questions to Confirm

Before implementing, verify:

1. **Exact Globus endpoint path** on Midway
2. **Knowledge Lab account names** in SLURM
3. **Date range** for backfill (if any)
4. **Frequency** - is daily sufficient or do you want more/less frequent exports?
5. **Privacy considerations** - any users whose jobs should be excluded?

Let me know if you hit any issues with SLURM permissions or Globus paths!
