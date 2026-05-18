#!/bin/bash
# Daily Jevans Directory Scanner
# Run via cron: 0 2 * * * /project/jevans/robbie/chorus/scripts/daily_jevans_scan.sh

set -e

# Configuration
CHORUS_DIR="/project/jevans/robbie/chorus"
LOG_DIR="$CHORUS_DIR/logs"

# Use the Midway venv with sentence-transformers for RAG indexing
# Falls back to base Python 3.11 if venv not available
VENV_PYTHON="$CHORUS_DIR/.venv_midway/bin/python3"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="/software/python-3.11.9-el8-x86_64/bin/python3"
fi

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Log file with date
LOG_FILE="$LOG_DIR/scan_$(date +%Y-%m-%d).log"

# Start logging
echo "========================================" >> "$LOG_FILE"
echo "Jevans Daily Scan - $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Change to chorus directory
cd "$CHORUS_DIR"

# Run the directory scanner
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from ingest.daily_scan import run_daily_scan
result = run_daily_scan(verbose=True)
" >> "$LOG_FILE" 2>&1

# Export SLURM job data
echo "" >> "$LOG_FILE"
echo "Exporting SLURM jobs..." >> "$LOG_FILE"
if [ -x "$HOME/bin/export_jobs_to_globus.sh" ]; then
    "$HOME/bin/export_jobs_to_globus.sh" >> "$LOG_FILE" 2>&1 || echo "Job export failed (non-fatal)" >> "$LOG_FILE"
else
    echo "Job export script not found, skipping" >> "$LOG_FILE"
fi

# Push results to GitHub
echo "" >> "$LOG_FILE"
echo "Pushing to GitHub..." >> "$LOG_FILE"
{
    git add Data/snapshots/ Data/jevans_resources.json Data/change_log.json Data/rag_indexes/jevans/ 2>/dev/null || true
    git add Data/midway_inventory_*.md Data/midway_jobs/ 2>/dev/null || true

    # Only commit if there are changes
    if git diff --cached --quiet; then
        echo "No changes to commit"
    else
        git commit -m "Daily scan $(date +%Y-%m-%d)"
        git push origin main
        echo "Pushed to GitHub"
    fi
} >> "$LOG_FILE" 2>&1

# Log completion
echo "" >> "$LOG_FILE"
echo "Scan completed at $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null || true

exit 0
