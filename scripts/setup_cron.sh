#!/bin/bash
# Setup cron job for daily jevans scanning
# Run this script once to add the cron entry

SCRIPT_PATH="/project/jevans/robbie/chorus/scripts/daily_jevans_scan.sh"
LOG_PATH="/project/jevans/robbie/chorus/logs/cron.log"

# Cron entry: Run at 2:00 AM every day
CRON_ENTRY="0 2 * * * $SCRIPT_PATH >> $LOG_PATH 2>&1"

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "daily_jevans_scan.sh"; then
    echo "Cron entry already exists:"
    crontab -l | grep "daily_jevans_scan.sh"
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Added cron entry:"
echo "  $CRON_ENTRY"
echo ""
echo "Current crontab:"
crontab -l
