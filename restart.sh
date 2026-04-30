#!/bin/bash
pkill -f "uvicorn app:app.*8091" 2>/dev/null
sleep 1
cd /home/admin/gtm-team
nohup uvicorn app:app --host 127.0.0.1 --port 8091 --workers 1 > /tmp/gtm-team.log 2>&1 &
sleep 2
curl -s http://localhost:8091/api/health
