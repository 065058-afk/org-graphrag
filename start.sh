#!/bin/bash
cd ~/org-graphrag
source venv/bin/activate
docker ps > /dev/null 2>&1 || sudo systemctl start docker
docker compose up -d
systemctl is-active ollama > /dev/null || sudo systemctl start ollama
echo "Waiting for databases..."
sleep 20
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "neo4j|postgres|adminer"
echo "Infrastructure ready. Now run uvicorn and streamlit in two other terminals."
