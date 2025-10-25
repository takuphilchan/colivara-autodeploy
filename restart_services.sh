#!/bin/bash
# Restart all ColiVara services

echo "🔄 Restarting all services..."

services=(
    "colivara-embedding"
    "colivara-api"
    "colivara-app"
    "colivara-query-api"
)

for service in "${services[@]}"; do
    echo "  Restarting ${service}..."
    sudo systemctl restart "${service}.service" 2>/dev/null || echo "  ⚠ ${service} not found"
done

echo ""
echo "✓ Services restarted"
echo ""
echo "Checking status..."
systemctl status 'colivara-*' --no-pager || true
