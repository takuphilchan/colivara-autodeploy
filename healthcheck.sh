#!/bin/bash
# Comprehensive health check for all services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
HOST_IP="${HOST_IP:-127.0.0.1}"

echo "🏥 Document Q&A - Health Check"
echo "================================"
echo ""

check_service() {
    local name=$1
    local url=$2
    
    if curl -sf "${url}" >/dev/null 2>&1; then
        echo "  ✓ ${name} is healthy"
        return 0
    else
        echo "  ✗ ${name} is not responding"
        return 1
    fi
}

all_ok=true

echo "Backend Services:"
check_service "Ollama" "http://${HOST_IP}:11434/api/tags" || all_ok=false
check_service "Embedding Service" "http://${HOST_IP}:8000/health" || all_ok=false
check_service "ColiVara API" "http://${HOST_IP}:8001/v1/docs" || all_ok=false
check_service "MinIO" "http://${HOST_IP}:9000/minio/health/live" || all_ok=false

echo ""
echo "Application Services:"
check_service "Flask App" "http://${HOST_IP}:5000" || all_ok=false
check_service "Query API" "http://${HOST_IP}:5001/health" || all_ok=false

echo ""
echo "Systemd Services:"
for service in colivara-embedding colivara-api colivara-app colivara-query-api; do
    if systemctl is-active --quiet "${service}.service" 2>/dev/null; then
        echo "  ✓ ${service} is running"
    else
        echo "  ✗ ${service} is not running"
        all_ok=false
    fi
done

echo ""
echo "Ollama Models:"
if command -v ollama >/dev/null 2>&1; then
    if ollama list 2>/dev/null | grep -q "qwen2.5vl\|llama3.2-vision"; then
        echo "  ✓ Vision models available"
        ollama list | grep -E "qwen2.5vl|llama3.2-vision" | sed 's/^/    /'
    else
        echo "  ⚠ No vision models found (users can use API keys instead)"
    fi
else
    echo "  ⚠ Ollama not installed (users can use API keys instead)"
fi

echo ""
if [[ "${all_ok}" == "true" ]]; then
    echo "✓ All critical services are healthy"
    exit 0
else
    echo "⚠ Some services need attention"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check logs: tail -f ${SCRIPT_DIR}/logs/*.log"
    echo "  - Restart services: sudo systemctl restart 'colivara-*'"
    echo "  - View service status: systemctl status 'colivara-*'"
    exit 1
fi
