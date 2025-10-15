#!/bin/bash
set -euo pipefail

# ColiVara Complete Setup Script
# Automated installation and configuration for ColiVara RAG system
# Repository: https://github.com/tjmlabs/ColiVara
# License: MIT

# ============================================================================
# CONFIGURATION
# ============================================================================

# Service Ports
readonly EMBEDDING_PORT="${EMBEDDING_PORT:-8000}"
readonly COLIVARA_API_PORT="${COLIVARA_API_PORT:-8001}"
readonly APP_PORT="${APP_PORT:-5000}"
readonly QUERY_API_PORT="${QUERY_API_PORT:-5001}"
readonly MINIO_PORT="${MINIO_PORT:-9001}"
readonly OLLAMA_PORT="${OLLAMA_PORT:-11434}"

# Storage Configuration
readonly MINIO_ROOT_USER="${MINIO_ROOT_USER:-miniokey}"
readonly MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-miniosecret}"
readonly BUCKET_NAME="${BUCKET_NAME:-colivara}"

# Model Configuration
readonly COLQWEN_MODEL="vidore/colqwen2-v1.0"
readonly OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5vl:32b}"

# PyTorch Configuration
readonly TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

# Python Version
readonly REQUIRED_PYTHON_VERSION="3.10"

# Directory Structure - Current directory setup
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COLIVARE_DIR="${SCRIPT_DIR}/ColiVarE"
readonly COLIVARA_DIR="${SCRIPT_DIR}/ColiVara"
readonly PROJECT_BACKEND_DIR="${SCRIPT_DIR}"  # app.py and api.py are in current dir

# State Files
readonly EMBEDDING_SETUP_DONE="${SCRIPT_DIR}/.embedding_setup_done"
readonly API_SETUP_DONE="${SCRIPT_DIR}/.api_setup_done"
readonly MODELS_VALIDATED="${SCRIPT_DIR}/.models_validated"
readonly OLLAMA_SETUP_DONE="${SCRIPT_DIR}/.ollama_setup_done"
readonly PYTHON_SETUP_DONE="${SCRIPT_DIR}/.python_setup_done"
readonly VENV_SETUP_DONE="${SCRIPT_DIR}/.venv_setup_done"
readonly MCP_SETUP_DONE="${SCRIPT_DIR}/.mcp_setup_done"

# Logging
readonly LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $*"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $*"
}

log_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $*"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
}

detect_host_ip() {
    local ip
    if command -v hostname >/dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    
    if [[ -z "${ip}" ]]; then
        ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
    fi
    
    if [[ -z "${ip}" ]]; then
        ip="127.0.0.1"
        log_warning "Could not detect host IP, using localhost"
    fi
    
    echo "${ip}"
}

readonly HOST_IP=$(detect_host_ip)

show_progress() {
    local pid=$1
    local message=$2
    local dots=""
    local elapsed=0
    
    echo -n "${message}"
    while kill -0 "${pid}" 2>/dev/null; do
        if ((elapsed % 5 == 0)); then
            if command -v nvidia-smi &>/dev/null; then
                local gpu_mem
                gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%.1fGB", $1/1024}')
                echo -ne "\r${message}${dots} [${elapsed}s] (GPU:${gpu_mem})"
            else
                echo -ne "\r${message}${dots} [${elapsed}s]"
            fi
            dots="${dots}."
            [[ ${#dots} -gt 3 ]] && dots=""
        fi
        sleep 1
        ((elapsed++))
    done
    echo -e "\r${message} completed in ${elapsed}s"
}

cleanup() {
    log_info "Shutting down services gracefully..."
    
    if [[ -n "${EMBED_PID:-}" ]]; then
        kill "${EMBED_PID}" 2>/dev/null || true
    fi
    
    if [[ -d "${COLIVARA_DIR}" ]]; then
        cd "${COLIVARA_DIR}" && docker-compose stop 2>/dev/null || true
    fi
    
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# PYTHON 3.10 INSTALLATION
# ============================================================================

install_python310() {
    if [[ -f "${PYTHON_SETUP_DONE}" ]]; then
        log_info "Python 3.10 already configured"
        return 0
    fi
    
    if command -v python3.10 >/dev/null 2>&1; then
        log_success "Python 3.10 already installed"
        touch "${PYTHON_SETUP_DONE}"
        return 0
    fi
    
    log_info "Python 3.10 not found, installing..."
    
    # Detect OS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        OS=$ID
    else
        log_error "Cannot detect OS"
        return 1
    fi
    
    case "$OS" in
        ubuntu|debian)
            log_info "Installing Python 3.10 on Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa || true
            sudo apt-get update
            sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
            ;;
        centos|rhel|fedora)
            log_info "Installing Python 3.10 on RHEL/CentOS/Fedora..."
            sudo dnf install -y python3.10 python3.10-devel || \
            sudo yum install -y python3.10 python3.10-devel
            ;;
        arch|manjaro)
            log_info "Installing Python 3.10 on Arch..."
            sudo pacman -S --noconfirm python
            ;;
        *)
            log_error "Unsupported OS: $OS"
            log_info "Please install Python 3.10 manually"
            return 1
            ;;
    esac
    
    if command -v python3.10 >/dev/null 2>&1; then
        log_success "Python 3.10 installed successfully"
        touch "${PYTHON_SETUP_DONE}"
        return 0
    else
        log_error "Python 3.10 installation failed"
        return 1
    fi
}

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

check_system_requirements() {
    log_info "Checking system requirements..."
    
    local missing_deps=()
    
    if ! command -v docker >/dev/null 2>&1; then
        missing_deps+=("docker")
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command -v git >/dev/null 2>&1; then
        missing_deps+=("git")
    fi
    
    if ! command -v curl >/dev/null 2>&1; then
        missing_deps+=("curl")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install them using your package manager"
        log_info "Example (Ubuntu/Debian): sudo apt-get install docker.io docker-compose git curl"
        exit 1
    fi
    
    # Check/install Python 3.10
    install_python310 || {
        log_error "Python 3.10 is required but could not be installed"
        exit 1
    }
    
    if ! docker ps >/dev/null 2>&1; then
        log_error "Docker daemon is not running or user lacks permissions"
        log_info "Try: sudo systemctl start docker"
        log_info "Or add user to docker group: sudo usermod -aG docker \$USER"
        exit 1
    fi
    
    log_success "All system requirements satisfied"
}

# ============================================================================
# PORT MANAGEMENT
# ============================================================================

free_ports() {
    log_info "Releasing required ports..."
    
    local services=(
        "colivara-api.service"
        "colivara-app.service"
        "colivara-embedding.service"
        "colivara-query-api.service"
    )
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "${service}" 2>/dev/null; then
            log_info "Stopping ${service}..."
            sudo systemctl stop "${service}" 2>/dev/null || true
        fi
    done
    
    if [[ -d "${COLIVARA_DIR}" ]]; then
        cd "${COLIVARA_DIR}" || true
        docker-compose down --remove-orphans 2>/dev/null || true
        cd "${SCRIPT_DIR}"
    fi
    
    if docker network inspect colivara_default &>/dev/null; then
        docker network rm colivara_default 2>/dev/null || true
    fi
    
    local ports=(
        "${EMBEDDING_PORT}"
        "${COLIVARA_API_PORT}"
        "${APP_PORT}"
        "${QUERY_API_PORT}"
        "${MINIO_PORT}"
    )
    
    for port in "${ports[@]}"; do
        if lsof -ti:"${port}" >/dev/null 2>&1; then
            log_info "Killing processes on port ${port}..."
            sudo lsof -ti:"${port}" | xargs sudo kill -9 2>/dev/null || true
        fi
        
        if command -v fuser >/dev/null 2>&1; then
            sudo fuser -k "${port}/tcp" 2>/dev/null || true
        fi
    done
    
    sleep 3
    
    local all_clear=true
    for port in "${ports[@]}"; do
        if lsof -ti:"${port}" >/dev/null 2>&1; then
            log_warning "Port ${port} still in use"
            all_clear=false
        fi
    done
    
    if [[ "${all_clear}" == "true" ]]; then
        log_success "All ports are now free"
    else
        log_warning "Some ports may still be in use"
    fi
}

# ============================================================================
# REPOSITORY CLONING
# ============================================================================

clone_repository() {
    local repo_url=$1
    local target_dir=$2
    
    if [[ -d "${target_dir}" ]]; then
        log_info "${target_dir} already exists, skipping clone"
        return 0
    fi
    
    log_info "Cloning ${repo_url}..."
    
    if git clone "${repo_url}" "${target_dir}" 2>/dev/null; then
        log_success "Repository cloned successfully"
        return 0
    fi
    
    log_error "Failed to clone repository"
    return 1
}

# ============================================================================
# MODEL VALIDATION
# ============================================================================

validate_colqwen_model() {
    local model_path="${COLIVARE_DIR}/models_hub/vidore/colqwen2-v1.0"
    
    if [[ -f "${MODELS_VALIDATED}" ]]; then
        log_info "Models already validated"
        return 0
    fi
    
    log_info "Validating ColQwen2 model..."
    
    if [[ ! -d "${model_path}" ]]; then
        log_info "Model not found, downloading from Hugging Face..."
        mkdir -p "$(dirname "${model_path}")"
        
        python3.10 -c "
from huggingface_hub import snapshot_download
import sys

try:
    print('Downloading model from Hugging Face...')
    snapshot_download(
        '${COLQWEN_MODEL}',
        local_dir='${model_path}',
        local_dir_use_symlinks=False
    )
    print('Model download completed')
except Exception as e:
    print(f'Download failed: {e}', file=sys.stderr)
    sys.exit(1)
" || return 1
    fi
    
    local required_files=("config.json" "model.safetensors")
    for file in "${required_files[@]}"; do
        if [[ ! -f "${model_path}/${file}" ]]; then
            log_error "Missing required model file: ${file}"
            return 1
        fi
    done
    
    log_info "Testing model loading..."
    python3.10 -c "
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    
    device_map = 'cuda' if torch.cuda.is_available() else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = ColQwen2.from_pretrained(
        '${model_path}',
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    processor = ColQwen2Processor.from_pretrained(
        '${model_path}',
        local_files_only=True
    )
    
    print('Model validation successful')
except Exception as e:
    print(f'Validation failed: {e}')
    raise
" || return 1
    
    touch "${MODELS_VALIDATED}"
    log_success "Model validation completed"
}

# ============================================================================
# OLLAMA SETUP
# ============================================================================

install_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        log_info "Ollama already installed"
        return 0
    fi
    
    log_info "Installing Ollama..."
    
    if curl -fsSL https://ollama.com/install.sh | sh; then
        log_success "Ollama installed successfully"
    else
        log_error "Failed to install Ollama"
        return 1
    fi
}

setup_ollama() {
    if [[ -f "${OLLAMA_SETUP_DONE}" ]]; then
        log_info "Ollama already configured"
        return 0
    fi
    
    install_ollama || return 1
    
    log_info "Starting Ollama service..."
    if ! pgrep -f "ollama serve" >/dev/null; then
        nohup ollama serve > "${LOG_DIR}/ollama.log" 2>&1 &
        sleep 5
    fi
    
    log_info "Pulling ${OLLAMA_MODEL} model (this may take a while)..."
    if ollama pull "${OLLAMA_MODEL}"; then
        log_success "Ollama model pulled successfully"
        touch "${OLLAMA_SETUP_DONE}"
    else
        log_error "Failed to pull Ollama model"
        return 1
    fi
}

# ============================================================================
# EMBEDDING SERVICE
# ============================================================================

create_handler_py() {
    local handler_path="${COLIVARE_DIR}/handler.py"
    
    cat > "${handler_path}" << 'EOF'
import base64
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster

app = FastAPI()

MODEL_PATH = os.environ.get("COLIVARA_MODEL_PATH")
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")

if torch.cuda.is_available():
    device_map = "cuda"
    torch_dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device_map = "mps"
    torch_dtype = torch.bfloat16
else:
    device_map = None
    torch_dtype = torch.float32

model = ColQwen2.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch_dtype,
    device_map=device_map,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

processor = ColQwen2Processor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

print(f"Model loaded on: {next(model.parameters()).device}")

def pool_embeddings(embeddings: torch.Tensor, pool_factor: int = 3) -> List[List[float]]:
    similarities = torch.mm(embeddings, embeddings.t())
    distances = 1 - similarities.cpu().numpy()
    del similarities
    torch.cuda.empty_cache()

    target_clusters = max(embeddings.shape[0] // pool_factor, 1)
    clusters = linkage(distances, method="ward")
    cluster_labels = fcluster(clusters, t=target_clusters, criterion="maxclust")

    pooled = []
    for cluster_id in range(1, target_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_mean = cluster_embeddings.mean(dim=0)
        pooled.append(cluster_mean.cpu().tolist())
        del cluster_embeddings, cluster_mean
        torch.cuda.empty_cache()

    return pooled

class EmbedRequest(BaseModel):
    input_data: List[str]
    task: str

def encode_image(input_data: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    images = []
    for image in input_data:
        try:
            img_data = base64.b64decode(image)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    with torch.no_grad():
        batch_images = processor.process_images(images).to(model.device)
        image_embeddings = model(**batch_images)

    results = []
    for idx, embedding in enumerate(image_embeddings):
        embedding = embedding.to(torch.float32)
        pooled = pool_embeddings(embedding)
        del embedding
        torch.cuda.empty_cache()
        results.append({"object": "embedding", "embedding": pooled, "index": idx})

    del batch_images, image_embeddings
    torch.cuda.empty_cache()
    total_tokens = len(results) * len(pooled) if pooled else 0
    return results, total_tokens

def encode_query(queries: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    batch_queries = processor.process_queries(queries).to(model.device)
    total_tokens = sum(len(ids) for ids in batch_queries["input_ids"])

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    results = []
    for idx, embedding in enumerate(query_embeddings):
        embedding = embedding.to(torch.float32).detach().cpu().numpy().tolist()
        results.append({"object": "embedding", "embedding": embedding, "index": idx})
        del embedding
        torch.cuda.empty_cache()
        
    del batch_queries, query_embeddings
    torch.cuda.empty_cache()
    return results, total_tokens

@app.post("/runsync/")
async def process(request: EmbedRequest):
    try:
        if request.task == "image":
            embeddings, total_tokens = encode_image(request.input_data)
        elif request.task == "query":
            embeddings, total_tokens = encode_query(request.input_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid task type")
        
        return {
            "object": "list",
            "data": embeddings,
            "model": "vidore/colqwen2-v1.0",
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": str(next(model.parameters()).device)
    }
EOF
    
    log_success "Handler created at ${handler_path}"
}

setup_embedding_service() {
    if [[ -f "${EMBEDDING_SETUP_DONE}" ]]; then
        log_info "Embedding service already configured"
        return 0
    fi
    
    log_info "Setting up embedding service..."
    
    clone_repository "https://github.com/tjmlabs/ColiVarE.git" "${COLIVARE_DIR}" || return 1
    
    cd "${COLIVARE_DIR}"
    
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment..."
        python3.10 -m venv .venv
    fi
    
    source .venv/bin/activate
    
    log_info "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet \
        torch torchvision \
        --index-url "${TORCH_INDEX_URL}"
    pip install --quiet \
        fastapi uvicorn[standard] python-multipart \
        scipy colpali-engine pillow transformers \
        huggingface_hub
    
    validate_colqwen_model || return 1
    
    create_handler_py
    
    export COLIVARA_MODEL_PATH="${COLIVARE_DIR}/models_hub/vidore/colqwen2-v1.0"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    
    log_info "Starting embedding service on port ${EMBEDDING_PORT}..."
    nohup .venv/bin/python -m uvicorn handler:app \
        --host 0.0.0.0 \
        --port "${EMBEDDING_PORT}" \
        > "${LOG_DIR}/embedding.log" 2>&1 &
    
    EMBED_PID=$!
    
    log_info "Waiting for service to become ready..."
    for i in {1..60}; do
        if curl -s "http://localhost:${EMBEDDING_PORT}/health" >/dev/null 2>&1; then
            log_success "Embedding service started (PID: ${EMBED_PID})"
            touch "${EMBEDDING_SETUP_DONE}"
            cd "${SCRIPT_DIR}"
            return 0
        fi
        sleep 2
    done
    
    log_error "Service startup timeout"
    tail -50 "${LOG_DIR}/embedding.log"
    return 1
}

# ============================================================================
# API AND STORAGE
# ============================================================================

setup_api_service() {
    if [[ -f "${API_SETUP_DONE}" ]]; then
        log_info "API service already configured"
        return 0
    fi
    
    log_info "Setting up API and storage services..."
    
    clone_repository "https://github.com/tjmlabs/ColiVara.git" "${COLIVARA_DIR}" || return 1
    
    cd "${COLIVARA_DIR}"
    
    log_info "Configuring docker-compose.yml..."
    if ! grep -q "extra_hosts:" docker-compose.yml; then
        sed -i "/web:/a \    extra_hosts:\n      - \"host.docker.internal:${HOST_IP}\"" docker-compose.yml
    else
        sed -i "s/host.docker.internal:[0-9.]\+/host.docker.internal:${HOST_IP}/g" docker-compose.yml
    fi
    
    log_info "Creating environment configuration..."
    cat > .env.dev << EOF
MINIO_ROOT_USER=${MINIO_ROOT_USER}
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
EMBEDDINGS_URL=http://${HOST_IP}:${EMBEDDING_PORT}/runsync/
EMBEDDINGS_URL_TOKEN=
MINIO_ENDPOINT=http://minio:9000
AWS_STORAGE_BUCKET_NAME=${BUCKET_NAME}
AWS_S3_ENDPOINT_URL=http://minio:9000
EOF
    
    log_info "Starting Docker containers..."
    docker-compose up -d
    
    log_info "Waiting for services to be ready..."
    sleep 10
    
    log_info "Configuring MinIO..."
    for i in {1..10}; do
        if docker-compose exec -T minio \
            mc alias set local http://minio:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null 2>&1; then
            break
        fi
        sleep 3
    done
    
    if ! docker-compose exec -T minio mc ls "local/${BUCKET_NAME}" >/dev/null 2>&1; then
        docker-compose exec -T minio mc mb "local/${BUCKET_NAME}" >/dev/null 2>&1
    fi
    
    docker-compose exec -T minio mc policy set download "local/${BUCKET_NAME}" >/dev/null 2>&1 || true
    
    log_info "Running database migrations..."
    for i in {1..5}; do
        if docker-compose exec -T web python manage.py migrate >/dev/null 2>&1; then
            break
        fi
        sleep 5
    done
    
    log_info "Creating admin user..."
    docker-compose exec -T web python manage.py shell << 'PYEOF' >/dev/null 2>&1
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
PYEOF
    
    touch "${API_SETUP_DONE}"
    cd "${SCRIPT_DIR}"
    log_success "API service configured"
}

# ============================================================================
# PROJECT VENV AND APP/API SETUP
# ============================================================================

setup_project_venv() {
    if [[ -f "${VENV_SETUP_DONE}" ]]; then
        log_info "Project virtual environment already configured"
        return 0
    fi
    
    log_info "Setting up project virtual environment..."
    
    cd "${PROJECT_BACKEND_DIR}"
    
    if [[ ! -d ".venv" ]]; then
        log_info "Creating project virtual environment..."
        python3.10 -m venv .venv
    fi
    
    source .venv/bin/activate
    
    log_info "Installing project dependencies..."
    pip install --quiet --upgrade pip
    
    # Install common dependencies for app.py and api.py
    pip install --quiet \
        flask flask-cors \
        fastapi uvicorn[standard] \
        requests python-multipart \
        python-dotenv \
        httpx aiofiles \
        pillow python-magic \
        colivara-py
    
    touch "${VENV_SETUP_DONE}"
    log_success "Project virtual environment configured"
    
    cd "${SCRIPT_DIR}"
}

# ============================================================================
# MCP MERMAID ENHANCED SETUP
# ============================================================================

setup_mcp_mermaid() {
    local MCP_SETUP_DONE="${SCRIPT_DIR}/.mcp_setup_done"
    
    if [[ -f "${MCP_SETUP_DONE}" ]]; then
        log_info "MCP Mermaid Enhanced already configured"
        return 0
    fi
    
    log_info "Setting up MCP Mermaid Enhanced..."
    
    # Check if Node.js is installed
    if ! command -v node >/dev/null 2>&1; then
        log_info "Node.js not found, installing..."
        
        # Detect OS and install Node.js
        if [[ -f /etc/os-release ]]; then
            source /etc/os-release
            OS=$ID
        else
            log_error "Cannot detect OS"
            return 1
        fi
        
        case "$OS" in
            ubuntu|debian)
                log_info "Installing Node.js on Debian/Ubuntu..."
                curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                sudo apt-get install -y nodejs
                ;;
            centos|rhel|fedora)
                log_info "Installing Node.js on RHEL/CentOS/Fedora..."
                curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
                sudo dnf install -y nodejs || sudo yum install -y nodejs
                ;;
            arch|manjaro)
                log_info "Installing Node.js on Arch..."
                sudo pacman -S --noconfirm nodejs npm
                ;;
            *)
                log_error "Unsupported OS for automatic Node.js installation: $OS"
                log_info "Please install Node.js 18+ manually from https://nodejs.org/"
                return 1
                ;;
        esac
        
        # Verify installation
        if ! command -v node >/dev/null 2>&1; then
            log_error "Node.js installation failed"
            return 1
        fi
        
        log_success "Node.js $(node --version) installed successfully"
    else
        log_info "Node.js $(node --version) already installed"
    fi
    
    # Check if npm is available
    if ! command -v npm >/dev/null 2>&1; then
        log_error "npm not found, please install Node.js with npm"
        return 1
    fi
    
    # Install mcp-mermaid-enhanced globally
    log_info "Installing mcp-mermaid-enhanced via npm..."
    if sudo npm install -g mcp-mermaid-enhanced; then
        log_success "mcp-mermaid-enhanced installed successfully"
    else
        log_warning "Global installation failed, trying without sudo..."
        npm install -g mcp-mermaid-enhanced || {
            log_error "Failed to install mcp-mermaid-enhanced"
            return 1
        }
    fi
    
    # Verify installation
    if command -v mcp-mermaid-enhanced >/dev/null 2>&1; then
        log_success "mcp-mermaid-enhanced is available in PATH"
    else
        log_warning "mcp-mermaid-enhanced not in PATH, checking npm global path..."
        local npm_bin=$(npm config get prefix)/bin
        if [[ -f "${npm_bin}/mcp-mermaid-enhanced" ]]; then
            log_info "Found at: ${npm_bin}/mcp-mermaid-enhanced"
            log_info "Adding to PATH in environment"
            export PATH="${npm_bin}:${PATH}"
        else
            log_warning "mcp-mermaid-enhanced installed but not accessible"
        fi
    fi
    
    # Create mermaid diagrams output directory
    local mermaid_dir="${PROJECT_BACKEND_DIR}/mermaid_diagrams"
    mkdir -p "${mermaid_dir}"
    log_success "Mermaid diagrams directory created: ${mermaid_dir}"
    
    # Install Playwright for mermaid-cli (used by MCP for rendering)
    log_info "Installing Playwright browsers for diagram rendering..."
    if command -v npx >/dev/null 2>&1; then
        npx -y playwright install chromium --with-deps >/dev/null 2>&1 || {
            log_warning "Playwright installation had issues, but continuing..."
        }
        log_success "Playwright browsers installed"
    else
        log_warning "npx not available, skipping Playwright setup"
    fi
    
    touch "${MCP_SETUP_DONE}"
    log_success "MCP Mermaid Enhanced setup completed"
    
    return 0
}

# ============================================================================
# SYSTEMD SERVICES
# ============================================================================

create_systemd_services() {
    log_info "Creating systemd services..."
    
    local current_user
    current_user=$(whoami)
    
    # Embedding Service
    sudo tee /etc/systemd/system/colivara-embedding.service >/dev/null << EOF
[Unit]
Description=ColiVara Embedding Service
After=network.target

[Service]
User=${current_user}
WorkingDirectory=${COLIVARE_DIR}
Environment="COLIVARA_MODEL_PATH=${COLIVARE_DIR}/models_hub/vidore/colqwen2-v1.0"
Environment="HF_HUB_OFFLINE=1"
Environment="TRANSFORMERS_OFFLINE=1"
Environment="HF_DATASETS_OFFLINE=1"
ExecStart=${COLIVARE_DIR}/.venv/bin/python -m uvicorn handler:app --host 0.0.0.0 --port ${EMBEDDING_PORT}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/embedding.log
StandardError=append:${LOG_DIR}/embedding-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # ColiVara API Service (Docker Compose)
    sudo tee /etc/systemd/system/colivara-api.service >/dev/null << EOF
[Unit]
Description=ColiVara API Service
After=network.target docker.service
Requires=docker.service

[Service]
User=${current_user}
WorkingDirectory=${COLIVARA_DIR}
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose stop
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/api.log
StandardError=append:${LOG_DIR}/api-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # Query API Service (api.py)
    sudo tee /etc/systemd/system/colivara-query-api.service >/dev/null << EOF
[Unit]
Description=ColiVara Query API Service
After=network.target colivara-api.service colivara-embedding.service

[Service]
User=${current_user}
WorkingDirectory=${PROJECT_BACKEND_DIR}
Environment="PATH=${PROJECT_BACKEND_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="EMBEDDING_PORT=${EMBEDDING_PORT}"
Environment="COLIVARA_API_PORT=${COLIVARA_API_PORT}"
Environment="HOST_IP=${HOST_IP}"
ExecStart=${PROJECT_BACKEND_DIR}/.venv/bin/python ${PROJECT_BACKEND_DIR}/api.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/query-api.log
StandardError=append:${LOG_DIR}/query-api-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # App Service (app.py)
    sudo tee /etc/systemd/system/colivara-app.service >/dev/null << EOF
[Unit]
Description=ColiVara App Service
After=network.target colivara-api.service colivara-query-api.service

[Service]
User=${current_user}
WorkingDirectory=${PROJECT_BACKEND_DIR}
Environment="PATH=${PROJECT_BACKEND_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="APP_PORT=${APP_PORT}"
Environment="QUERY_API_PORT=${QUERY_API_PORT}"
Environment="HOST_IP=${HOST_IP}"
ExecStart=${PROJECT_BACKEND_DIR}/.venv/bin/python ${PROJECT_BACKEND_DIR}/app.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/app.log
StandardError=append:${LOG_DIR}/app-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable colivara-embedding.service
    sudo systemctl enable colivara-api.service
    sudo systemctl enable colivara-query-api.service
    sudo systemctl enable colivara-app.service
    
    log_success "Systemd services created and enabled"
}

# ============================================================================
# HEALTH MONITORING
# ============================================================================

setup_health_monitoring() {
    log_info "Setting up health monitoring..."
    
    cat > "${SCRIPT_DIR}/healthcheck.sh" << EOF
#!/bin/bash

check_service() {
    local url=\$1
    local service_name=\$2
    
    if ! curl -sf "\${url}" >/dev/null 2>&1; then
        echo "\$(date): \${service_name} is down, restarting..."
        sudo systemctl restart "\${service_name}"
    fi
}

check_service "http://localhost:${EMBEDDING_PORT}/health" "colivara-embedding.service"
check_service "http://localhost:${COLIVARA_API_PORT}/v1/docs" "colivara-api.service"
check_service "http://localhost:${QUERY_API_PORT}/health" "colivara-query-api.service"
check_service "http://localhost:${APP_PORT}" "colivara-app.service"
EOF
    
    chmod +x "${SCRIPT_DIR}/healthcheck.sh"
    
    if ! crontab -l 2>/dev/null | grep -q "healthcheck.sh"; then
        (crontab -l 2>/dev/null; echo "*/5 * * * * ${SCRIPT_DIR}/healthcheck.sh >> ${LOG_DIR}/healthcheck.log 2>&1") | crontab -
        log_success "Health monitoring configured (runs every 5 minutes)"
    else
        log_info "Health monitoring already configured"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print_banner() {
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗ ██████╗ ██╗     ██╗██╗   ██╗ █████╗ ██████╗   ║
║  ██╔════╝██╔═══██╗██║     ██║██║   ██║██╔══██╗██╔══██╗  ║
║  ██║     ██║   ██║██║     ██║██║   ██║███████║██████╔╝  ║
║  ██║     ██║   ██║██║     ██║╚██╗ ██╔╝██╔══██║██╔══██╗  ║
║  ╚██████╗╚██████╔╝███████╗██║ ╚████╔╝ ██║  ██║██║  ██║  ║
║   ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝  ║
║                                                           ║
║            Complete Setup Script v1.0                     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
}

print_summary() {
    cat << EOF

╔═══════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE                         ║
╚═══════════════════════════════════════════════════════════╝

Service Endpoints:
  Embedding Service:  http://${HOST_IP}:${EMBEDDING_PORT}/health
  ColiVara API:       http://${HOST_IP}:${COLIVARA_API_PORT}/v1/docs
  Query API:          http://${HOST_IP}:${QUERY_API_PORT}
  App Service:        http://${HOST_IP}:${APP_PORT}
  MinIO Console:      http://${HOST_IP}:${MINIO_PORT}
  Ollama API:         http://${HOST_IP}:${OLLAMA_PORT}

Credentials:
  Admin Username:     admin
  Admin Password:     admin123
  MinIO User:         ${MINIO_ROOT_USER}
  MinIO Password:     ${MINIO_ROOT_PASSWORD}

Project Structure:
  Script Directory:   ${SCRIPT_DIR}
  ColiVarE (Embed):   ${COLIVARE_DIR}
  ColiVara (API):     ${COLIVARA_DIR}
  App Files:          ${PROJECT_BACKEND_DIR}/app.py, api.py
  Templates:          ${PROJECT_BACKEND_DIR}/templates/

Logs:
  All logs:           ${LOG_DIR}/

Service Management:
  Start all:          sudo systemctl start colivara-{embedding,api,query-api,app}.service
  Stop all:           sudo systemctl stop colivara-{embedding,api,query-api,app}.service
  Status:             sudo systemctl status colivara-embedding.service
  View logs:          journalctl -u colivara-embedding.service -f
  
Health Monitoring:
  Automatic health checks run every 5 minutes via cron
  Manual check:       ${SCRIPT_DIR}/healthcheck.sh

EOF
}

reset_setup() {
    log_warning "Resetting setup state..."
    rm -f "${EMBEDDING_SETUP_DONE}" "${API_SETUP_DONE}" "${MODELS_VALIDATED}" "${OLLAMA_SETUP_DONE}" "${PYTHON_SETUP_DONE}" "${VENV_SETUP_DONE}" "${MCP_SETUP_DONE}"
    
    log_info "Stopping all services..."
    sudo systemctl stop colivara-embedding.service 2>/dev/null || true
    sudo systemctl stop colivara-api.service 2>/dev/null || true
    sudo systemctl stop colivara-query-api.service 2>/dev/null || true
    sudo systemctl stop colivara-app.service 2>/dev/null || true
    
    if [[ -d "${COLIVARA_DIR}" ]]; then
        cd "${COLIVARA_DIR}" && docker-compose down -v 2>/dev/null || true
        cd "${SCRIPT_DIR}"
    fi
    
    log_success "Setup state reset complete"
}

verify_project_files() {
    log_info "Verifying project files..."
    
    local required_files=("app.py" "api.py")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${PROJECT_BACKEND_DIR}/${file}" ]]; then
            missing_files+=("${file}")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing required project files: ${missing_files[*]}"
        log_error "Please ensure app.py and api.py are in the same directory as this script"
        exit 1
    fi
    
    if [[ -d "${PROJECT_BACKEND_DIR}/templates" ]]; then
        log_success "Templates directory found"
    else
        log_warning "Templates directory not found - app.py may need it"
    fi
    
    log_success "Project files verified"
}

main() {
    print_banner
    
    # Parse command line arguments
    case "${1:-}" in
        --reset)
            reset_setup
            exit 0
            ;;
        --help|-h)
            cat << EOF
ColiVara Complete Setup Script

Usage: $0 [OPTIONS]

Options:
  --reset       Reset all setup state and stop services
  --help, -h    Show this help message
  
Environment Variables:
  EMBEDDING_PORT       Port for embedding service (default: 8000)
  COLIVARA_API_PORT    Port for ColiVara API (default: 8001)
  APP_PORT             Port for app service (default: 5000)
  QUERY_API_PORT       Port for query API (default: 5001)
  MINIO_PORT           Port for MinIO console (default: 9001)
  OLLAMA_PORT          Port for Ollama service (default: 11434)
  MINIO_ROOT_USER      MinIO admin username (default: miniokey)
  MINIO_ROOT_PASSWORD  MinIO admin password (default: miniosecret)
  OLLAMA_MODEL         Ollama model to pull (default: qwen2.5vl:32b)

Project Structure:
  Your repository should contain:
  - setup_colivara.sh (this script)
  - app.py (Flask application)
  - api.py (Query API service)
  - templates/ (HTML templates directory)
  
  The script will automatically:
  - Clone ColiVara and ColiVarE repositories
  - Download the ColQwen2 model from Hugging Face
  - Install Python 3.10 if not available
  - Install and configure Ollama
  - Set up all services with systemd

Examples:
  # Standard installation
  ./setup_colivara.sh
  
  # Reset and start fresh
  ./setup_colivara.sh --reset
  
  # Custom ports
  EMBEDDING_PORT=9000 COLIVARA_API_PORT=9001 ./setup_colivara.sh

Repository: https://github.com/tjmlabs/ColiVara
License: MIT
EOF
            exit 0
            ;;
        "")
            # Normal execution
            ;;
        *)
            log_error "Unknown option: $1"
            log_info "Use --help for usage information"
            exit 1
            ;;
    esac
    
    log_info "Starting ColiVara setup process..."
    log_info "Host IP detected: ${HOST_IP}"
    log_info "Script directory: ${SCRIPT_DIR}"
    
    # Verify project structure
    verify_project_files
    
    # Execute setup steps
    log_info "Step 1/9: Checking system requirements..."
    check_system_requirements
    
    log_info "Step 2/9: Freeing ports..."
    free_ports
    
    log_info "Step 3/9: Setting up Ollama..."
    setup_ollama || log_warning "Ollama setup failed, continuing anyway"
    
    log_info "Step 4/9: Setting up embedding service..."
    setup_embedding_service || {
        log_error "Embedding service setup failed"
        exit 1
    }
    
    log_info "Step 5/9: Setting up API service..."
    setup_api_service || {
        log_error "API service setup failed"
        exit 1
    }
    
    log_info "Step 6/10: Setting up project virtual environment..."
    setup_project_venv || {
        log_error "Project venv setup failed"
        exit 1
    }
    
    log_info "Step 7/10: Setting up MCP Mermaid Enhanced..."
    setup_mcp_mermaid || {
        log_warning "MCP Mermaid setup failed, but continuing (diagram features may not work)"
    }
    
    log_info "Step 8/10: Creating systemd services..."
    create_systemd_services
    
    log_info "Step 9/10: Setting up health monitoring..."
    setup_health_monitoring
    
    log_info "Step 10/10: Starting persistent services..."
    sudo systemctl start colivara-embedding.service
    sudo systemctl start colivara-api.service
    sudo systemctl start colivara-query-api.service
    sudo systemctl start colivara-app.service
    
    # Wait for services to stabilize
    sleep 5
    
    # Verify services are running
    log_info "Verifying service status..."
    local all_services_ok=true
    
    if ! systemctl is-active --quiet colivara-embedding.service; then
        log_warning "Embedding service not running"
        all_services_ok=false
    fi
    
    if ! systemctl is-active --quiet colivara-api.service; then
        log_warning "API service not running"
        all_services_ok=false
    fi
    
    if ! systemctl is-active --quiet colivara-query-api.service; then
        log_warning "Query API service not running"
        all_services_ok=false
    fi
    
    if ! systemctl is-active --quiet colivara-app.service; then
        log_warning "App service not running"
        all_services_ok=false
    fi
    
    if [[ "${all_services_ok}" == "true" ]]; then
        log_success "All services are running"
    else
        log_warning "Some services may need attention"
        log_info "Check logs with: journalctl -u colivara-embedding.service -f"
    fi
    
    print_summary
    
    log_info "Testing service endpoints..."
    
    if curl -sf "http://localhost:${EMBEDDING_PORT}/health" >/dev/null 2>&1; then
        log_success "✓ Embedding service is responding"
    else
        log_warning "✗ Embedding service not responding yet (may still be initializing)"
    fi
    
    if curl -sf "http://localhost:${COLIVARA_API_PORT}/v1/docs" >/dev/null 2>&1; then
        log_success "✓ ColiVara API is responding"
    else
        log_warning "✗ ColiVara API not responding yet (may still be initializing)"
    fi
    
    if curl -sf "http://localhost:${QUERY_API_PORT}/health" >/dev/null 2>&1; then
        log_success "✓ Query API is responding"
    else
        log_warning "✗ Query API not responding yet (check if api.py has /health endpoint)"
    fi
    
    if curl -sf "http://localhost:${APP_PORT}" >/dev/null 2>&1; then
        log_success "✓ App service is responding"
    else
        log_warning "✗ App service not responding yet (may still be initializing)"
    fi
    
    log_success "Setup complete! Services are running as systemd units."
    log_info ""
    log_info "Quick Commands:"
    log_info "  View all service status:  systemctl status 'colivara-*'"
    log_info "  Restart a service:        sudo systemctl restart colivara-embedding.service"
    log_info "  View logs:                tail -f ${LOG_DIR}/embedding.log"
    log_info "  Run health check:         ${SCRIPT_DIR}/healthcheck.sh"
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi