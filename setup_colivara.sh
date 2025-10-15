#!/usr/bin/env bash
# Setup script for Colivara Launcher (skeleton)
# Creates a virtual environment and installs Python dependencies.
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "Using python: $PYTHON"
echo "Creating virtual environment at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "NOTE: This is a skeleton setup script."
echo "Make sure Ollama is installed and running locally (if you plan to use local Ollama)."
echo "Customize environment variables in a .env file if needed (e.g. OLLAMA_HOST, INDEX_PATH)."
echo ""
echo "Done."
