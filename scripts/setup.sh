#!/usr/bin/env bash
set -e

# Change to the root of the repository
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "======================================"
echo "    Connect Four AI Setup"
echo "======================================"

# 1. Build Rust Project
echo ""
echo "[1/4] Building Rust Project..."
if ! command -v cargo &> /dev/null; then
    echo "Cargo not found. Installing Rust toolchain..."
    if command -v curl &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo "curl is required to install Rust. Please install curl and try again."
        exit 1
    fi
fi
(cd crates && cargo build --release)

# 2. Install Python Package
echo ""
echo "[2/4] Installing Python extension (connect_four_ai)..."
(cd crates/python && pip install -e .)

# 3. Install Robot Game Frontend Dependencies
echo ""
echo "[3/4] Installing Robot Game Frontend Dependencies..."
(cd robot-game/frontend && npm install)

# 4. Install Calibration Frontend Dependencies and Build
echo ""
echo "[4/4] Installing & Building Calibration Frontend..."
(cd computer-vision/calibration/frontend && npm install && npm run build)

echo ""
echo "======================================"
echo "Setup Complete! You can now run:"
echo "  pixi run game"
echo "  pixi run calibrate"
echo "======================================"
