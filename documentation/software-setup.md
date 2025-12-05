# Software Setup

This guide covers the complete setup process for the Connect Four AI project.

---

## Prerequisites

Before starting, ensure you have:
- A Linux-based operating system (Debian/Ubuntu recommended)
- `sudo` access for package installation
- An active internet connection

---

## Installation

First, clone the repository:

```shell
git clone https://github.com/benjaminrall/connect-four-ai.git
cd connect-four-ai
```

Choose one of the following setup methods:

| Method | Best For |
|--------|----------|
| [Automatic Setup](#automatic-setup) | Quick installation, recommended for most users |
| [Manual Setup](#manual-setup) | Custom configurations or troubleshooting |

---

## Automatic Setup

The easiest way to get started is using the automated setup script:

```shell
./setup.sh
```

### Available Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-s, --skip-python` | Skip automatic Python installation |
| `-v, --verbose` | Enable detailed output |
| `--no-venv` | Skip virtual environment creation |

### Examples

```shell
# Full setup with all features
./setup.sh

# Show detailed progress
./setup.sh --verbose
```

---

## Manual Setup

Follow these steps if you prefer manual installation or need to troubleshoot.

### Step 1: Check Python Version

The project requires **Python 3.13 or higher**.

```shell
python3 --version
```

If your version is lower than 3.13, proceed to install it:

```shell
# Add the deadsnakes PPA repository
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package lists
sudo apt update

# Install Python 3.13 with venv support
sudo apt install python3.13 python3.13-venv -y
```

### Step 2: Install Build Tools

Install Cargo (Rust's package manager) for building the native components:

```shell
sudo apt install cargo
```

> **Note:** If you encounter toolchain issues, set the stable toolchain:
> ```shell
> export RUSTUP_TOOLCHAIN=stable
> ```

### Step 3: Build the Rust Project

Navigate to the crates directory and build in release mode:

```shell
cd crates
cargo build --release
```

### Step 4: Set Up Python Environment

Create and configure a virtual environment for the Python bindings:

1. **Navigate to the Python crate directory:**
   ```shell
   cd crates/python
   ```

2. **Create a virtual environment:**
   ```shell
   python3.13 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```shell
   source .venv/bin/activate
   ```

4. **Install the package in editable mode:**
   ```shell
   pip install -e .
   ```

---

## Verification

After setup, verify the installation:

```shell
# Activate the virtual environment (if not already active)
source crates/python/.venv/bin/activate

# Test the import
python -c "import connect_four_ai; print('✓ Installation successful!')"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `python3.13: command not found` | Ensure Python 3.13 is installed and in your PATH |
| `cargo: command not found` | Install Cargo with `sudo apt install cargo` |
| Build fails with toolchain error | Run `export RUSTUP_TOOLCHAIN=stable` |
| Permission denied on `setup.sh` | Run `chmod +x setup.sh` first |
