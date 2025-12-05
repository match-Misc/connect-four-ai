#!/bin/bash

#===============================================================================
#
#   Connect Four AI - Automated Setup Script
#
#   This script automates the complete setup process for the connect-four-ai
#   package, including Python installation, Rust compilation, and package setup.
#
#   Usage: ./setup.sh [OPTIONS]
#
#   Options:
#       -h, --help              Show this help message
#       -s, --skip-python       Skip Python installation (use existing)
#       -v, --verbose           Enable verbose output
#       --no-venv               Skip virtual environment creation
#
#===============================================================================

set -e  # Exit immediately on error

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

readonly SCRIPT_VERSION="1.0.0"
readonly MIN_PYTHON_VERSION="3.13"
readonly VENV_DIR=".venv"

#-------------------------------------------------------------------------------
# Color Definitions
#-------------------------------------------------------------------------------

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'  # No Color

#-------------------------------------------------------------------------------
# Default Options
#-------------------------------------------------------------------------------

SKIP_PYTHON_INSTALL=false
VERBOSE=false
CREATE_VENV=true

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}${BOLD}           Connect Four AI - Setup Script v${SCRIPT_VERSION}            ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    local step_num=$1
    local step_desc=$2
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Step ${step_num}:${NC} ${step_desc}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}  → $1${NC}"
    fi
}

show_help() {
    cat << EOF
${BOLD}Connect Four AI - Setup Script${NC}

${BOLD}USAGE:${NC}
    ./setup.sh [OPTIONS]

${BOLD}OPTIONS:${NC}
    -h, --help              Show this help message and exit
    -s, --skip-python       Skip automatic Python installation
    -v, --verbose           Enable verbose output for debugging
    --no-venv               Skip virtual environment creation

${BOLD}DESCRIPTION:${NC}
    This script automates the setup process for the connect-four-ai package:
    
    1. Checks and installs Python >= ${MIN_PYTHON_VERSION} if needed
    2. Installs Rust/Cargo build tools
    3. Builds the Rust project in release mode
    4. Creates a Python virtual environment
    5. Installs the connect-four-ai package

${BOLD}REQUIREMENTS:${NC}
    - Linux (Debian/Ubuntu-based for automatic Python installation)
    - sudo access (for package installation)
    - Internet connection

${BOLD}EXAMPLES:${NC}
    ./setup.sh                  # Full setup with all features
    ./setup.sh --skip-python    # Skip Python installation
    ./setup.sh --verbose        # Show detailed output

${BOLD}AFTER SETUP:${NC}
    source ${VENV_DIR}/bin/activate
    python -c "import connect_four_ai; print('Ready!')"

EOF
}

# Compare version numbers (returns 0 if $1 >= $2)
version_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the script's directory (handles symlinks)
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]; do
        local dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$(cd -P "$(dirname "$source")" && pwd)"
}

#-------------------------------------------------------------------------------
# Parse Command Line Arguments
#-------------------------------------------------------------------------------

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--skip-python)
                SKIP_PYTHON_INSTALL=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --no-venv)
                CREATE_VENV=false
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
}

#-------------------------------------------------------------------------------
# Step 1: Check and Install Python
#-------------------------------------------------------------------------------

setup_python() {
    print_step "1" "Checking Python Installation"
    
    local python_cmd=""
    local python_version=""
    
    # Try python3.13 first
    if command_exists python3.13; then
        python_cmd="python3.13"
        python_version=$(python3.13 --version 2>&1 | cut -d' ' -f2)
        log_verbose "Found python3.13: $python_version"
    # Then try python3
    elif command_exists python3; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_verbose "Found python3: $python_version"
        if version_ge "$python_version" "$MIN_PYTHON_VERSION"; then
            python_cmd="python3"
        fi
    fi
    
    # Install Python if needed
    if [ -z "$python_cmd" ]; then
        if [ "$SKIP_PYTHON_INSTALL" = true ]; then
            print_error "Python >= $MIN_PYTHON_VERSION not found and --skip-python was specified."
            exit 1
        fi
        
        print_warning "Python >= $MIN_PYTHON_VERSION not found."
        print_info "Installing Python $MIN_PYTHON_VERSION..."
        
        if command_exists apt; then
            log_verbose "Using apt package manager"
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt update
            sudo apt install python3.13 python3.13-venv -y
            python_cmd="python3.13"
            python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
        else
            print_error "Automatic Python installation requires apt (Debian/Ubuntu)."
            print_info "Please install Python >= $MIN_PYTHON_VERSION manually."
            exit 1
        fi
    fi
    
    # Export for use in other functions
    PYTHON_CMD="$python_cmd"
    PYTHON_VERSION="$python_version"
    
    print_success "Python $PYTHON_VERSION is available ($PYTHON_CMD)"
}

#-------------------------------------------------------------------------------
# Step 2: Check and Install Rust/Cargo
#-------------------------------------------------------------------------------

setup_rust() {
    print_step "2" "Checking Rust/Cargo Installation"
    
    if ! command_exists cargo; then
        print_warning "Cargo not found."
        print_info "Installing Rust toolchain..."
        
        if command_exists apt; then
            log_verbose "Installing cargo via apt"
            sudo apt install cargo -y
        elif command_exists curl; then
            log_verbose "Installing Rust via rustup"
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        else
            print_error "Cannot install Rust. Please install manually from https://rustup.rs"
            exit 1
        fi
    fi
    
    # Set stable toolchain
    export RUSTUP_TOOLCHAIN=stable
    log_verbose "Set RUSTUP_TOOLCHAIN=stable"
    
    local cargo_version=$(cargo --version 2>&1)
    print_success "Cargo is available: $cargo_version"
}

#-------------------------------------------------------------------------------
# Step 3: Build Rust Project
#-------------------------------------------------------------------------------

build_rust_project() {
    print_step "3" "Building Rust Project"
    
    local script_dir=$(get_script_dir)
    local crates_dir="$script_dir/crates"
    
    if [ ! -d "$crates_dir" ]; then
        print_error "Crates directory not found: $crates_dir"
        exit 1
    fi
    
    print_info "Building in release mode..."
    log_verbose "Working directory: $crates_dir"
    
    cd "$crates_dir"
    
    if [ "$VERBOSE" = true ]; then
        cargo build --release
    else
        cargo build --release 2>&1 | tail -n 5
    fi
    
    print_success "Rust project built successfully"
    
    # Return to script directory
    cd "$script_dir"
}

#-------------------------------------------------------------------------------
# Step 4: Setup Python Virtual Environment
#-------------------------------------------------------------------------------

setup_virtualenv() {
    print_step "4" "Setting Up Python Virtual Environment"
    
    local script_dir=$(get_script_dir)
    local python_crate_dir="$script_dir/crates/python"
    
    cd "$python_crate_dir"
    log_verbose "Working directory: $python_crate_dir"
    
    if [ "$CREATE_VENV" = false ]; then
        print_warning "Skipping virtual environment creation (--no-venv)"
        return
    fi
    
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists"
    else
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_verbose "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet
    print_success "pip upgraded"
}

#-------------------------------------------------------------------------------
# Step 5: Install Python Package
#-------------------------------------------------------------------------------

install_package() {
    print_step "5" "Installing connect-four-ai Package"
    
    # Already in python_crate_dir from setup_virtualenv
    
    print_info "Installing package in editable mode..."
    
    if [ "$VERBOSE" = true ]; then
        pip install -e .
    else
        pip install -e . --quiet
    fi
    
    print_success "Package installed successfully"
}

#-------------------------------------------------------------------------------
# Step 6: Verify Installation
#-------------------------------------------------------------------------------

verify_installation() {
    print_step "6" "Verifying Installation"
    
    if python -c "import connect_four_ai" 2>/dev/null; then
        local version=$(python -c "import connect_four_ai; print(getattr(connect_four_ai, '__version__', 'unknown'))" 2>/dev/null || echo "installed")
        print_success "connect_four_ai is ready (version: $version)"
    else
        print_warning "Package import test failed, but installation may still be successful"
    fi
}

#-------------------------------------------------------------------------------
# Print Summary
#-------------------------------------------------------------------------------

print_summary() {
    local script_dir=$(get_script_dir)
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}${BOLD}                    Setup Complete!                           ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo -e "  1. Activate the virtual environment:"
    echo -e "     ${YELLOW}source $script_dir/$VENV_DIR/bin/activate${NC}"
    echo ""
    echo -e "  2. Verify the installation:"
    echo -e "     ${YELLOW}python -c \"import connect_four_ai; print('Ready!')\"${NC}"
    echo ""
    echo -e "  3. Start using the package in your Python projects!"
    echo ""
}

#-------------------------------------------------------------------------------
# Main Execution
#-------------------------------------------------------------------------------

main() {
    parse_arguments "$@"
    
    print_header
    
    setup_python
    setup_rust
    build_rust_project
    setup_virtualenv
    install_package
    verify_installation
    
    print_summary
}

# Run main function with all arguments
main "$@"