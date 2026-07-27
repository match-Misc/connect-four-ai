# Software Setup (Pixi)

This project has been modernized to use **Pixi**, a fast package manager that automatically isolates and installs all dependencies (Python, Node.js, and Rust tools) without modifying your host system.

## Prerequisites

1. **Install Pixi** (the package manager used for this project):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
   *Note: You may need to restart your terminal or source your shell profile after installation.*

2. **Install `curl` and `git`** (Optional but recommended, assuming Ubuntu/Debian):
   ```bash
   sudo apt update
   sudo apt install curl git
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/match-Misc/connect-four-ai
   cd connect-four-ai
   ```

2. Run the automated setup:
   ```bash
   pixi run setup
   ```
   *This single command will automatically download Rust/Cargo (if missing), compile the core AI engine, install the Python bindings, and build the frontend web applications.*

## Running the Showcase

You can launch the backend and frontend simultaneously with:
```bash
pixi run game
```

For camera calibration, stop the game and run:
```bash
pixi run calibrate
```

Everything is fully isolated inside the `.pixi` environment directory!
