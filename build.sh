cd crates
cargo build --release
cd python
pip install -e .
cd ../..