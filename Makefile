run:
	cargo build --release
	RUST_LOG=INFO ./target/release/mccs

build-all:
	make -j -C src/collectives
	cargo build --release