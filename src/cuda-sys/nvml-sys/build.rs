use std::env;
use std::path::PathBuf;

use cuda_finder::{find_cuda, find_cuda_windows};

fn main() {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_type("^NVML.*")
        .allowlist_var("^NVML.*")
        .allowlist_type("^nvml.*")
        .allowlist_function("^nvml.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .generate_comments(false)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .clang_arg("-I/usr/local/cuda/include")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        for path in find_cuda() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    };

    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
}
