use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-search=../collectives/build/device");
    println!("cargo:rustc-link-lib=colldevice");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg("-I../collectives/include")
        .clang_arg("-I/usr/local/cuda/include")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        .allowlist_type("^mccsDev.*")
        .allowlist_function("^mccsKernel.*")
        .allowlist_var("^MCCS.*")
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .derive_eq(true)
        .derive_hash(true)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}