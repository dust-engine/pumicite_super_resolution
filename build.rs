//! Standalone Cargo build for the DLSS-RR backend (the `dlss` feature).
//!
//! This path is fully independent of Bazel: it obtains the NVIDIA DLSS SDK,
//! compiles the NGX C shim with `cc`, links the NGX static library, and stages
//! the runtime DLL. Under Bazel this script is disabled (the consumer sets
//! `gen_build_script = "off"` on the crate) and Bazel supplies the C shim + SDK.
//!
//! Runs only when the `dlss` feature is enabled and the target isn't Apple. The
//! SDK comes from `$DLSS_SDK` if set, otherwise it is downloaded (with
//! `nyquest`) at a pinned commit. Set `DLSS_NO_FETCH=1` to skip the download
//! (offline / `cargo check`); the compile + link are then skipped with a
//! warning, so the crate still type-checks.

use std::path::{Path, PathBuf};

const DLSS_COMMIT: &str = "d1bef2006b41eefd9d44b0a05f123993f3acbf3c";

fn main() {
    println!("cargo:rerun-if-changed=src/dlss/dlss_wrapper.c");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=DLSS_NO_FETCH");

    // Only obtain/build DLSS when the feature is enabled (and not on Apple,
    // which uses the MetalFX backend).
    if std::env::var_os("CARGO_FEATURE_DLSS").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        return;
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let Some(sdk) = locate_sdk() else {
        println!(
            "cargo:warning=DLSS SDK unavailable (set DLSS_SDK, or unset DLSS_NO_FETCH to \
             download); skipping DLSS-RR compile/link."
        );
        return;
    };

    // Compile the NGX C shim around the SDK's static-inline DLSS-RR helpers.
    let mut build = cc::Build::new();
    build
        .file("src/dlss/dlss_wrapper.c")
        .include(sdk.join("include"));
    if let Ok(vulkan_sdk) = std::env::var("VULKAN_SDK") {
        build.include(Path::new(&vulkan_sdk).join("include"));
    }
    if let Err(e) = build.try_compile("dlss_helpers") {
        println!("cargo:warning=failed to compile the NGX C shim: {e}; skipping DLSS-RR link.");
        return;
    }

    // Link the NGX static library and stage the runtime DLL next to the build
    // output, baking its directory so the crate can hand it to NGX at runtime.
    let (lib_dir, ngx_lib, dll_dir, dll_name) = match target_os.as_str() {
        "windows" => (
            "lib/Windows_x86_64/x64",
            "nvsdk_ngx_d",
            "lib/Windows_x86_64/rel",
            "nvngx_dlssd.dll",
        ),
        _ => (
            "lib/Linux_x86_64",
            "nvsdk_ngx",
            "lib/Linux_x86_64/rel",
            "libnvidia-ngx-dlssd.so.310.6.0",
        ),
    };
    println!(
        "cargo:rustc-link-search=native={}",
        sdk.join(lib_dir).display()
    );
    println!("cargo:rustc-link-lib=static={ngx_lib}");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let src_dll = sdk.join(dll_dir).join(dll_name);
    if src_dll.is_file() {
        let _ = std::fs::copy(&src_dll, out_dir.join(dll_name));
        println!("cargo:rustc-env=DLSS_RUNTIME_DIR={}", out_dir.display());
    } else {
        println!(
            "cargo:warning=DLSS runtime library not found at {}",
            src_dll.display()
        );
    }
}

/// Resolves the DLSS SDK root: `$DLSS_SDK` if it looks valid, otherwise a pinned
/// download cached under `$OUT_DIR` (skipped when `DLSS_NO_FETCH` is set).
fn locate_sdk() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("DLSS_SDK") {
        let sdk = PathBuf::from(dir);
        if sdk.join("include").is_dir() {
            return Some(sdk);
        }
        println!(
            "cargo:warning=DLSS_SDK is set but {}/include was not found",
            sdk.display()
        );
    }

    if std::env::var_os("DLSS_NO_FETCH").is_some() {
        return None;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    let sdk = out_dir.join("dlss-sdk");
    if sdk.join("include").is_dir() {
        return Some(sdk);
    }
    match download_sdk(&sdk) {
        Ok(()) => Some(sdk),
        Err(e) => {
            println!("cargo:warning=failed to download DLSS SDK: {e}");
            None
        }
    }
}

/// Downloads the pinned DLSS SDK source tarball with `nyquest` and extracts it
/// into `dest` (so `dest/include`, `dest/lib/...` exist).
fn download_sdk(dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("https://codeload.github.com/NVIDIA/DLSS/tar.gz/{DLSS_COMMIT}");
    println!("cargo:warning=downloading DLSS SDK from {url}");

    nyquest_preset::register();
    let client = nyquest::ClientBuilder::default().build_blocking()?;
    let body = client
        .request(nyquest::blocking::Request::get(url))?
        .with_successful_status()?
        .bytes()?;

    // GitHub archives prefix every entry with `DLSS-<commit>/`; unpack to a temp
    // dir, then promote that single top-level directory to `dest`.
    let decoder = flate2::read::GzDecoder::new(body.as_slice());
    let mut archive = tar::Archive::new(decoder);
    let tmp = dest.with_extension("unpacking");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    archive.unpack(&tmp)?;

    let top = std::fs::read_dir(&tmp)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .ok_or("DLSS archive had no top-level directory")?;
    let _ = std::fs::remove_dir_all(dest);
    std::fs::rename(&top, dest)?;
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
