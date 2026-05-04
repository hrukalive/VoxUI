use std::fs;
use std::path::Path;

#[test]
fn inference_source_does_not_spawn_or_embed_python_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let denied = [
        "std::process::Command",
        "python.exe",
        "python/",
        "pyo3",
        "PyModule",
        "PyObject",
        "VoxCPM.generate(",
    ];

    for entry in fs::read_dir(src).expect("read src dir") {
        let path = entry.expect("read src entry").path();
        if path.extension().and_then(|v| v.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source file");
        for needle in denied {
            assert!(
                !text.contains(needle),
                "{} contains prohibited runtime token `{}`",
                path.display(),
                needle
            );
        }
    }

    assert!(
        !root.join("python").exists(),
        "runtime helper directory must not exist in voxui-inference"
    );
}
