use std::fs;

use tempfile::TempDir;
use voxui_desktop::model_discovery::{choice_id, discover_models};

#[cfg(unix)]
use std::{ffi::OsString, os::unix::ffi::OsStringExt};

#[test]
fn discovers_base_and_lora_choices() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(model_dir.join("lora_a1.gguf"), [1u8; 2]).unwrap();
    fs::write(model_dir.join("lora_a2.gguf"), [2u8; 3]).unwrap();
    fs::write(model_dir.join("notes.txt"), b"ignored").unwrap();

    let choices = discover_models(temp.path()).unwrap();
    let names = choices
        .iter()
        .map(|c| c.display_name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "voxcpm2-fp16",
            "voxcpm2-fp16 | lora_a1",
            "voxcpm2-fp16 | lora_a2",
        ]
    );
    assert_eq!(choices[0].model_bytes, 4);
    assert_eq!(choices[1].lora_bytes, 2);
    assert_eq!(choices[2].lora_bytes, 3);
}

#[test]
fn ignores_directories_without_model_gguf() {
    let temp = TempDir::new().unwrap();
    let invalid = temp.path().join("not-a-model");
    fs::create_dir(&invalid).unwrap();
    fs::write(invalid.join("lora_a1.gguf"), [1u8; 2]).unwrap();

    let choices = discover_models(temp.path()).unwrap();

    assert!(choices.is_empty());
}

#[test]
fn discovers_models_with_deterministic_ordering_and_ids() {
    let temp = TempDir::new().unwrap();
    let z_model = temp.path().join("z-model");
    let a_model = temp.path().join("a-model");
    fs::create_dir(&z_model).unwrap();
    fs::create_dir(&a_model).unwrap();

    fs::write(z_model.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(z_model.join("z_lora.gguf"), [1u8; 2]).unwrap();
    fs::write(z_model.join("a_lora.gguf"), [2u8; 3]).unwrap();
    fs::write(a_model.join("model.gguf"), [3u8; 5]).unwrap();
    fs::write(a_model.join("b_lora.gguf"), [4u8; 6]).unwrap();
    fs::write(a_model.join("a_lora.gguf"), [5u8; 7]).unwrap();

    let choices = discover_models(temp.path()).unwrap();
    let names = choices
        .iter()
        .map(|choice| choice.display_name.as_str())
        .collect::<Vec<_>>();
    let ids = choices
        .iter()
        .map(|choice| choice.id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "a-model",
            "a-model | a_lora",
            "a-model | b_lora",
            "z-model",
            "z-model | a_lora",
            "z-model | z_lora",
        ]
    );
    assert_eq!(
        ids,
        vec![
            "a-model",
            "a-model|a_lora.gguf",
            "a-model|b_lora.gguf",
            "z-model",
            "z-model|a_lora.gguf",
            "z-model|z_lora.gguf",
        ]
    );
}

#[cfg(unix)]
#[test]
fn errors_on_non_utf8_discovered_model_dir_name() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp
        .path()
        .join(OsString::from_vec(b"bad-model-\xFF".to_vec()));
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();

    let error = discover_models(temp.path()).unwrap_err();

    assert!(error
        .to_string()
        .contains("model directory name is not UTF-8"));
}

#[cfg(unix)]
#[test]
fn errors_on_non_utf8_discovered_lora_file_name() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();
    fs::write(model_dir.join("model.gguf"), [0u8; 4]).unwrap();
    fs::write(
        model_dir.join(OsString::from_vec(b"bad-lora-\xFF.gguf".to_vec())),
        [1u8; 2],
    )
    .unwrap();

    let error = discover_models(temp.path()).unwrap_err();

    assert!(error
        .to_string()
        .contains("LoRA candidate file name is not UTF-8"));
}

#[test]
fn choice_ids_are_relative_and_stable() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("voxcpm2-fp16");
    fs::create_dir(&model_dir).unwrap();

    assert_eq!(
        choice_id(temp.path(), &model_dir, None).unwrap(),
        "voxcpm2-fp16"
    );
    assert_eq!(
        choice_id(
            temp.path(),
            &model_dir,
            Some(&model_dir.join("lora_a1.gguf"))
        )
        .unwrap(),
        "voxcpm2-fp16|lora_a1.gguf"
    );
}
