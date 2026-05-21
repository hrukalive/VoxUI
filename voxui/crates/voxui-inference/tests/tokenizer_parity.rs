use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use voxui_inference::VoxTokenizer;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn model_dir() -> PathBuf {
    repo_root().join("models/voxcpm2-fp16")
}

fn direct_encode(text: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(model_dir().join("tokenizer.json")).unwrap();
    tokenizer.encode(text, false).unwrap().get_ids().to_vec()
}

#[test]
fn splits_multichar_chinese_tokens_like_python_reference() {
    let tokenizer = VoxTokenizer::from_dir(&model_dir()).unwrap();
    let text = "\u{7b2c}\u{4e00}\u{767e}\u{4e94}\u{5341}\u{56db}\u{6761}\u{88c1}\u{5b9a}\u{9002}\u{7528}\u{4e8e}\u{4e0b}\u{5217}\u{8303}\u{56f4}";

    let actual = tokenizer.encode(text).unwrap();

    assert_eq!(direct_encode(text), vec![59320, 47804]);
    assert_eq!(
        actual,
        vec![
            59320, 59438, 59382, 59635, 59637, 59482, 59614, 59548, 59659, 59421, 59823, 59415,
            59433, 59454, 59913, 59951, 60016,
        ]
    );
}

#[test]
fn splits_sentence_chinese_tokens_like_python_reference() {
    let tokenizer = VoxTokenizer::from_dir(&model_dir()).unwrap();
    let text = "\u{6211}\u{8bf4}\u{4ec0}\u{4e48}\u{6765}\u{7740}\u{ff0c}\u{6211}\u{4e0d}\u{77e5}\u{9053}\u{4f60}\u{662f}\u{4ec0}\u{4e48}\u{813e}\u{6c14}\u{554a}\u{ff0c}\u{6211}\u{80af}\u{5b9a}\u{8981}\u{90a6}\u{90a6}\u{6572}\u{4e00}\u{4e0b}\u{3002}";

    assert_eq!(
        tokenizer.encode(text).unwrap(),
        vec![
            59320, 59422, 59522, 59747, 59551, 59455, 59620, 65, 59422, 59397, 59618, 59575, 59496,
            59390, 59747, 59551, 61558, 59754, 60240, 65, 59422, 60630, 59421, 59430, 61192, 61192,
            61868, 59382, 59454, 66,
        ]
    );
}

#[test]
fn keeps_non_chinese_tokenization_direct() {
    let tokenizer = VoxTokenizer::from_dir(&model_dir()).unwrap();
    let cases = [
        "\u{3053}\u{308c}\u{306f}\u{30c6}\u{30b9}\u{30c8}\u{3067}\u{3059}\u{306a}\u{306e}\u{ff01}\u{306a}\u{3093}\u{3067}\u{3060}\u{3088}\u{ff1f}",
        "This inference matrix sentence exercises q4 language model coverage on every backend.",
    ];

    for text in cases {
        assert_eq!(
            tokenizer.encode(text).unwrap(),
            direct_encode(text),
            "{text}"
        );
    }
}
