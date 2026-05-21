#[cfg(feature = "cuda")]
use std::process::Command;
#[cfg(feature = "cuda")]
use std::time::Duration;

const MIB: u64 = 1024 * 1024;
#[cfg(feature = "cuda")]
const VRAM_WARNING_THRESHOLD_MIB: u64 = 7 * 1024;
#[cfg(feature = "cuda")]
const CHILD_MODEL_ENV: &str = "VOXUI_VRAM_CHILD_MODEL";
#[cfg(feature = "cuda")]
const CHILD_JSON_PREFIX: &str = "VOXUI_VRAM_JSON=";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessMemorySnapshot {
    used_mib: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlobalCudaMemorySnapshot {
    free_mib: u64,
    total_mib: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemorySample {
    process: Option<ProcessMemorySnapshot>,
    global: Option<GlobalCudaMemorySnapshot>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct ModelVramReport {
    model_name: String,
    baseline: MemorySample,
    after_load: MemorySample,
    after_synth: MemorySample,
    peak_process_delta_mib: Option<i64>,
}

fn parse_process_memory_mib(output: &str, pid: u32) -> Option<ProcessMemorySnapshot> {
    let mut used_mib = 0u64;
    let mut found = false;

    for line in output.lines() {
        let mut parts = line.split(',').map(str::trim);
        let Some(row_pid_text) = parts.next() else {
            continue;
        };
        let Some(row_used_text) = parts.next() else {
            continue;
        };
        let Ok(row_pid) = row_pid_text.parse::<u32>() else {
            continue;
        };
        if row_pid != pid {
            continue;
        }
        let row_used_text = row_used_text
            .strip_suffix("MiB")
            .unwrap_or(row_used_text)
            .trim();
        let Ok(row_used_mib) = row_used_text.parse::<u64>() else {
            continue;
        };
        used_mib += row_used_mib;
        found = true;
    }

    found.then_some(ProcessMemorySnapshot { used_mib })
}

fn parse_windows_dedicated_usage_bytes(output: &str) -> Option<ProcessMemorySnapshot> {
    let used_bytes = output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .parse::<u64>()
        .ok()?;
    Some(ProcessMemorySnapshot {
        used_mib: used_bytes / MIB,
    })
}

#[cfg(feature = "cuda")]
fn current_process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    std::thread::sleep(Duration::from_millis(100));
    let nvidia_smi_snapshot = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|output| {
            if !output.status.success() {
                return None;
            }
            let stdout = String::from_utf8(output.stdout).ok()?;
            parse_process_memory_mib(&stdout, std::process::id())
        });

    nvidia_smi_snapshot.or_else(windows_current_process_memory_snapshot)
}

#[cfg(all(feature = "cuda", windows))]
fn windows_current_process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    let pid = std::process::id();
    let script = format!(
        "$samples = (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object {{ $_.InstanceName -like 'pid_{pid}_*' }}; if ($samples) {{ [int64](($samples | Measure-Object -Property CookedValue -Sum).Sum) }}"
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_windows_dedicated_usage_bytes(&stdout)
}

#[cfg(all(feature = "cuda", not(windows)))]
fn windows_current_process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    None
}

#[cfg(feature = "cuda")]
fn cuda_global_memory_snapshot() -> Option<GlobalCudaMemorySnapshot> {
    let (free, total) = candle_core::cuda::cudarc::driver::result::mem_get_info().ok()?;
    Some(GlobalCudaMemorySnapshot {
        free_mib: (free as u64) / MIB,
        total_mib: (total as u64) / MIB,
    })
}

#[cfg(feature = "cuda")]
fn take_memory_sample() -> MemorySample {
    MemorySample {
        process: current_process_memory_snapshot(),
        global: cuda_global_memory_snapshot(),
    }
}

fn process_delta_mib(
    baseline: Option<ProcessMemorySnapshot>,
    sample: Option<ProcessMemorySnapshot>,
) -> Option<i64> {
    Some(sample?.used_mib as i64 - baseline?.used_mib as i64)
}

fn peak_process_delta_mib(
    baseline: Option<ProcessMemorySnapshot>,
    after_load: Option<ProcessMemorySnapshot>,
    after_synth: Option<ProcessMemorySnapshot>,
) -> Option<i64> {
    let load_delta = process_delta_mib(baseline, after_load);
    let synth_delta = process_delta_mib(baseline, after_synth);
    match (load_delta, synth_delta) {
        (Some(load), Some(synth)) => Some(load.max(synth)),
        (Some(load), None) => Some(load),
        (None, Some(synth)) => Some(synth),
        (None, None) => None,
    }
}

fn format_mib(mib: u64) -> String {
    format!("{mib} MiB ({:.2} GiB)", mib as f64 / 1024.0)
}

fn format_signed_mib(mib: i64) -> String {
    let sign = if mib >= 0 { "+" } else { "-" };
    let abs_mib = mib.unsigned_abs();
    format!("{sign}{}", format_mib(abs_mib))
}

#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use candle_core::Device;
#[cfg(feature = "cuda")]
use voxui_inference::{SynthesisRequest, VoxCPMEngine};

#[cfg(feature = "cuda")]
const TEST_DIT_STEPS: usize = 10;
#[cfg(feature = "cuda")]
const CHINESE_20: &str = "\u{8fd9}\u{662f}\u{7528}\u{4e8e}\u{6d4b}\u{8bd5}\u{663e}\u{5b58}\u{62a5}\u{544a}\u{7684}\u{4e8c}\u{5341}\u{4e2a}\u{4e2d}\u{6587}\u{6c49}\u{5b57}\u{8f93}\u{5165}";

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn model_dir(name: &str) -> PathBuf {
    repo_root().join("models").join(name)
}

#[cfg(feature = "cuda")]
fn artifact_dir() -> PathBuf {
    repo_root().join("voxui").join("target").join("cuda-vram-report")
}

#[cfg(feature = "cuda")]
fn print_sample_line(label: &str, baseline: MemorySample, sample: MemorySample) {
    match sample.process {
        Some(process) => {
            let delta = process_delta_mib(baseline.process, sample.process)
                .map(format_signed_mib)
                .unwrap_or_else(|| "delta unavailable".to_string());
            println!("{label:<22} {:>22}  {delta}", format_mib(process.used_mib));
        }
        None => println!("{label:<22} process VRAM unavailable"),
    }

    if let Some(global) = sample.global {
        println!(
            "{label:<22} global free/total {:>22} / {}",
            format_mib(global.free_mib),
            format_mib(global.total_mib)
        );
    }
}

#[cfg(feature = "cuda")]
fn print_model_report(
    model_name: &str,
    baseline: MemorySample,
    after_load: MemorySample,
    after_synth: MemorySample,
) {
    println!();
    println!("=== CUDA VRAM report: {model_name} ===");
    match baseline.process {
        Some(process) => println!(
            "{:<22} {:>22}",
            "baseline process:",
            format_mib(process.used_mib)
        ),
        None => println!("{:<22} process VRAM unavailable", "baseline process:"),
    }
    if let Some(global) = baseline.global {
        println!(
            "{:<22} global free/total {:>22} / {}",
            "baseline:",
            format_mib(global.free_mib),
            format_mib(global.total_mib)
        );
    }

    print_sample_line("after load:", baseline, after_load);
    print_sample_line("after synth:", baseline, after_synth);

    match peak_process_delta_mib(baseline.process, after_load.process, after_synth.process) {
        Some(peak) => {
            let status = if peak > VRAM_WARNING_THRESHOLD_MIB as i64 {
                "WARNING"
            } else {
                "OK"
            };
            println!(
                "{:<22} {:>22}  threshold {}  {status}",
                "peak process delta:",
                format_signed_mib(peak),
                format_mib(VRAM_WARNING_THRESHOLD_MIB)
            );
            if peak > VRAM_WARNING_THRESHOLD_MIB as i64 {
                println!(
                    "WARNING: peak process VRAM delta exceeded 7 GiB; investigate runtime quantization or more aggressive export/inference strategy."
                );
            }
        }
        None => println!(
            "{:<22} process VRAM unavailable; threshold warning skipped",
            "peak process delta:"
        ),
    }
}

#[cfg(feature = "cuda")]
fn run_model_report(model_name: &str) -> Result<Option<ModelVramReport>> {
    let dir = model_dir(model_name);
    if !dir.join("model.gguf").is_file() {
        eprintln!(
            "[SKIP] {model_name}: model.gguf not found at {}",
            dir.display()
        );
        return Ok(None);
    }

    let device = Device::new_cuda(0).context("create CUDA device 0")?;
    device
        .synchronize()
        .context("synchronize CUDA device before baseline")?;
    let baseline = take_memory_sample();

    let mut engine = VoxCPMEngine::load(&dir, device.clone())
        .with_context(|| format!("load {model_name} on CUDA"))?;
    device
        .synchronize()
        .context("synchronize CUDA device after load")?;
    let after_load = take_memory_sample();

    let request = SynthesisRequest {
        text: CHINESE_20.to_string(),
        inference_timesteps: TEST_DIT_STEPS,
        retry_badcase: false,
        ..SynthesisRequest::default()
    };
    let samples = engine
        .generate(request, |_, _| {})
        .with_context(|| format!("synthesize 20-character Chinese sentence with {model_name}"))?;
    device
        .synchronize()
        .context("synchronize CUDA device after synthesis")?;
    let after_synth = take_memory_sample();

    assert!(
        !samples.is_empty(),
        "generate returned empty audio for {model_name}"
    );
    assert!(
        samples.iter().all(|sample| sample.is_finite()),
        "generate produced NaN/Inf for {model_name}"
    );

    print_model_report(model_name, baseline, after_load, after_synth);
    Ok(Some(ModelVramReport {
        model_name: model_name.to_string(),
        baseline,
        after_load,
        after_synth,
        peak_process_delta_mib: peak_process_delta_mib(
            baseline.process,
            after_load.process,
            after_synth.process,
        ),
    }))
}

#[test]
#[cfg(feature = "cuda")]
fn vram_report_child() -> Result<()> {
    let Some(model_name) = std::env::var(CHILD_MODEL_ENV).ok() else {
        eprintln!("[SKIP] child env not set");
        return Ok(());
    };
    let Some(report) = run_model_report(&model_name)? else {
        println!("{CHILD_JSON_PREFIX}{{\"model_name\":\"{model_name}\",\"skipped\":true}}");
        return Ok(());
    };
    println!(
        "{CHILD_JSON_PREFIX}{}",
        serde_json::json!({
            "model_name": report.model_name,
            "peak_process_delta_mib": report.peak_process_delta_mib,
            "baseline_process_mib": report.baseline.process.map(|v| v.used_mib),
            "after_load_process_mib": report.after_load.process.map(|v| v.used_mib),
            "after_synth_process_mib": report.after_synth.process.map(|v| v.used_mib),
        })
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_child_model_report(model_name: &str) -> Result<serde_json::Value> {
    let output = Command::new(std::env::current_exe()?)
        .args(["--exact", "vram_report_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_MODEL_ENV, model_name)
        .output()
        .with_context(|| format!("run child VRAM report for {model_name}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        anyhow::bail!("child VRAM report failed for {model_name}\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }
    let json_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(CHILD_JSON_PREFIX))
        .ok_or_else(|| anyhow::anyhow!("child VRAM report did not emit JSON for {model_name}"))?;
    serde_json::from_str(json_line).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn write_report_artifacts(reports: &[serde_json::Value]) -> Result<()> {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir)?;
    let json_path = dir.join("voxcpm-vram-report.json");
    let md_path = dir.join("voxcpm-vram-report.md");
    std::fs::write(&json_path, serde_json::to_string_pretty(reports)?)?;

    let mut markdown = String::from("# VoxCPM CUDA VRAM Report\n\n");
    markdown.push_str("| model | peak process delta MiB | skipped |\n");
    markdown.push_str("| --- | ---: | --- |\n");
    for report in reports {
        markdown.push_str(&format!(
            "| {} | {} | {} |\n",
            report["model_name"].as_str().unwrap_or("<unknown>"),
            report["peak_process_delta_mib"]
                .as_i64()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            report["skipped"].as_bool().unwrap_or(false),
        ));
    }
    markdown.push_str("\n`cargo test` captures stdout by default. Use `-- --nocapture --test-threads=1` for console output.\n");
    std::fs::write(&md_path, markdown)?;
    println!(
        "wrote VRAM artifacts: {} and {}",
        json_path.display(),
        md_path.display()
    );
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn reports_voxcpm2_cuda_vram_for_fp16_and_q4_lm() -> Result<()> {
    assert_eq!(CHINESE_20.chars().count(), 20);
    let fp16 = run_child_model_report("voxcpm2-fp16")?;
    let q4 = run_child_model_report("voxcpm2-q4-lm")?;
    let reports = vec![fp16.clone(), q4.clone()];
    write_report_artifacts(&reports)?;

    if fp16["skipped"].as_bool().unwrap_or(false) || q4["skipped"].as_bool().unwrap_or(false) {
        eprintln!("[SKIP] one or more model bundles are missing");
        return Ok(());
    }
    if let (Some(fp16_peak), Some(q4_peak)) = (
        fp16["peak_process_delta_mib"].as_i64(),
        q4["peak_process_delta_mib"].as_i64(),
    ) {
        assert!(
            q4_peak < fp16_peak,
            "expected q4 peak process VRAM ({q4_peak} MiB) to be below fp16 ({fp16_peak} MiB)"
        );
    }

    Ok(())
}

#[test]
#[cfg(not(feature = "cuda"))]
fn reports_voxcpm2_cuda_vram_for_fp16_and_q4_lm() {
    eprintln!("[SKIP] CUDA feature not enabled");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_memory_parser_filters_current_pid_and_sums_rows() {
        let output = "1234, 2048\n9999, 333\n1234, 256\nbad, data\n";

        let snapshot = parse_process_memory_mib(output, 1234);

        assert_eq!(snapshot, Some(ProcessMemorySnapshot { used_mib: 2304 }));
    }

    #[test]
    fn process_memory_parser_accepts_mib_suffix_when_present() {
        let output = "42, 7168 MiB\n";

        let snapshot = parse_process_memory_mib(output, 42);

        assert_eq!(snapshot, Some(ProcessMemorySnapshot { used_mib: 7168 }));
    }

    #[test]
    fn process_memory_parser_returns_none_without_matching_pid() {
        let output = "11, 1024\n22, 2048\n";

        let snapshot = parse_process_memory_mib(output, 33);

        assert_eq!(snapshot, None);
    }

    #[test]
    fn windows_dedicated_usage_parser_converts_byte_sum_to_mib() {
        let output = "7516192768\r\n";

        let snapshot = parse_windows_dedicated_usage_bytes(output);

        assert_eq!(snapshot, Some(ProcessMemorySnapshot { used_mib: 7168 }));
    }

    #[test]
    fn memory_formatting_is_stable_for_report_output() {
        assert_eq!(format_mib(0), "0 MiB (0.00 GiB)");
        assert_eq!(format_mib(7168), "7168 MiB (7.00 GiB)");
        assert_eq!(format_signed_mib(256), "+256 MiB (0.25 GiB)");
        assert_eq!(format_signed_mib(-128), "-128 MiB (0.12 GiB)");
    }

    #[test]
    fn peak_delta_uses_largest_available_process_delta() {
        let baseline = ProcessMemorySnapshot { used_mib: 1000 };
        let after_load = ProcessMemorySnapshot { used_mib: 6500 };
        let after_synth = ProcessMemorySnapshot { used_mib: 6200 };

        let peak = peak_process_delta_mib(Some(baseline), Some(after_load), Some(after_synth));

        assert_eq!(peak, Some(5500));
    }
}
