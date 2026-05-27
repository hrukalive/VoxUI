use anyhow::Result;
use voxui_inference_sidecar::SidecarEngine;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    SidecarEngine::default().run(stdin.lock(), stdout.lock())
}
