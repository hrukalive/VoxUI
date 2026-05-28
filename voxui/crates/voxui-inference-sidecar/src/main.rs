use anyhow::Result;
use voxui_inference_sidecar::SidecarEngine;

fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "info,voxui_inference_sidecar=debug,voxui_inference=info",
        )
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
    tracing::info!("voxui inference sidecar starting");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    SidecarEngine::default().run(stdin, stdout.lock())
}
