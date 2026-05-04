use voxui_audio::{AudioSystem, AudioPlayer};

fn main() {
    let system = AudioSystem::new();

    println!("=== Audio Hosts & Devices ===");
    for host in system.hosts() {
        println!("Host: {}", host.name);
        if let Ok(devices) = system.devices(&host.name) {
            for dev in &devices {
                println!("  Device: {}", dev.name);
            }
        }
    }

    let host = system.default_host_name();
    let device = system.default_device_name(&host).unwrap();

    let sample_rate: u32 = 48000;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| {
            (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.3
        })
        .collect();

    let player = AudioPlayer::new(&host, &device, sample_rate).unwrap();
    println!("\nPlaying 440Hz sine on {} / {}", host, device);
    player.play_blocking(samples).unwrap();
    println!("Done!");
}
