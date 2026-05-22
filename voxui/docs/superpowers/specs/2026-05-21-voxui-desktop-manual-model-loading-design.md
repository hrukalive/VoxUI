# VoxUI Desktop Manual Model Loading Design

## Context

`voxui-inference` now uses a single canonical base GGUF file per model directory (`model.gguf`) and direct LoRA GGUF files in the same directory. `voxui-desktop` currently discovers model directories, discovers LoRA files separately, and automatically loads a model during startup or model selection. The new UI must make model loading explicit and make LoRA selection part of the model dropdown.

The product title changes to `焓言焓语` in Chinese and `AhanSays` in English.

## Goals

- Do not auto-load a model on startup.
- Default the model root to the program folder's `models` directory.
- Allow the model root folder to be changed from Settings with a folder browser.
- Show a flattened model dropdown next to the title.
- Duplicate dropdown entries for each direct LoRA GGUF file in a model directory.
- Keep selected model state separate from loaded engine state.
- Add explicit `Load` and cancellable `Cancel` behavior.
- Show determinate byte-read progress before switching to an indeterminate device-loading progress bar.

## Model Discovery

The backend scans the configured model root. A valid model directory contains `model.gguf`. A valid LoRA file is a direct `.gguf` file in that same model directory whose filename starts with `lora_`.

For each valid model directory, discovery emits:

- One base choice, displayed as the model directory name.
- One LoRA choice per valid LoRA GGUF file, displayed as `<model name> | <lora name>`.

For example, a directory named `voxcpm2-q4-lm` containing `model.gguf` and `lora_ft2.gguf` produces:

- `voxcpm2-q4-lm`
- `voxcpm2-q4-lm | lora_ft2`

Each choice carries a stable id, display name, model directory, model GGUF path, optional LoRA GGUF path, and known file sizes. The LoRA display name is the file stem, so `lora_ft2.gguf` displays as `lora_ft2`.

## Startup Behavior

Startup loads config, scans the configured model root, restores the last selected model choice if it still exists, and leaves the engine unloaded. If the saved selection no longer exists, the app selects the first available choice and remains unloaded.

The default model root is `<program folder>/models`, where `<program folder>` means the directory containing the desktop executable. In development this should resolve from the desktop executable/current Tauri context rather than the repository root heuristic currently used by `discover_models_root`.

## Selection And Loaded State

The UI keeps two separate concepts:

- `selected_model_choice`: the current dropdown value.
- `loaded_model_choice`: the model and optional LoRA currently installed in the engine.

Changing the dropdown does not unload or reload the engine. If a model is already loaded, generation continues to use `loaded_model_choice` until a new load completes.

The `Load` button is enabled only when:

- A model choice is selected.
- No load is in progress.
- No synthesis is in progress.
- The selected choice differs from the loaded choice.

If the user changes away from a loaded choice and then changes back, `Load` is disabled because the selected choice already matches the loaded engine.

## Loading Flow

Loading is a single backend command that receives `model_dir`, optional `lora_path`, and `backend`.

The command builds the new engine without replacing the current engine slot. If loading succeeds, it installs the new engine and updates `loaded_model_choice`. If loading fails or is cancelled, the previous loaded engine remains available.

When `lora_path` is present, the backend loads the base engine and then loads the LoRA into that engine before emitting readiness.

## Progress And Cancellation

Loading progress has two phases:

1. `reading`: determinate progress based on bytes read from `model.gguf`, then the selected LoRA GGUF if present.
2. `device_loading`: indeterminate progress while parsing, building tensors, and moving weights to CPU/CUDA.

During load, the header `Load` button changes to `Cancel`. Pressing `Cancel` signals the backend load cancellation token. Once cancellation resolves, controls unlock. If there was a previously loaded engine, it remains active.

The byte-reading phase should be implemented by reading file sizes and using chunked byte reads to report progress. After bytes have been read, model construction can use the existing GGUF parser path or a staged file abstraction, as long as progress semantics remain correct and cancellation is checked before installing the new engine.

## UI

The header contains the localized title, the flattened model dropdown, the `Load`/`Cancel` button, and the Settings button.

The existing synthesis progress bar remains for generation. A separate model-load progress bar appears near the end of the main content area. During `reading`, it shows a determinate percentage and file label such as `Reading model.gguf`. During `device_loading`, it switches to an indeterminate bar with text such as `Loading to CUDA`.

The status bar should distinguish loaded and selected choices when they differ, for example:

`Loaded: voxcpm2-q4-lm | Selected: voxcpm2-q4-lm | lora_ft2`

## Settings

Settings keeps backend, audio host/device, prompt WAV path, prompt text, reference WAV path, max characters, diffusion steps, and language.

Settings adds a `Models folder` field with a folder browser button. Changing the model root rescans model choices, preserves the previous selected choice if still available, otherwise selects the first available choice. It does not auto-load.

The separate LoRA selector is removed from Settings because LoRA is encoded in the flattened model choice.

Settings are locked while a model load or synthesis is active.

## Config

Config persists:

- Model root folder.
- Last selected model choice id, derived from the model path relative to the model root plus the optional LoRA path relative to the same model directory.
- Backend.
- Audio settings.
- Prompt/reference settings.
- Generation limits.
- Language.

Config may retain legacy `model_dir` and `lora_dir` fields during migration. New UI behavior uses `model_root` and the selected model choice id as the source of truth.

## Error Handling

If no model choices are found, the dropdown is disabled and the status area reports that no models were found in the configured folder. Settings remains available so the user can choose another model root.

If load fails, the selected choice remains selected, `Load` becomes available again, and the existing engine remains active if one was loaded before the failed attempt.

If cancellation succeeds, the UI returns to the same selected and loaded state that existed before the cancelled load.

## Testing

Add or update tests for:

- Default model root resolution.
- Flattened model and LoRA choice scanning.
- Stable model choice ids.
- Config round trip for model root and selected choice.
- Load button state derived from selected, loaded, loading, and generating states.
- Existing desktop command behavior that should remain stable.

Where feasible without full model weights, test that a failed or cancelled load does not replace an already loaded engine. Existing inference tests should continue passing.
