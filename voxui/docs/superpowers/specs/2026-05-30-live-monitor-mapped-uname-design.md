# Live Monitor Mapped Username Editing Design

## Context

The Bilibili live monitor shows feed rows built from `LiveMonitorItem`, which already includes `open_id`, `uname`, and `mapped_uname`. Streamers can edit username mappings in Settings, but doing that during a stream requires opening Settings and finding the matching entry.

Mappings are stored in `LiveConfig.mapped_unames`, while `LiveConfig.original_unames` records the username observed when the record was initialized. Live config changes already persist through `LiveConfigPatch` and return a refreshed `LiveSnapshot`, which re-renders feed rows.

## Requirements

- Add an SVG icon button to every live monitor feed item.
- The button opens a modal for that item.
- The modal shows readonly labels for `open_id` and the current `uname`.
- The modal contains one editable `mapped_uname` input.
- The input is initialized from `mapped_unames[open_id]` when present, otherwise from the current `uname`.
- Saving is equivalent to editing Settings: it updates `mapped_unames`, persists immediately through the existing live config patch pipeline, and lets the backend re-evaluate feed rows.
- The per-item button uses `live-monitor-button` styling when the mapping is configured and the viewer has not changed their username since the record was initialized.
- The per-item button uses `primary-button` styling when the mapping is missing or the viewer's current `uname` differs from `original_unames[open_id]`.

## Chosen Approach

Add the feature directly to `LiveMonitor`.

Each row will keep its existing send and replacement actions, plus a compact mapped-name edit button. Clicking the edit button stores the selected `LiveMonitorItem` in component state and opens a local modal. This keeps the workflow in the live monitor window and avoids forcing streamers back into Settings during active monitoring.

Settings remains the bulk editor for mappings. The monitor modal is a focused shortcut for one viewer at a time and writes to the same `mapped_unames` config field.

## Data Flow

1. A feed item button is clicked.
2. `LiveMonitor` opens a modal with the selected `open_id`, `uname`, and initial mapped name.
3. On Save, the component clones `snapshot.config.mapped_unames`, inserts or updates the selected `open_id`, and calls `on_live_patch`.
4. The existing frontend optimistic patch updates config immediately.
5. The Tauri command persists the live config and returns a refreshed `LiveSnapshot`.
6. The refreshed snapshot re-renders feed rows with updated `mapped_uname` and regenerated suggestions.

## Button State

The attention state is derived from the current snapshot config:

- Configured and unchanged: `mapped_unames` contains `open_id`, and `original_unames[open_id] == item.uname`.
- Needs attention: `mapped_unames` does not contain `open_id`, or `original_unames[open_id] != item.uname`.

Missing `original_unames` is treated as needing attention because the monitor cannot prove the mapping matches the current username.

## Error Handling

The existing `on_live_patch` path currently does optimistic frontend updates and accepts the backend snapshot when the command succeeds. This feature will use that same behavior. The modal can close after Save because the requested behavior is immediate persistence through the current pipeline.

## Testing

Add focused Rust unit tests in the desktop frontend module for helper logic:

- Initial mapped-name value uses existing mapped config when present.
- Initial mapped-name value falls back to current `uname` when missing.
- Button attention state is false only when mapping exists and original username matches current username.
- Button attention state is true when mapping is missing.
- Button attention state is true when the current username changed after the mapping record was initialized.

Keep existing tests that check monitor buttons use SVG icons and rendered feed keys track mapped name changes.
