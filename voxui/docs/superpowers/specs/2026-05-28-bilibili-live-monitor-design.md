# Bilibili Live Monitor Design

Date: 2026-05-28

## Summary

VoxUI should add Bilibili OpenLive support for streamers who want live chat and paid interaction messages turned into ready-to-send TTS input. The feature adds a Bilibili-only `Live` settings tab and a separate native Tauri live monitor window inside `voxui-desktop`.

The backend owns the OpenLive connection lifecycle, signing, websocket protocol, heartbeats, event parsing, raw JSON retention, and communication with Bilibili. The frontend owns compact display, settings editing, and replacing the main input draft with a selected suggested text.

The monitor window opens only after a successful connection and successful Bilibili websocket auth. Closing the monitor window ends the OpenLive app session through the appropriate Bilibili `/v2/app/end` call. Exiting the whole desktop app performs the same cleanup.

## Goals

- Connect to Bilibili OpenLive from `voxui-desktop`.
- Use the same connection sequence demonstrated by `D:\Dev\py-demo-new\ws2.py`.
- Add a separate native Tauri monitor window after successful auth.
- Show suggested TTS input text for supported live message types.
- Store raw JSON for each monitor item so existing items can be recomputed when settings change.
- Replace the main input box text when a monitor item send button is clicked.
- Add a clear button in the main composer.
- Keep monitor controls compact, using icon-style buttons where practical.
- Make username display configurable through `open_id -> mapped_uname`.
- Make message-type visibility and templates configurable in a new `Live` settings tab.
- Make danmu pronoun replacement rules configurable, enabled by default, and toggleable.

## Non-Goals

- OBS integration is not part of this feature.
- Other live platforms are not part of this feature.
- Automatic TTS generation from live messages is not part of the first implementation.
- Moderation, spam scoring, or AI rewriting of suggestions is not part of this design.
- Persisting monitor feed history across app launches is not required.

## Current State

The app is a Tauri 2 desktop app with a Leptos frontend:

- `crates/voxui-desktop/src-tauri/src/lib.rs` registers commands and owns Tauri setup.
- `crates/voxui-desktop/src-tauri/src/commands.rs` contains command handlers, background tasks, and frontend event emission.
- `crates/voxui-desktop/src-tauri/src/types.rs` defines persisted config and DTOs.
- `crates/voxui-desktop/src-tauri/src/config.rs` reads and writes `voxui_config.json`.
- `crates/voxui-desktop/src/app.rs` owns root frontend state and long-lived Tauri listeners.
- `crates/voxui-desktop/src/components/settings_modal.rs` contains settings pages.
- `crates/voxui-desktop/src/components/input_box.rs` owns the main text draft and generate button.

Tauri 2 supports multiple windows in `tauri.conf.json` and backend-to-frontend events. The existing app already uses Tauri commands/events, so the live monitor should follow the same shape rather than introducing a separate UI framework.

## Architecture

### Window Layout

The main window remains the primary TTS app. A second native Tauri window, labelled `live-monitor`, is used for the live feed.

The `live-monitor` window should not be created on startup. It should be created or shown only after:

1. the signing server accepts the OpenLive start request body and returns valid Bilibili signing headers;
2. Bilibili `/v2/app/start` succeeds;
3. the websocket connects;
4. websocket auth packet `op=7` succeeds.

If any step fails, the monitor window is not opened and the main UI receives a connection error.

### Backend Modules

Add a focused backend module, for example `src-tauri/src/openblive.rs`, responsible for:

- OpenLive API request body serialization;
- calling the configured signing endpoints;
- signed `/v2/app/start`, `/v2/app/heartbeat`, and `/v2/app/end`;
- websocket connection and binary packet encoding/decoding;
- websocket auth and heartbeat packets;
- parsing `op=5` JSON callbacks;
- emitting raw live events to app state;
- stopping cleanly on monitor close or app exit.

Add a live state module or section in `AppCore` for:

- current connection status;
- current game id;
- live monitor items;
- configured message filters and templates;
- configured identity code;
- username mappings;
- replacement rules.

The backend remains authoritative for connection status and raw events. The frontend can render and request actions, but it should not own the Bilibili connection.

### Tauri Commands and Events

Commands:

- `connect_openblive(identity_code)` starts the connection flow and opens `live-monitor` after successful auth.
- `disconnect_openblive()` ends the OpenLive session and closes or marks the monitor disconnected.
- `get_live_state()` returns live settings, status, and current recomputed items.
- `set_live_config_patch(patch)` updates live settings and recomputes monitor items.
- `send_live_suggestion(item_id, mode)` emits a main-input replacement event for the chosen item and mode.
- `clear_live_items()` clears the current in-memory monitor feed.

Events:

- `live_status_changed` updates both windows with connecting, connected, disconnecting, disconnected, or error status.
- `live_items_changed` tells the monitor to refresh its visible list.
- `main_input_replace` tells the main window input box to replace its draft text.

The main input text remains frontend state in `InputBox`, so replacing it should be done with a Tauri event listener in the main window root or in `InputBox`.

## OpenLive Connection Lifecycle

The implementation should mirror `ws2.py`:

1. Build compact JSON request bodies with `serde_json` in a stable form equivalent to Python `separators=(",", ":")`.
2. POST the exact body bytes to signing endpoints.
3. Use the returned signed headers for Bilibili OpenLive API calls.
4. POST `/v2/app/start` with identity code and app id.
5. Store `game_id`.
6. Connect to the returned websocket URL with native ping disabled.
7. Send auth packet `op=7` with `auth_body`.
8. Confirm auth JSON response has `code == 0`.
9. Open/show the live monitor window.
10. Run websocket heartbeat loop using packet `op=2`.
11. Run signed OpenLive app heartbeat loop with `/v2/app/heartbeat`.
12. Optionally retain the ceve-market local heartbeat behavior if needed by the signing service, but keep it internal and non-blocking.
13. On monitor window close, disconnect command, or whole app shutdown, send `/v2/app/end` when `game_id` exists.

Cleanup should be idempotent. Multiple close paths must not panic or send duplicate state transitions.

## Supported Message Types

Each stored monitor item contains:

- generated app item id;
- received timestamp;
- normalized message kind;
- raw JSON as `serde_json::Value`;
- source `cmd`;
- source `open_id`;
- source `uname`;
- current visible/suggested text derived from settings.

Existing items should be recomputed when filters, templates, username mappings, language, or replacement rules change.

### Danmu

Command: `LIVE_OPEN_PLATFORM_DM`.

Rules:

- Skip when `data.dm_type == 1`.
- Start from `data.msg`.
- Strip bracket emotes: any substring enclosed by `[` and `]`.
- Collapse one or more whitespace characters into a single period.
- Use English `.` for English UI mode and Chinese `。` for Chinese UI mode.
- The normal send button uses the cleaned message.
- The special replacement-rule send button applies enabled replacement rules before replacing the main input.

### Paid Gifts

Command: `LIVE_OPEN_PLATFORM_SEND_GIFT`.

Rules:

- Include only when `data.paid == true`.
- Mark clearly as paid in the monitor.
- Suggested Chinese default: `感谢{mapped_uname}送出的{gift_num}个{gift_name}`.
- Suggested English default: `Thank you {mapped_uname} for {gift_num} {gift_name}`.

### Superchats

Command: `LIVE_OPEN_PLATFORM_SUPER_CHAT`.

Rules:

- Mark clearly as paid in the monitor.
- Suggested Chinese default: `感谢{mapped_uname}的醒目留言：{message}`.
- Suggested English default: `Thank you {mapped_uname} for the superchat saying {message}`.

### Guard Purchases

Command: `LIVE_OPEN_PLATFORM_GUARD`.

Rules:

- Mark clearly as paid in the monitor.
- Source user is `data.user_info`.
- Guard level labels:
  - `1`: `总督`
  - `2`: `提督`
  - `3`: `舰长`
  - any other value: `航海`
- Suggested Chinese default: `感谢{mapped_uname}开通的{guard_label}`.
- Suggested English default: `Thank you {mapped_uname} for joining as {guard_label}`.

### Likes

Command: `LIVE_OPEN_PLATFORM_LIKE`.

Rules:

- Supported but hidden by default.
- Suggested Chinese default: `感谢{mapped_uname}给直播间点赞`.
- Suggested English default: `Thank you {mapped_uname} for liking the stream`.

### Viewer Enters

Command: `LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER`.

Rules:

- Suggested Chinese default: `欢迎{mapped_uname}进入直播间`.
- Suggested English default: `Hi {mapped_uname}, welcome to the stream`.

## Live Settings Tab

Add `SettingsPage::Live` and a `Live` tab in the existing settings modal.

Fields:

- identity code text input labelled `身份码` in Chinese and `Identity code` in English;
- connect/disconnect button with status text;
- message-type checkboxes for danmu, paid gifts, superchats, guard purchases, likes, and viewer enters;
- template editor for each message type;
- replacement-rule editor for danmu switch mode;
- username mapping editor for `open_id -> mapped_uname`.

Default filters:

- danmu: enabled;
- paid gifts: enabled;
- superchats: enabled;
- guard purchases: enabled;
- likes: disabled;
- viewer enters: enabled.

Template placeholders should be simple named tokens such as `{mapped_uname}`, `{msg}`, `{gift_num}`, `{gift_name}`, `{message}`, and `{guard_label}`. Unknown placeholders are left unchanged so templates remain debuggable.

### Username Mapping

When a raw event arrives, initialize missing mappings from `open_id -> uname`. The streamer can edit the mapped name later.

The settings UI should show mappings in a compact table with:

- source `open_id`;
- original `uname`;
- editable `mapped_uname`.

Changing a mapped name recomputes all existing monitor item suggestions that reference that `open_id`.

### Replacement Rules

Replacement rules are ordered pairs with an enabled checkbox.

Defaults:

- `我的` -> `你的`
- `我` -> `你`
- `I` -> `you`
- `me` -> `you`
- `my` -> `your`

The first implementation can apply rules as literal string replacements in order. Longer Chinese phrases should appear before shorter ones to avoid partial replacement surprises. Rules are applied only by the danmu special switch-send button, not by normal danmu send.

## Live Monitor Window UI

The monitor window should be compact and scanner-friendly:

- status strip at the top;
- scrollable feed filling the window;
- newest items appended at the bottom;
- auto-scroll to bottom when already near the bottom;
- short smooth scroll animation for auto-scroll to reduce accidental clicks during sudden list movement;
- clear visual labels for danmu, gift, superchat, guard, like, and enter;
- paid message kinds should be visually distinct and labelled as paid;
- icon-first compact buttons.

Each item should show:

- message type;
- mapped username when available;
- concise source details such as gift count/name or superchat amount;
- suggested text preview;
- compact send button;
- for danmu, compact switch-send button.

Manual scrolling away from the bottom should pause auto-scroll until the streamer returns near the bottom.

## Main Composer Changes

`InputBox` should expose a way to replace its internal draft from a root event or prop. A `main_input_replace` event from the backend is the preferred path because monitor and main window are separate webviews.

Add a clear button below or beside the generate button. It clears the local draft and is disabled when the draft is empty or input is disabled.

Generate behavior remains unchanged: it enqueues the current trimmed draft and clears it after successful submit.

## Error Handling

Connection failures:

- show a connection error in the main window;
- do not open the monitor window;
- stop any partially started websocket or heartbeat task;
- call `/v2/app/end` only if a `game_id` was obtained.

Websocket closes or auth fails:

- update status to disconnected or error;
- stop heartbeat loops;
- call `/v2/app/end` if connected enough to have `game_id`;
- keep current monitor items in memory until cleared or app exits.

Monitor window close:

- request disconnect;
- stop websocket and heartbeat loops;
- call `/v2/app/end`;
- mark live status disconnected.

Whole app exit:

- perform the same disconnect path.

Malformed messages:

- log at debug/warn level;
- ignore unsupported or malformed event payloads;
- do not crash the monitor.

## Testing Strategy

### Unit Tests

- Parse every supported sample JSON command into the correct internal kind.
- Skip emote-only danmu.
- Strip bracket emotes from danmu.
- Collapse spaces to English or Chinese periods.
- Generate paid gift suggestions only when `paid == true`.
- Map guard levels to `总督`, `提督`, `舰长`, or `航海`.
- Initialize missing username mappings from event `open_id` and `uname`.
- Recompute suggestions after username mapping changes.
- Apply enabled replacement rules in order.
- Ignore disabled replacement rules.
- Render templates with known placeholders and leave unknown placeholders unchanged.
- Filter monitor items by enabled message types, with likes disabled by default.

### Backend Lifecycle Tests

- Connection state transitions from disconnected to connecting to connected.
- Failed signing or failed Bilibili auth does not open the monitor.
- Disconnect cleanup is idempotent.
- Closing the monitor triggers OpenLive app end when `game_id` exists.
- App shutdown triggers the same cleanup path.

### Frontend Tests or Manual Verification

- Live tab appears in settings and persists changes.
- Connect button opens monitor only after success.
- Monitor appends items and auto-scrolls only when near bottom.
- Item send replaces main input draft.
- Danmu switch-send replaces main input draft with transformed text.
- Clear button clears the main draft.

## Implementation Notes

- Keep all new work inside `crates/voxui-desktop` unless a small protocol/helper module is clearly reusable.
- Add backend dependencies only as needed for websocket and HTTP support.
- Preserve the existing app config file by using serde defaults for new live settings.
- Avoid touching unrelated inference or audio crates.
- Keep raw JSON in memory for the monitor feed; persisted config should contain settings and mappings, not feed history.
- The existing modified `crates/voxui-desktop/src-tauri/Cargo.toml` in the working tree should be reviewed before dependency edits so user changes are not overwritten.
