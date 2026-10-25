# RGB-D capture session format (iOS app, Android app → rgbd_ingest.py)

One capture = one session folder (zipped for transfer: `session_<ts>.zip`), or the same
records streamed live over a WebSocket.

```
session_20261024-1310/
  session.json            # {"device": "iphone"|"android", "model": "...", "app_version": "...",
                          #  "started": ISO8601, "coord": "arkit_y_up"|"arcore_y_up",
                          #  "depth_units": "meters", "has_person_mask": true}
  frames/
    000000.jpg            # RGB, landscape, downscaled to <= 960 px wide, JPEG q85
    000000_depth.png      # 16-bit PNG, depth in millimetres (0 = invalid), depth-map resolution
    000000_conf.png       # 8-bit PNG, confidence 0/1/2 (low/med/high), depth resolution (optional)
    000000_mask.png       # 8-bit PNG, 255 = person, depth resolution (optional)
    000000.json           # per-frame metadata, see below
```

Per-frame JSON:
```json
{
  "index": 0,
  "timestamp": 12.345,                    // seconds since session start
  "pose": [[r00,r01,r02,tx],[...],[...],[0,0,0,1]],   // camera-to-world 4x4, row-major
  "intrinsics_rgb":   {"fx":..,"fy":..,"cx":..,"cy":..,"width":..,"height":..},
  "intrinsics_depth": {"fx":..,"fy":..,"cx":..,"cy":..,"width":..,"height":..},
  "tracking": "normal" | "limited" | "not_available"
}
```

Conventions:
- `pose` is the platform's native camera transform (ARKit `frame.camera.transform`,
  ARCore `Camera.getPose()`): world is gravity-aligned Y-up; camera looks down its -Z,
  +Y up, +X right (OpenGL convention). Ingest converts to the Z-up map frame.
- `intrinsics_depth` = RGB intrinsics scaled to the depth resolution.
- Keyframe policy: save when the camera moved >= 0.15 m or rotated >= 10 deg since the
  last keyframe, at most 3 Hz, and only when tracking == normal.

Live streaming (WebSocket `ws://<mac-ip>:8770/ingest`):
- First message: text, the `session.json` object.
- Then per keyframe: one text message (the per-frame JSON with an extra `"files"` list
  naming which binaries follow, in order), followed by binary messages for each file
  in that order (jpg, depth png, conf png, mask png).
- Last message: text `{"end": true}`.

## RoomPlan session (iOS only)

Made by the iOS app's "Room scan (RoomPlan)" mode. It is never streamed live and is
transferred as a zip only (`session_<ts>_roomplan.zip`, with the folder at the top level of the zip).
An RGB-D session's `session.json` has no `"kind"` key. Treat a missing `"kind"` as RGB-D.

```
session_20261024-1310_roomplan/      # "_2", "_3"... appended on name collision
  session.json
  roomplan.json          # JSONEncoder().encode(CapturedRoom) of the first (or only) room
  roomplan.usdz          # CapturedRoom.export(to:exportOptions: .parametric)
  rooms/room_00.json     # only when >1 room was scanned: every room, in scan order
  rooms/room_00.usdz
  structure.json         # only when >1 room AND StructureBuilder succeeded:
  structure.usdz         #   JSONEncoder().encode(CapturedStructure) / .export(.parametric)
```

`session.json`:
```json
{"device": "iphone", "kind": "roomplan", "model": "iPhone16,1", "app_version": "1.0",
 "started": ISO8601, "coord": "arkit_y_up", "units": "meters",
 "room_count": 1, "has_structure": false,
 "files": ["roomplan.json", "roomplan.usdz", ...],   // relative paths actually written
 "warnings": ["..."]}                                  // optional, e.g. a failed usdz export
```

Conventions:
- Each room is built with `RoomBuilder(options: [.beautifyObjects])`. The structure is built with
  `StructureBuilder(options: [.beautifyObjects])`.
- Rooms in one session share one ARKit world frame. The ARSession is kept running between
  rooms, so world is gravity-aligned Y-up and units are metres. Each wall/door/window/opening/object has
  `dimensions` (x = width, y = height, z = depth) and a 4x4 `transform` (local to world). The
  JSON keys are Apple's Codable encoding of `CapturedRoom`, which can vary between iOS versions,
  so don't hard-code more than `walls`, `doors`, `windows`, `openings`, `objects`,
  `floors` (iOS 17+), and per item `dimensions`, `transform`, `category`, `identifier`.
- The USDZ files are for viewing (Quick Look / Blender). Ingest should prefer the JSON.
- `transform` in RoomPlan's JSON is expected to be a flat array of 16 floats in column-major
  order (simd layout: elements 12-14 = translation). This hasn't been checked on a device yet, so
  confirm it on the first real file.

## Android extensions (Android app only)

The Android app (`android_app/`) writes the RGB-D session above with `"device": "android"` and
`"coord": "arcore_y_up"`. `pose` is ARCore `Camera.getPose()` (physical sensor camera, not
display-oriented). It uses the ARKit convention: gravity-aligned Y-up world, camera looks down -Z, +Y up,
+X right in the landscape image. `Pose.toMatrix()` is column-major, and the app transposes it to the
row-major `pose` above. Notes:

- Depth comes from `acquireRawDepthImage16Bits()` (fallback `acquireDepthImage16Bits()`): uint16 mm, 0 = invalid.
  ARCore's 0-255 raw confidence is bucketed into `_conf.png` 0/1/2 by thirds (<85, <170, else).
- The RGB JPEG is center-cropped to the depth map's aspect ratio (usually a no-op) before downscaling,
  so it covers the same field of view as the depth map. `intrinsics_rgb` is `Camera.getImageIntrinsics()`
  adjusted for that crop and scale. `intrinsics_depth` is `Camera.getTextureIntrinsics()` cropped and scaled
  to the depth resolution (the depth image matches the GPU texture's field of view). The app picks
  a camera config whose CPU image has the same aspect ratio as the texture, so both describe the same view.
- `_mask.png` comes from ML Kit Selfie Segmentation (255 = person). It is optional, behind a toggle, and is
  reported by `has_person_mask`.
- `session.json` may carry `"has_objects": true` when on-device Gemma labelling was on.

On-device object labels (optional, behind a toggle): roughly every 1 s, Gemma (LiteRT-LM, `.litertlm`
model on the phone) labels the newest keyframe with the prompt
`List notable objects as JSON [{label, box_2d:[ymin,xmin,ymax,xmax] 0-1000, category in structure|movable_furniture|clutter|person}]`.
Unknown categories are mapped to `clutter`. Results are written next to the frame:

```
frames/000012_objects.json   # {"index": 12, "objects": [{"label", "box_2d", "category"}],
                             #  "backend": "npu"|"gpu"|"cpu"|..., "latency_ms": 840,
                             #  "raw": "..."}   (raw model reply, only when nothing parsed)
```

Batch ingest skips these files: `NNNNNN_objects.json` has no matching `.jpg`, so `iter_session` warns and moves on.

Live stream: when "Send object messages on the live stream" is on (off by default), the app also sends the extra
text message `{"objects": [...], "index": N}` at any point after keyframe N. It has no `"files"` and
no binaries follow it. **The current `rgbd_ingest.py --serve` reads every non-`end` text message as a
frame header, so this message breaks the session.** Keep the toggle off until the ingester skips such
messages, e.g. in `handle_ingest` before `idx = int(meta["index"])`:
`if "objects" in meta: save it as frames/NNNNNN_objects.json; continue`.
