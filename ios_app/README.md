# RGBD Capture (iOS)

Records an RGB-D walkthrough on a LiDAR iPhone/iPad Pro, in the format described in
`../docs/rgbd_format.md`. You can AirDrop the session as a zip or live-stream it to the Mac
over WebSocket (`ws://<mac-ip>:8770/ingest`).

The app has two modes, chosen with the segmented picker at the top of the screen:
- **RGB-D walkthrough**: keyframed color, depth, confidence, and person mask, plus poses. It can live-stream.
- **Room scan (RoomPlan)**: Apple RoomPlan with the standard on-screen coaching. It exports `roomplan.json` and `roomplan.usdz`
  (and a merged `structure.json`/`.usdz` for multi-room scans). It is zip/AirDrop only. The format is in the
  "RoomPlan session" section of `../docs/rgbd_format.md`.

## 1. Open the project

```bash
cd ios_app
open RGBDCapture.xcodeproj
```

If you change `project.yml`, regenerate with `xcodegen generate` (`brew install xcodegen`).

## 2. Signing: pick your Personal Team (one time)

1. In Xcode, sign in under **Xcode > Settings > Accounts** with your Apple ID. A free account works.
2. Click the **RGBDCapture** project in the navigator, select the **RGBDCapture** target, and open **Signing & Capabilities**.
3. Leave **Automatically manage signing** checked. Set **Team** to "Your Name (Personal Team)".
4. If Xcode reports that the bundle id `com.kdotmahesh.rgbdcapture` is taken, change it to something unique, such as `com.<you>.rgbdcapture`.

With a free Personal Team the app expires after 7 days. To renew it, run the app from Xcode again.

## 3. Prepare the iPhone (one time)

1. Connect the iPhone to the Mac with a cable, unlock it, and tap **Trust This Computer**.
2. **Enable Developer Mode:** on the iPhone, open **Settings > Privacy & Security > Developer Mode** and turn it on. The phone restarts. After it restarts, tap **Turn On** and enter your passcode.
   - If Developer Mode isn't in the list, select the phone as the run destination in Xcode once. The option then appears.
3. In Xcode, choose the iPhone as the run destination (top toolbar) and press **Run** (Cmd-R).
4. **Trust the developer certificate:** the first launch fails with "Untrusted Developer". On the iPhone, open **Settings > General > VPN & Device Management**. Under "Developer App", tap your Apple ID, then tap **Trust**. Press Run in Xcode again, or tap the app icon.
5. When the app asks for **Camera** access, allow it. If you turn on Live stream, it also asks for **Local Network** access. Allow that too.

## 4. Record

1. Hold the phone in portrait. Wait until the badge shows **Tracking: Normal** and the depth preview is live.
2. Tap **START** and walk slowly through the house. Point the phone at walls and floor, and avoid fast turns.
3. A keyframe is saved each time you move 0.15 m or turn 10 degrees, at most 3 per second, and only while tracking is Normal. The screen shows **Keyframes**, **Elapsed**, and **Dropped**. Dropped counts frames skipped because the writer or the network was behind.
4. Tap **STOP**.

Sessions are saved to the app's Documents folder as `session_YYYYMMDD-HHMM/`. You can also browse them in the **Files** app under *On My iPhone > RGBD Capture*.

## 4b. Room scan (RoomPlan)

1. Switch the picker to **Room scan (RoomPlan)** and tap **START ROOM SCAN**.
2. Follow the on-screen coaching: **walk the walls first**, then **sweep the middle** of the room,
   and **end where you started**.
3. Tap **Done** to finish. The room is processed with `RoomBuilder(.beautifyObjects)` and saved to
   `session_YYYYMMDD-HHMM_roomplan/`.
4. For several rooms, tap **Next room** instead of Done. Walk into the next room (the session keeps
   tracking) and scan it. Tap **Done** after the last room, and the rooms are merged with `StructureBuilder`.
5. Share it with the same **Share last session** button (see 5a).

RoomPlan needs a LiDAR device. On other devices and in the Simulator the button is disabled and a message explains why.

## 5a. Transfer via AirDrop

Tap **Share last session**. This works for both RGB-D and RoomPlan sessions. The app zips the folder to `session_….zip`; the zip contains the
`session_…/` folder at its top level. Pick AirDrop and your Mac. The zip lands in `~/Downloads`.

## 5b. Live-stream to the Mac

1. Start the ingester on the Mac. It listens on port 8770 at `/ingest`.
2. Find the Mac's IP with `ipconfig getifaddr en0`. The phone and the Mac must be on the same Wi-Fi.
3. In the app, enter the IP and turn on **Live stream to Mac**, then tap START.
4. The **Stream** stat shows Connecting… → Streaming → Sent. If the socket fails, a yellow warning appears and recording **continues locally**. You can AirDrop the session afterwards.
5. If the connection fails at once, check **Settings > Privacy & Security > Local Network > RGBD Capture**. Also check that the macOS firewall allows the ingester.

## Notes

- The Simulator and non-LiDAR phones can run the app, but recording is disabled and a banner explains why.
- Unit tests for the keyframe policy, depth conversion, PNG16, JPEG, and zip run on the Simulator. To run them: `xcodebuild -project RGBDCapture.xcodeproj -scheme RGBDCapture -destination 'platform=iOS Simulator,name=<any iPhone>' test`
- CLI device build check: `xcodebuild -project RGBDCapture.xcodeproj -scheme RGBDCapture -destination 'generic/platform=iOS' -configuration Debug CODE_SIGNING_ALLOWED=NO build`
