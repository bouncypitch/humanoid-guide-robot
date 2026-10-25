# RGBD Capture (Android)

This app records an RGB-D walkthrough on an ARCore phone (built Pixel-first) in the format described in
`../docs/rgbd_format.md`. It is the Android counterpart of `../ios_app`. You can share each session as a zip
or live-stream it to the Mac ingester at `ws://<mac-ip>:8770/ingest`. It can also label keyframes with
Gemma running **on the phone** (LiteRT-LM, NPU/GPU/CPU).

Kotlin, Jetpack Compose, ARCore 1.57, ML Kit Selfie Segmentation, OkHttp, LiteRT-LM 0.18.1.
minSdk 30 (Android 11), target/compile SDK 37. Package `com.kdotmahesh.rgbdcapture`.

## 1. Build

You need a JDK 17+. Android Studio's bundled JBR works, and so does the Android SDK with platform 37.

```bash
cd android_app
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"   # macOS + Android Studio
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties                       # if Studio didn't create it
./gradlew assembleDebug          # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest      # JVM unit tests (keyframe policy, depth/PNG16, intrinsics, pose, JSON, zip)
./gradlew lintDebug
```

Or open the `android_app/` folder in Android Studio and press Run.

## 2. Install

Turn on **Developer options > USB debugging** on the phone (tap *Build number* 7 times first), plug it in, and accept the prompt.

```bash
./gradlew installDebug
# or, with a built APK:
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

To sideload without adb, copy `app-debug.apk` to the phone (Drive, email, USB) and tap it. Allow
"Install unknown apps" for the app you opened it from.

On first launch the app asks for **Camera** access. If **Google Play Services for AR** is missing or
outdated, it opens the Play Store to install it. The phone must be
[ARCore-supported with the Depth API](https://developers.google.com/ar/devices). All recent Pixels are.

## 3. Optional: put a Gemma model on the phone

The app never bundles weights. Get a `.litertlm` from
[litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
(log in and accept the license first, or you'll download an HTML page instead of a model):

- `gemma-4-E2B-it_Google_Tensor_G6.litertlm` is the NPU build for Tensor G6 phones. Check your SoC with
  `adb shell getprop ro.soc.model`. An NPU build only runs on that SoC's NPU.
- `gemma-4-E2B-it.litertlm` is the generic build, for GPU/CPU on any phone.

There are three ways to get the model onto the phone. Pick one:

```bash
# A. Into the app's own folder (no permission or copy needed; found automatically):
adb shell mkdir -p /sdcard/Android/data/com.kdotmahesh.rgbdcapture/files/models
adb push gemma-4-E2B-it_Google_Tensor_G6.litertlm /sdcard/Android/data/com.kdotmahesh.rgbdcapture/files/models/

# B. Into Downloads:
adb push gemma-4-E2B-it_Google_Tensor_G6.litertlm /sdcard/Download/
#    then in the app either tap "Load model" and pick it (copies it into the app),
#    or tap "Read Downloads" (grants all-files access) and then "Rescan" to use it in place.
```

Turn on **On-device Gemma labels**. The app tries the backends in this order: NPU (Tensor), then NPU with GPU vision, then GPU, then CPU. The status line shows which
backend loaded and the latency per request. Loading takes several seconds.

## 4. Record

1. Hold the phone in portrait. Wait for **Tracking: Normal** and a live depth preview.
2. Tap **START** and walk slowly. Point the phone at walls and floor, and avoid fast turns.
3. A keyframe is saved each time you move 0.15 m or turn 10 degrees, at most 3 per second, and only while tracking is
   Normal. **Dropped** counts keyframes skipped because more than 5 were waiting to be written.
4. Tap **STOP**. (Leaving the app also stops and finalises the recording.)

Sessions are saved to `/sdcard/Android/data/com.kdotmahesh.rgbdcapture/files/sessions/session_YYYYMMDD-HHMM/`:

```bash
adb pull /sdcard/Android/data/com.kdotmahesh.rgbdcapture/files/sessions/ ./captures/
```

Each keyframe writes `NNNNNN.jpg`, `_depth.png` (16-bit mm), `_conf.png`, `_mask.png` (with person masks on),
and `NNNNNN.json`. With Gemma on, roughly one keyframe per second also gets `NNNNNN_objects.json`.

## 5a. Share

Tap **Share last session**. The app zips the folder to `session_….zip`, with the folder at the top level of the
zip, and opens the Android share sheet (Drive, Quick Share, Gmail, …). On the Mac:
`python rgbd_ingest.py session_….zip scans/house`.

## 5b. Live-stream to the Mac

1. On the Mac, run `python rgbd_ingest.py --serve` (port 8770, path `/ingest`).
2. Find the Mac's IP with `ipconfig getifaddr en0`. Both devices must be on the same Wi-Fi.
3. Enter the IP, turn on **Live stream to Mac**, and tap START. **Stream** shows Connecting… → Streaming → Sent N.
   If the socket fails, a yellow warning appears and recording **continues locally**.
4. Leave **Send object messages on the live stream** off. The current ingester doesn't understand
   the extra `{"objects": …}` message yet (see "Android extensions" in `../docs/rgbd_format.md`).
   The objects are still saved in the session folder.

## Code map

| File | What it does |
|---|---|
| `ar/ArEngine.kt` | Headless ARCore loop on its own EGL thread: camera config, depth mode, tracking, keyframe grab, depth preview |
| `ar/SessionRecorder.kt` | Writes one session on a background executor, handles person masks and Gemma, and drives the streamer |
| `ar/FrameData.kt` | Copies YUV/depth/confidence out of ARCore `Image`s, converts YUV to RGB |
| `CaptureMath.kt` | Pure Kotlin: keyframe policy, pose transposition, intrinsics crop/scale, depth conversion, colormap |
| `io/PngWriter.kt` | 16-bit and 8-bit grayscale PNG encoder (Android's Bitmap can't write 16-bit) |
| `io/Json.kt` | session/frame/objects JSON, tolerant parser for Gemma replies |
| `stream/LiveStreamer.kt` | OkHttp WebSocket, protocol per spec, back-pressure |
| `ml/ObjectLabeler.kt`, `ml/LiteRtLmLabeler.kt` | Labeller interface plus the LiteRT-LM implementation (NPU→GPU→CPU) |
| `ml/PersonSegmenter.kt` | ML Kit Selfie Segmentation → depth-resolution mask |
| `ui/CaptureScreen.kt` | Compose UI (same layout as the iOS app) |

## Tested on

Be aware of what has and hasn't been tested:

- **Build:** `./gradlew assembleDebug` succeeds, the 25 JVM unit tests pass, and `lintDebug` reports no issues.
  Run on macOS (Apple Silicon) with AGP 9.4.1, Gradle 9.8.2, Kotlin 2.4.21, JBR 21.
- **Emulator (API 36 Google Play image, arm64):** the app installs and launches. The camera permission works.
  Without ARCore installed, the app opens the ARCore install prompt. With the ARCore APK sideloaded,
  `Session()` throws `FatalException`. The arm64 emulator isn't supported by ARCore (Google's emulator
  build of ARCore is x86-only), and the app shows a banner instead of crashing. The Gemma toggle was tested
  with two invalid model files. A non-LiteRT-LM file is rejected before native code. A file with a valid
  header but corrupt contents falls through NPU, GPU, then CPU, and ends with an error message. The app keeps
  running in both cases. LiteRT-LM's native library loads on arm64.
- **Not tested (no physical Android device was available):** an actual ARCore capture, the depth
  preview, the keyframe images and depth PNG contents from real frames, the intrinsics and pose on a real device,
  ML Kit masks, live streaming to the ingester, the share sheet, and Gemma inference on any backend. Test these first
  on a Pixel. The YUV→RGB colour conversion and the 90-degree rotations (mask, preview) were written to
  ARCore's documented landscape sensor orientation. Check them visually on the first real session.
