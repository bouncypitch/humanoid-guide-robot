# Real K1 runbook (robot table)

Status: everything below was tested only on the Mac against `robot_server.py --dry-run`. It has
**not** been run on a K1. Each step is a check that should pass before you go on to the next one.
Keep one hand on the e-stop or the handler's remote the whole time.

## 0. Network
- Connect the Mac to the robot over Ethernet if you can (use the dongle), otherwise use the robot Wi-Fi.
  Ask the handler for the Jetson IP and login. Below, `K1=<jetson-ip>`.
- `ping $K1`, then `ssh <user>@$K1`.
- The robot must be standing and in **walk** mode (handler remote: prepare -> walk) before you
  send any motion command. `reset_odom` and head commands only work in walk mode.

## 1. Copy files to the Jetson and start the server
```bash
# Mac
scp robot_server.py robot.py <user>@$K1:~/gemma/
# Jetson
cd ~/gemma
python3 -c "import boosteros, numpy, cv2; print('ok')"   # if this fails: python3 -m pip install --user boosteros opencv-python-headless pillow
source /opt/ros/humble/setup.bash
ros2 topic list | grep -i -E "camera|odom|imu"           # expect /boostercamera/head/{rgb,depth,rgb/camera_info}, /odom
ros2 topic hz /boostercamera/head/depth                  # should be ~10-30 Hz. If there is no depth topic, see step 4b
python3 robot_server.py --backend boosteros --port 8765
```
Mac: `curl http://$K1:8765/status` should return `{"mode": "walk"}`.

## 2. Odometry: push the robot by hand
```bash
watch -n 0.3 curl -s http://$K1:8765/odom      # or a loop in another terminal
curl -s -X POST http://$K1:8765/reset_odom      # x, y, th should go to ~0
```
- Walk it forward about 1 m (handler steers, or send `/drive` from step 5): **x should go up by about 1**.
- Turn it left (counter-clockwise from above): **th should go up** (radians).
- If x or th has the wrong sign, stop and fix `K1Sensors.odom` before you navigate anything.

## 3. Depth slice: stand 1 m from a wall
Face the robot squarely at a plain wall 1.0 m away, with the head level (`curl -X POST -d '{"yaw":0,"pitch":0}' http://$K1:8765/head`).
```bash
curl -s http://$K1:8765/scan | python3 -c "
import json,sys; j=json.load(sys.stdin); r=j['ranges']; n=len(r)
print('floor', j['floor'], 'valid', j['valid'], '/', n)
print('centre ranges', r[n//2-3:n//2+4])
print('left edge angle', j['angles'][-1], 'right edge', j['angles'][0])"
```
Pass criteria:
- `floor.fit` is true, `cam_h` is about 0.8-0.95 m and `pitch_deg` matches the head pitch to within a few degrees.
  If `fit` is false, the slice uses `--cam-height/--cam-pitch`. Measure the camera height and restart with those values.
- The centre ranges read about **1.0 m** (+-0.1).
- Put a box at the robot's **left**: the short ranges should appear at **positive** angles.
- `valid` should be well over half of the beams in a furnished room.
- If the ranges are about 1000x off, the depth encoding is not 16UC1 mm/32FC1 m. Check
  `ros2 topic echo --once /boostercamera/head/depth --field encoding` and fix `depth_to_metres`.

## 4b. Only if there is no depth topic
- Run `ros2 topic list | grep boostercamera`. If depth is published under another name, restart with
  `--depth-source ros --depth-topic <topic> --info-topic <camera_info topic>`.
- If only rectified left/right images exist, use
  `--depth-source sgbm --left-topic ... --right-topic ... --baseline <m>`. This runs OpenCV
  StereoSGBM at 320 px wide on the CPU. The images must already be rectified, and you need the baseline from the handler or the
  camera_info P matrix (`-P[0,3]/P[0,0]` of the right camera). This path is the least tested.

## 5. 1 m drive test (Mac)
```bash
curl -s -X POST http://$K1:8765/reset_odom
for i in $(seq 1 25); do curl -s -X POST -d '{"vx":0.3,"vyaw":0}' http://$K1:8765/drive >/dev/null; sleep 0.1; done   # ~2.5 s at 0.3 m/s
curl -s http://$K1:8765/odom
```
Expect the robot to move about 0.6-0.8 m and the odometry to agree with a tape measure to within about 10%.
Once the loop ends, the **0.5 s deadman** must stop the robot on its own, with no stop command sent.

## 6. Hard-stop tests
- Start the loop above and press Ctrl-C after 1 s. The robot must stop within 0.5 s (deadman).
- `curl -X POST http://$K1:8765/stop` stops it immediately.
- Run `navigator.py` (step 7) and pull the Mac's Ethernet cable or turn off Wi-Fi. The robot must stop within 0.5 s.
- Ctrl-C in `navigator.py` sends `/stop` before it exits.

## 7. Navigate
Put the K1 at the front entrance facing into the house (the map heading there is 90 deg; pass `--yaw` if it faces another way):
```bash
.venv/bin/python navigator.py maps/agihouse --robot real --url http://$K1:8765 \
    --start-at front_entrance --goal restroom
```
- `start_fix moved_m` should be small (< 0.3). A large value means the start pose or yaw is wrong.
- Leave out `--start-at` to make it spin 360 deg and localize globally (slower; it may step 1.5 m into open space and spin again).
- Output goes to `runs/<ts>-nav/trial_00.png`, showing the plan and the belief path (there is no ground truth on the real robot).
  Arrival is judged on the estimated pose (< 0.75 m).
- Rehearse on the Mac first, with no robot needed:
  `.venv/bin/python robot_server.py --dry-run` and then the same command with `--url http://127.0.0.1:8765`.

## 8. Gemma 4 E2B for conversation.py
`conversation.py` uses **LiteRT-LM** (`litert_lm`) with `models/gemma-4-E2B-it.litertlm` (2.6 GB).

**A. Recommended: Gemma on the Mac, robot over the network.** This is the setup that has actually been run on the Mac:
```bash
.venv/bin/python conversation.py --robot remote --url http://$K1:8765
```
Audio and vision run through Gemma on the Mac. `robot_server.py` handles motion, head, `say` and `frame.jpg`. Over Ethernet the latency is negligible. Over Wi-Fi expect
50-200 ms per frame. Nothing leaves the local network, so this still runs on-device, on the "edge laptop".

**B. Gemma on the K1's Jetson (Orin NX 8 GB, shared CPU/GPU memory). Unverified, so treat it as a stretch goal.**
Memory budget: BoosterOS, the walking controller, the camera and DDS already use part of the 8 GB. Check with `free -h`
on the robot before you try. E2B in 4-bit needs about 3 GB with a short context, and the vision/audio encoders add more.
It may fit, but run nothing else on the Jetson besides `robot_server.py`.
Options, in rough order of realism:
1. **Ollama** (has JetPack 6 / CUDA arm64 builds): `curl -fsSL https://ollama.com/install.sh | sh`, then
   `ollama run gemma4:e2b` (check the exact tag on ollama.com). This uses the GPU. `conversation.py` does **not**
   have an Ollama backend yet; you would need a small adapter that calls the OpenAI-compatible `http://$K1:11434/v1` endpoint.
2. **NVIDIA Jetson AI Lab containers** (jetson-ai-lab.com "Gemma 4 on Jetson"): llama.cpp/GGUF or vLLM with tool-calling flags.
   This is the same OpenAI-compatible API, so it needs the same missing adapter. The containers are multi-GB, so pull them before arriving at the table.
3. **LiteRT-LM on arm64 Linux**: `python3 -m pip install litert-lm` (if an aarch64 wheel exists), copy the
   `.litertlm` file, and run `conversation.py --backend cpu` on the Jetson. Expect only a few tokens/s on the CPU cores. The GPU backend
   on Jetson is unknown.
If B isn't working within 15 minutes, use A. The demo does not depend on where Gemma runs.

## Reference: K1 interfaces used (from the BoosterOS 1.2.2 wheel in `reference/` and `booster_robotics_sdk/`)
- Motion: `K1().set_velocity(vx, vy, vyaw)` = `B1LocoClient.Move`. `reset_odom()` = `ResetOdometry` (walk mode only).
- Odometry: `K1().get_odom().pose_2d` -> [x, y, yaw]. SDK DDS topics: `rt/odometer_state` (Odometer x/y/theta) and
  `rt/odom` (ROS 2 `/odom`, nav_msgs/Odometry).
- Fall: `K1().get_fall_down_state().state` in normal/falling/fallen/getting_up (`rt/fall_down`).
- Depth: `K1().get_image(img_type="depth")` = ROS 2 `/boostercamera/head/depth` (sensor_msgs/Image). Intrinsics come from
  `/boostercamera/head/rgb/camera_info`. The server rescales them if the depth resolution differs. The units are assumed to be
  16UC1 = mm or 32FC1 = m. Depth registered to the RGB camera is assumed and has not been verified.

## The full talking demo on the real K1 (after the steps above pass)
```zsh
# Mac (Gemma 4 E2B via LiteRT-LM runs here; no cloud). Robot placed at the front door facing into the house:
.venv/bin/python escort.py maps/agihouse --robot real --url http://$K1:8765 --start-at front_entrance --yaw 90
# unknown start instead (robot looks around, asks "which room are we in?" if unsure):
.venv/bin/python escort.py maps/agihouse --robot real --url http://$K1:8765
```
Push-to-talk with Enter (or add `--text` to type in a loud room). Type `s` + Enter to stop at once; the server
deadman also stops the robot 0.5 s after the Mac goes quiet. Verified on the Mac against `robot_server.py --dry-run`
(entrance → restroom: arrived, 0.32 m from the door by the dry server's ground truth).

Note: `--rerun` launches the Rerun viewer binary from the venv, so activate it first
(`source .venv/bin/activate`), otherwise you get "Rerun Viewer executable not found".
