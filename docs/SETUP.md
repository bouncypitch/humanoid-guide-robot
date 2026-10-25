# gemma-hack: on-device Gemma 4 brain for the Booster K1

Gemma 4 runs 100% locally (Ollama on 127.0.0.1, or LiteRT-LM in-process). It sees a camera frame,
picks one tool (walk, turn, look, wave, say, stop), and `robot.py` executes it with a velocity clamp
(at most 0.5 m/s) and a rate limit (at most 1 motion command per second). The same code drives the
MuJoCo K1 (Booster's own walking policy) or the real K1.

## Layout
| Path | What |
|---|---|
| `brain.py` | Main loop: frame, then Gemma (Ollama), then tool call, then `SafeRobot` |
| `robot.py` | `SafeRobot` (clamp, deadlines, 1 Hz) and backends: `sim`, `booster_sdk` (Mac to real K1 over DDS), `boosteros` (official SDK, Linux/Jetson only), `remote` (HTTP to `robot_server.py`) |
| `robot_server.py` | HTTP bridge you run next to the robot (BoosterOS on Linux), or `--backend sim` to test |
| `k1_sim.py` | Headless MuJoCo K1 (booster_deploy `k1_walk` policy) with a head camera, a chase camera and coloured props |
| `tools.py` | Tool schemas that Gemma sees |
| `bench_gemma.py` | Ollama check: image + tools give a well-formed tool call; prints TTFT and tok/s |
| `litert_bench.py` | The same check through LiteRT-LM (`.litertlm`), including audio input |
| `booster_assets/`, `booster_deploy/`, `booster_robotics_sdk/` | Clones from github.com/BoosterRobotics |
| `models/gemma-4-E2B-it.litertlm` | LiteRT-LM weights (litert-community, ungated) |
| `reference/boosteros_whl/` | Unpacked `boosteros` 1.2.2 wheel, kept for API reading only (not installed) |
| `docs/booster_doc.*` | The Ultimate Bots doc (text and HTML with links) |
| `runs/` | Per-run frames and `log.jsonl` |

Ollama weights live in `~/.ollama/models` (Ollama's default location): `gemma4:e2b-it-qat` (4.3 GB) and `gemma4:e4b-it-qat` (6.1 GB).

## Run it
```zsh
cd ~/Code/gemma-hack
# 0. start Ollama (keep this terminal open, or use: brew services start ollama)
OLLAMA_FLASH_ATTENTION=1 ollama serve

# 1. check Gemma: image + tools give a tool call, with timings
.venv/bin/python bench_gemma.py --model gemma4:e2b-it-qat
.venv/bin/python bench_gemma.py --model gemma4:e4b-it-qat

# 2. sim K1 smoke test (walk, turn, look, wave; saves runs/smoke_*.png)
.venv/bin/python k1_sim.py
./mjpy k1_sim.py --viewer                     # same, in the MuJoCo window
.venv/bin/python robot.py                     # SafeRobot self-test (clamps a 2 m/s request)
cd booster_deploy && ../mjpy scripts/deploy.py --task k1_walk --mujoco   # Booster's own sim; type "0.4 0 0" + Enter
cd ..

# 3. the brain
.venv/bin/python brain.py --goal "Find the red box, walk up close to it, then wave." --steps 20
./mjpy brain.py --viewer --goal "..."                     # watch it live
.venv/bin/python brain.py --model gemma4:e4b-it-qat ...   # better tool use, about 1.7x slower
.venv/bin/python brain.py --camera webcam --goal "If you see a person, wave and say hi"
#   (on first use, macOS asks for camera permission for your terminal app)
#   Type a new goal + Enter while it runs; q to quit. Ctrl-C always sends stop().

# 4. LiteRT-LM (for the "Best Use of LiteRT" prize)
.venv-litert/bin/python litert_bench.py                          # image + tools
say -o runs/cmd.wav --data-format=LEI16@16000 "turn left and find the green pillar"
.venv-litert/bin/python litert_bench.py --audio runs/cmd.wav     # spoken command + image + tools
.venv-litert/bin/litert-lm run models/gemma-4-E2B-it.litertlm --backend gpu --attachment runs/smoke_head.png --prompt "What do you see?"
```

## Real K1 (at the venue)
Option A: drive it from the Mac with the community Rust/Python SDK `booster-sdk` (installed; has macOS wheels; untested against a real robot):
```zsh
# Put the Mac on the robot network (Ethernet dongle recommended so DDS discovery works). Robot standing, in walk mode.
.venv/bin/python brain.py --robot booster_sdk --camera webcam --goal "..."
# booster-sdk has no image API. For the robot's own camera, run robot_server.py on the robot and pass
#   --frame-url http://<robot-ip>:8765/frame.jpg
```
Option B: use the official BoosterOS, which runs only on Linux (the K1's Jetson or an Ubuntu 22.04 laptop):
```bash
# on the robot or Linux box
python3 -m pip install boosteros requests pillow numpy
python3 robot_server.py --backend boosteros --port 8765
# on the Mac (Gemma stays on the Mac)
.venv/bin/python brain.py --robot remote --url http://<robot-ip>:8765 --goal "..."
```
Test the remote path on the Mac without a robot: `.venv/bin/python robot_server.py --backend sim --host 127.0.0.1`,
then `brain.py --robot remote`.

Safety: `SafeRobot` clamps vx to [-0.3, 0.5], |vy| to 0.3 and the combined speed to 0.5 m/s. It also caps yaw at 0.8 rad/s,
stops every walk or turn after its deadline (at most 3 s), and limits motion to 1 command per second. `stop()` is never delayed.
`robot_server.py` clamps again and has a 3 s no-command watchdog.
Speech uses on-device TTS (macOS `say`, or espeak-ng on Linux). The BoosterOS `brain.speech` provider is cloud, so it isn't used.

## Measured on this Mac (M1 Pro, 16 GB); one decision = system prompt + 1 image + 6 tools, about 590 prompt tokens
| Runtime / model | Well-formed tool calls | Time per decision* | Decode | Prefill |
|---|---|---|---|---|
| Ollama `gemma4:e2b-it-qat` | 4/4 | 1.05 s | 79 tok/s | about 700 tok/s |
| Ollama `gemma4:e4b-it-qat` | 4/4 | 1.8 s | 47 tok/s | about 410 tok/s |
| LiteRT-LM E2B `.litertlm` (GPU) | 3/3 image, 1/1 audio | about 3 s wall (TTFT 0.42 s) | 60 tok/s | about 1700 tok/s |

\*Ollama returns a tool call in one chunk, so TTFT is roughly the full decision time.
Cold model load: E2B about 4 s, E4B about 5.5 s. `--think` gives about the same TTFT, but a decision takes about 3.3 s.
Ollama's gemma4 tags take text and images only (no audio). For audio, use LiteRT-LM (verified) or llama.cpp with ggml-org GGUFs.

## iPhone scan -> floor plan -> route -> rehearsal (scan_map.py)
1. Scan the hallway and kitchen with the iPhone Pro's LiDAR (3D Scanner App or Polycam LiDAR mode, NOT photo mode). Include the tape X at the K1's start spot. Export OBJ, AirDrop it to `scans/`.
2. `.venv/bin/python scan_map.py build scans/agihouse.obj maps/agihouse`, then open `maps/agihouse/map.png` (1 m grid).
3. `scan_map.py place maps/agihouse start X Y --yaw DEG` (the K1's start pose) and `scan_map.py place maps/agihouse kitchen X Y`
4. `scan_map.py rehearse maps/agihouse kitchen --gif`: plans the route with A*, then walks it in MuJoCo with Booster's RL walking policy. Writes rehearsal.png and rehearsal.gif. Exits with 2 if it hit something, fell or timed out.
Test without a scan: `scan_map.py synth scans/synth_room.obj` (start 0.8,3.0 yaw 0; kitchen 6.8,4.5).
