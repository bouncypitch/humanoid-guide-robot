# Gemma Hardware Hackathon: research brief (compiled 2026-10-23)

## Rules and prizes (from the logged-in event page)
- Gemma must run 100% on the device in front of you, with no cloud. Projects must be open source and built on the day. Teams of 3-4.
- Save a draft by 7pm (finalists are picked from drafts). Final submit by 8pm; Submit locks the project. Demos are 3 minutes; the top 6 demo on stage.
- Pick one main track: Mobile AI or Physical AI/IoT/Robotics ($7.5k/$5k/$2.5k each).
- Open to every project: Best in Show for an Android/Pixel end-to-end app ($15k + Pixel 11), Best Use of LiteRT ($7.5k/$5k/$2.5k), and hardware awards (Pixel/TPU, Raspberry Pi, Coral, Qualcomm).

## Robots (Ultimate Bots)
- Booster K1: a team can get dedicated access. Jetson Orin NX 8 GB, stereo camera, 6-mic array, ROS 2 Humble. Speed capped at 0.5 m/s.
- Unitree G1, Agibot X2 and YAM Ultra arms: only by working with their handlers. Simulation first.
- Possible NVIDIA loaners: 15 Orin Nano Super and 15 Jetson Thor kits (secondary source).

## On-device Gemma stack
| Need | Use | Notes |
|---|---|---|
| Android/Pixel | google-ai-edge/gallery + LiteRT-LM v0.18 (Kotlin `ToolSet`, `@Tool`) | Building Gallery yourself needs your own HF OAuth app, or sideload the `.litertlm` file. GPU needs `uses-native-library` in the manifest. Init takes ~10 s. |
| Model files | litert-community/gemma-4-E2B-it-litert-lm, -E4B- | Apache-2.0, no gate. The Tensor G6 NPU file is listed in the docs but wasn't found on HF. |
| Jetson | ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin | `--enable-auto-tool-choice --tool-call-parser gemma4`. Audio only works under vLLM. 8 GB fits only E2B. |
| Mac dev | Ollama `gemma4:e2b/e4b`; llama.cpp ggml-org/gemma-4-E*-it-GGUF (audio is experimental) | |
| Custom tool model | gemma-cookbook FunctionGemma notebook + google/mobile-actions dataset | |
| Avoid | Pi + Hailo HAT (no Gemma 4), Coral (Gemma 3 270M only), Tensor SDK (beta approval needed), Qualcomm IQ8 (fragile version matching) | |

## Robot glue
- K1 from a Mac: `pip install booster-sdk` (IntelligentRoboticsLab, Apache, has a macOS arm64 wheel). Official boosteros and the official SDK are Linux-only.
- Kaweees/booster-rpc (MIT): bridges a phone or laptop to the K1 over gRPC/WebSocket.
- robotmcp/ros-mcp-server (Apache): an MCP tool layer over rosbridge. nasa-jpl/rosa (Apache): a LangChain ROS agent.
- The booster SDK's LuiClient gives local ASR/TTS on the robot.
- Prompt patterns: google-research code_as_policies.
- SeeDo (no license): video → plan pipeline. Borrow ideas, not code.

## Prior art: Break Glass (physical prompt injection)
- "Hijacking Robots with a Piece of Paper" (arXiv 2608.05715): GPT-4o 27%, Gemini 2.5 Flash 29%; "OPERATOR OVERRIDE"-style signs ~80%. Four categories: signage, task redefinition, authority impersonation, conflict.
- CHAI (2510.00181): 82-95% on drones and driving, and on a real RC car.
- Novel as of now: a live humanoid demo, measured success rates on on-device Gemma 4, and a guardrail that doesn't rely on the model, with before/after numbers.
- Reuse: RoboGuard (MIT), FigStep (MIT), SCAM dataset. ShieldGemma 2 is not an injection detector; use a second E2B as an intent checker.
- Card deck: authority, task redefinition, fake signage, ignore-user, fake tool-call text, low-contrast text, image-only, multilingual, split across cards.
- Metrics: attack success rate per category (10+ trials), false-block rate, monitor latency, time to stop.

## What wins and what's overdone
- Recent winners:
  - GEM-4 (Kaggle Gemma 4 Good, 1st): a physical robot; 31B labels its data and a fine-tuned E2B acts as the VLA.
  - Gilbeot won the LiteRT prize with E2B on Android.
- Overdone: blind narrators, offline disaster apps, AAC/speech fine-tunes, Gallery Skills copies, talking boxes on a Pi or Jetson.
- Rewarded:
  - a physical closed loop
  - a small model gating Gemma, plus deterministic safety code
  - live on-device numbers in airplane mode
  - a specific user with a purpose-built form factor
  - a small fine-tune with a before/after metric

## Round 2: recent hackathon winners (researched 2026-10-24)
**On-device and mobile**
- ExecuTorch SF, Jun 2026, 1st: SafeScreen AI, a real-time on-device safety layer.
- Liquid × AMD SF, 1st: video search from a plain-language description.
- Arm challenge, 1st: private search over screenshots.
- Cactus × DeepMind: hybrid FunctionGemma routing.
- Winning themes: live camera with an instant reaction (~12 of 27), private data (~10), physical feedback (~7). The winners were ones where being local was essential to the product.
- Rarely won: chatbots, document RAG, copilots, benchmark-only projects.

**Robotics**
- Humanoid Hack Tokyo, G1, 1st place: mochi pounding (May 2026), Rescue G1 (Jul 2026; finds a fallen person in the dark, talks in 3 languages, alerts staff).
- RoboHacks YC, Apr 2026, 1st: DimiMax, a pharmacy robot that learned a building from one video and fetched by voice.
- Seeed/NVIDIA, Oct 2025, 1st: Matcha Bot.
- LeRobot 2025: T-shirt folding plus pill sorting.
- Winning themes:
  - one clear task with a named beneficiary (elder care, rescue, food or craft)
  - live completeness on real hardware
  - generalization shown live
  - voice in, speech out
  - ready-made perception plus the robot's existing controllers
- Safety touches were praised but never the reason a project won. Sim-only, companion bots and chess or drawing arms didn't place.
