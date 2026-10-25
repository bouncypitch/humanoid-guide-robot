"""Tool schemas Gemma sees (OpenAI/Ollama function-calling format).

Keep these few and simple: the Booster doc recommends walk, turn, look, wave,
say, stop. Limits here are advisory; robot.SafeRobot enforces the real ones.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "walk",
        "description": "Walk the robot. Positive vx is forward, positive vy is left. "
                       "Speeds in m/s (max 0.5). Walks for `seconds` then stops.",
        "parameters": {"type": "object", "properties": {
            "vx": {"type": "number", "description": "forward speed m/s, -0.3..0.5"},
            "vy": {"type": "number", "description": "sideways speed m/s, -0.3..0.3 (left +)"},
            "seconds": {"type": "number", "description": "how long, 0.5..3"},
        }, "required": ["vx"]}}},
    {"type": "function", "function": {
        "name": "turn",
        "description": "Turn in place. Positive degrees turns left (counter-clockwise).",
        "parameters": {"type": "object", "properties": {
            "degrees": {"type": "number", "description": "-90..90"},
        }, "required": ["degrees"]}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "Point the head. yaw_deg positive looks left; pitch_deg positive looks down.",
        "parameters": {"type": "object", "properties": {
            "yaw_deg": {"type": "number", "description": "-55..55"},
            "pitch_deg": {"type": "number", "description": "-15..45"},
        }, "required": ["yaw_deg", "pitch_deg"]}}},
    {"type": "function", "function": {
        "name": "wave",
        "description": "Wave hello with the right hand.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "say",
        "description": "Speak a short sentence out loud.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "under 20 words"},
        }, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "stop",
        "description": "Stop all motion immediately and stand still.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
