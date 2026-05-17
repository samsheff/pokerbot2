import anthropic
import base64
import json
import requests
from pathlib import Path

client = anthropic.Anthropic()

POKER_EXTRACTION_PROMPT = """
Analyze this poker table screenshot and extract ALL visible game state into the following JSON structure.
Be precise with numbers. Use null for values you cannot determine. Return ONLY valid JSON, no prose.

{
  "game": {
    "stage": "preflop|flop|turn|river|showdown",
    "pot": 0.00,
    "side_pots": [],
    "community_cards": [],        // e.g. ["Ah", "Kd", "2c"]
    "current_bet": 0.00,
    "dealer_position": 1,         // seat number
    "hero_seat": null             // if identifiable
  },
  "players": [
    {
      "seat": 1,
      "name": "",
      "stack": 0.00,
      "bet": 0.00,                // chips in front of player this street
      "cards": [],                // hole cards if visible, e.g. ["Ac", "Kh"]
      "status": "active|folded|all_in|sitting_out|empty",
      "is_dealer": false,
      "is_hero": false,
      "action_indicator": null    // "thinking", "posted_blind", etc. if visible
    }
  ],
  "blinds": {
    "small": 0.00,
    "big": 0.00,
    "ante": null
  },
  "table_name": null,
  "hand_number": null,
  "timestamp": null
}
"""

def extract_poker_state(image_path: str | Path) -> dict:
    img_data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    ext = Path(image_path).suffix.lower()
    media_type = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/png")
    
    response = client.messages.create(
        model="claude-opus-4-7",  # Use Opus for best accuracy on complex tables
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": POKER_EXTRACTION_PROMPT}
            ]
        }]
    )
    
    raw = response.content[0].text.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw.strip())


def extract_from_screen(monitor: int = 1) -> dict:
    """Live screen capture variant using mss"""
    import mss
    import mss.tools
    import tempfile
    
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[monitor])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=f.name)
            return extract_poker_state(f.name)


def get_action(
    state: dict,
    actions: list[str],
    pov: int,
    server_url: str = "http://localhost:8888",
) -> dict:
    """
    Query the inference server for the bot's action given the current game state.

    Args:
        state:      Output of extract_poker_state() or extract_from_screen().
        actions:    Action strings since the start of the hand, NOT including
                    blinds. Each string matches the server's Action format:
                    "fold", "check", "call <amount>", "raise <amount>".
                    Example: ["raise 3", "call 3", "raise 9"]
        pov:        Hero's position — 0 = button (acts first preflop out of
                    position), 1 = big blind.
        server_url: Base URL of the running backend server.

    Returns:
        dict with keys:
            "action" — the chosen action string (e.g. "raise 12")
            "legal"  — list of all legal actions at this decision point
    """
    hero = next((p for p in state["players"] if p.get("is_hero")), None)
    if hero is None or not hero.get("cards"):
        raise ValueError("hero's hole cards not visible in state")

    hole = hero["cards"]
    board = state["game"].get("community_cards") or []

    resp = requests.post(
        f"{server_url}/api/decide",
        json={"hole": hole, "board": board, "actions": actions, "pov": pov},
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # From file
    state = extract_poker_state("table.png")
    print(json.dumps(state, indent=2))

    # Get action from inference server (example — track actions across the hand)
    # result = get_action(state, actions=[], pov=0)
    # print(result["action"])

    # Live from screen
    # state = extract_from_screen(monitor=1)
