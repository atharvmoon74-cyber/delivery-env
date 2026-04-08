import os
import json
from typing import Any, Dict, List
from openai import OpenAI
from environment import DeliveryEnv

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

VALID_ACTIONS = {0, 1, 2}


def clamp_score(value: float) -> float:
    # Strictly inside (0, 1)
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    return value


def fallback_action(state: Dict[str, Any]) -> int:
    agent_pos = state.get("agent_pos", 0)
    pending_orders: List[int] = state.get("pending_orders", [])

    if not pending_orders:
        return 2

    target = pending_orders[0]

    if agent_pos < target:
        return 0
    elif agent_pos > target:
        return 1
    return 2


def parse_action(text: str, state: Dict[str, Any]) -> int:
    if not text:
        return fallback_action(state)

    text = text.strip()

    try:
        action = int(text)
        if action in VALID_ACTIONS:
            return action
    except Exception:
        pass

    for ch in text:
        if ch in {"0", "1", "2"}:
            return int(ch)

    return fallback_action(state)


def choose_action(state: Dict[str, Any]) -> int:
    prompt = f"""
You are controlling a delivery agent in a 1D grid world.

Actions:
0 = move_right
1 = move_left
2 = deliver_order

Current state:
{json.dumps(state)}

Return only one digit: 0, 1, or 2.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Reply with only one digit: 0, 1, or 2."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=5,
        )
        text = response.choices[0].message.content
        return parse_action(text, state)
    except Exception:
        return fallback_action(state)


def run_episode(difficulty: str) -> float:
    env = DeliveryEnv(difficulty=difficulty)

    try:
        state = env.reset()
    except Exception:
        return 0.01

    total_reward = 0
    step_count = 0
    max_steps = 100
    done = False

    while not done and step_count < max_steps:
        action = choose_action(state)

        try:
            state, reward, done = env.step(action)
        except Exception:
            return 0.01

        total_reward += reward
        step_count += 1

    # Safe score mapping guaranteed inside (0,1)
    # Keep it simple and validator-safe
    raw_score = 0.5 + (total_reward / 200.0)
    return clamp_score(raw_score)


def main() -> None:
    scores = {
        "easy": run_episode("easy"),
        "medium": run_episode("medium"),
        "hard": run_episode("hard"),
    }

    # Print only clean machine-readable output
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
