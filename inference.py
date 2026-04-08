import os
import json
import math
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


def fallback_action(state: Dict[str, Any]) -> int:
    """
    Safe deterministic fallback policy.
    Move toward the first pending order and deliver when on target.
    """
    agent_pos = state.get("agent_pos", 0)
    pending_orders: List[int] = state.get("pending_orders", [])

    if not pending_orders:
        return 2

    target = pending_orders[0]

    if agent_pos < target:
        return 0  # move right
    elif agent_pos > target:
        return 1  # move left
    else:
        return 2  # deliver


def parse_action(text: str, state: Dict[str, Any]) -> int:
    """
    Parse model output safely into action 0/1/2.
    """
    if not text:
        return fallback_action(state)

    cleaned = text.strip()

    try:
        action = int(cleaned)
        if action in VALID_ACTIONS:
            return action
    except Exception:
        pass

    for ch in cleaned:
        if ch in {"0", "1", "2"}:
            return int(ch)

    return fallback_action(state)


def choose_action(state: Dict[str, Any]) -> int:
    """
    Ask LLM for next action.
    Never crashes.
    """
    prompt = f"""
You are controlling a delivery agent in a 1D grid world.

Available actions:
0 = move_right
1 = move_left
2 = deliver_order

Current state:
{json.dumps(state)}

Rules:
- Return ONLY one number: 0, 1, or 2
- No explanation
- Be efficient
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a delivery optimization agent. Reply with only one digit: 0, 1, or 2."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=5,
        )
    except Exception as e:
        print(f"[WARNING] API call failed: {e}")
        return fallback_action(state)

    try:
        text = response.choices[0].message.content
    except Exception as e:
        print(f"[WARNING] Could not read response: {e}")
        return fallback_action(state)

    try:
        return parse_action(text, state)
    except Exception as e:
        print(f"[WARNING] Could not parse action: {e}")
        return fallback_action(state)


def normalize_score(total_reward: float, difficulty: str) -> float:
    """
    Convert raw reward to a score strictly between 0 and 1.
    Uses sigmoid-like mapping and clamps away from exact 0 and 1.
    """
    scale_map = {
        "easy": 10.0,
        "medium": 12.0,
        "hard": 15.0,
    }

    scale = scale_map.get(difficulty, 12.0)

    score = 1.0 / (1.0 + math.exp(-total_reward / scale))

    # strict clamp so score is never exactly 0 or 1
    eps = 1e-6
    if score <= 0.0:
        score = eps
    elif score >= 1.0:
        score = 1.0 - eps

    # extra safe clamp
    score = max(eps, min(1.0 - eps, score))
    return score


def run_episode(difficulty: str = "easy") -> float:
    """
    Run one episode and return normalized score in (0, 1).
    """
    env = DeliveryEnv(difficulty=difficulty)

    try:
        state = env.reset()
    except Exception as e:
        print(f"[ERROR] env.reset failed for {difficulty}: {e}")
        return 0.000001

    total_reward = 0
    step_count = 0
    max_guard_steps = 200
    done = False

    print("[START]")
    print(f"Difficulty: {difficulty}")
    print(f"Initial State: {state}")

    while not done and step_count < max_guard_steps:
        try:
            action = choose_action(state)
        except Exception as e:
            print(f"[WARNING] choose_action crashed: {e}")
            action = fallback_action(state)

        try:
            next_state, reward, done = env.step(action)
        except Exception as e:
            print(f"[ERROR] env.step failed: {e}")
            break

        print("[STEP]")
        print(f"Step: {step_count}")
        print(f"Action: {action}")
        print(f"State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        total_reward += reward
        state = next_state
        step_count += 1

    if step_count >= max_guard_steps:
        print("[WARNING] Max guard steps reached, stopping episode.")

    print("[END]")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {step_count}")
    print("=" * 40)

    final_score = normalize_score(total_reward, difficulty)
    return final_score


def main() -> None:
    scores = {}

    for difficulty in ["easy", "medium", "hard"]:
        try:
            scores[difficulty] = run_episode(difficulty)
        except Exception as e:
            print(f"[ERROR] Episode crashed for {difficulty}: {e}")
            scores[difficulty] = 0.000001

    print("\nFINAL SCORES:")
    for difficulty, score in scores.items():
        print(f"{difficulty}: {score:.6f}")


if __name__ == "__main__":
    main()
