import os
import json
from openai import OpenAI
from environment import DeliveryEnv

# MUST use injected env vars only
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

def choose_action(state):
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
- Do not explain anything
- Try to deliver all pending orders efficiently
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a delivery optimization agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        action = int(text)
        if action in [0, 1, 2]:
            return action
    except:
        pass

    return 2  # safe fallback


def run_episode(difficulty="easy"):
    env = DeliveryEnv(difficulty=difficulty)
    state = env.reset()

    total_reward = 0
    step_count = 0

    print("[START]")
    print(f"Difficulty: {difficulty}")
    print(f"Initial State: {state}")

    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        print("[STEP]")
        print(f"Step: {step_count}")
        print(f"Action: {action}")
        print(f"State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        total_reward += reward
        state = next_state
        step_count += 1

    print("[END]")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {step_count}")
    print("=" * 40)

    return total_reward


if __name__ == "__main__":
    scores = {}

    for difficulty in ["easy", "medium", "hard"]:
        score = run_episode(difficulty)
        scores[difficulty] = score

    print("\nFINAL SCORES:")
    for k, v in scores.items():
        print(f"{k}: {v}")
