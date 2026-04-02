import os
import random
from environment import DeliveryEnv

# =========================
# CONFIG (MANDATORY VARS)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY", "your-api-key")


# =========================
# SIMPLE AGENT (Baseline)
# =========================
def choose_action(state):
    """
    Simple random policy (baseline agent)
    """
    return random.randint(0, 2)


# =========================
# RUN ONE EPISODE
# =========================
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


# =========================
# MAIN RUN (ALL TASKS)
# =========================
if __name__ == "__main__":
    scores = {}

    for difficulty in ["easy", "medium", "hard"]:
        score = run_episode(difficulty)
        scores[difficulty] = score

    print("\nFINAL SCORES:")
    for k, v in scores.items():
        print(f"{k}: {v}")