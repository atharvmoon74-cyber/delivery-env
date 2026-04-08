import os
import json
from openai import OpenAI
from environment import DeliveryEnv

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

Actions:
0 = move_right
1 = move_left
2 = deliver_order

Current state:
{json.dumps(state)}

Return only one number: 0, 1, or 2.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a delivery optimization agent."},
            {"role": "user", "content": prompt}
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

    return 0

def run_episode(difficulty="easy"):
    env = DeliveryEnv(difficulty=difficulty)
    state = env.reset()

    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        state, reward, done = env.step(action)
        total_reward += reward

    return total_reward

if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        score = run_episode(difficulty)
        print(f"{difficulty}: {score}")
