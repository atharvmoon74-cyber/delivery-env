from fastapi import FastAPI
from environment import DeliveryEnv

app = FastAPI()

env = None


@app.post("/reset")
def reset():
    global env
    env = DeliveryEnv(difficulty="easy")
    state = env.reset()
    return state


@app.post("/step")
def step(action: int):
    global env
    state, reward, done = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }


@app.get("/state")
def state():
    return env.state()