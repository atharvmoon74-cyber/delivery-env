from fastapi import FastAPI
from environment import DeliveryEnv

app = FastAPI()

env = DeliveryEnv(difficulty="easy")


@app.get("/reset")
def reset():
    global env
    env = DeliveryEnv(difficulty="easy")
    return env.reset()


@app.post("/step")
def step(action: int):
    state, reward, done = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }


@app.get("/state")
def state():
    return env.state()
