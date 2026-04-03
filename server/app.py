from fastapi import FastAPI
from environment import DeliveryEnv

app = FastAPI()

env = DeliveryEnv(difficulty="easy")


# SUPPORT BOTH GET + POST
@app.get("/reset")
@app.post("/reset")
def reset():
    global env
    env = DeliveryEnv(difficulty="easy")
    return env.reset()


@app.get("/step")
@app.post("/step")
def step(action: int = 0):
    state, reward, done = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }


@app.get("/state")
def state():
    return env.state()
