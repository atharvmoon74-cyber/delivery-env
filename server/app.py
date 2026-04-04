from fastapi import FastAPI
import uvicorn
from environment import DeliveryEnv

app = FastAPI()

env = None


@app.post("/reset")
def reset():
    global env
    env = DeliveryEnv(difficulty="easy")
    return env.reset()


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


# 🔥 REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# 🔥 REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
