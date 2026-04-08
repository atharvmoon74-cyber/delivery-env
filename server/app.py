from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import DeliveryEnv

app = FastAPI()
env = None

class StepRequest(BaseModel):
    action: int

@app.get("/")
def root():
    return {"message": "DeliveryEnv API is running"}

@app.post("/reset")
def reset():
    global env
    env = DeliveryEnv(difficulty="easy")
    return env.reset()

@app.post("/step")
def step(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    state, reward, done = env.step(req.action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()
