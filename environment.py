import random
from typing import Dict, List, Tuple

class DeliveryEnv:
    """
    Smart Delivery Optimization Environment
    ---------------------------------------
    Agent delivers orders placed at different locations.
    Goal: Deliver all orders efficiently within limited steps.
    """

    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.max_steps = 0
        self.grid_size = 6
        self.reset()

    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.agent_position = 0
        self.steps = 0

        # Difficulty settings
        if self.difficulty == "easy":
            self.orders = [2]
            self.max_steps = 10

        elif self.difficulty == "medium":
            self.orders = [2, 4]
            self.max_steps = 15

        elif self.difficulty == "hard":
            self.orders = [1, 3, 5]
            self.max_steps = 20

        else:
            raise ValueError("Invalid difficulty: choose easy, medium, hard")

        self.total_orders = len(self.orders)

        return self.state()

    def state(self) -> Dict:
        """Return current environment state"""
        return {
            "agent_position": self.agent_position,
            "pending_orders": self.orders.copy(),
            "steps_left": self.max_steps - self.steps,
            "total_orders": self.total_orders
        }

    def step(self, action: int) -> Tuple[Dict, float, bool]:
        """
        Perform action:
        0 = move right
        1 = move left
        2 = deliver order
        """
        reward = 0.0
        done = False

        # Move Right
        if action == 0:
            if self.agent_position < self.grid_size - 1:
                self.agent_position += 1
                reward -= 0.1  # small movement cost
            else:
                reward -= 1  # penalty for hitting boundary

        # Move Left
        elif action == 1:
            if self.agent_position > 0:
                self.agent_position -= 1
                reward -= 0.1
            else:
                reward -= 1

        # Deliver Order
        elif action == 2:
            if self.agent_position in self.orders:
                self.orders.remove(self.agent_position)
                reward += 10  # successful delivery
            else:
                reward -= 2  # wrong delivery attempt

        else:
            reward -= 1  # invalid action

        self.steps += 1

        # Bonus for finishing all deliveries
        if len(self.orders) == 0:
            reward += 20
            done = True

        # End if steps exceeded
        if self.steps >= self.max_steps:
            done = True

        return self.state(), reward, done


# ------------------- TEST SCRIPT -------------------

if __name__ == "__main__":
    print("🚀 Running Delivery Environment Test...\n")

    for level in ["easy", "medium", "hard"]:
        print(f"\n=== Testing {level.upper()} Mode ===")

        env = DeliveryEnv(difficulty=level)
        state = env.reset()

        print("Initial State:", state)

        done = False

        while not done:
            action = random.randint(0, 2)
            state, reward, done = env.step(action)

            print(
                f"Action: {action} | Position: {state['agent_position']} | "
                f"Orders: {state['pending_orders']} | Reward: {reward:.2f} | Done: {done}"
            )

        print("✅ Episode Finished\n")