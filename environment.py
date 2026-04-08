class DeliveryEnv:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.size = 5 if difficulty == "easy" else 7 if difficulty == "medium" else 9
        self.agent_pos = 0
        self.orders = [self.size - 1]
        self.done = False
        self.steps = 0
        self.max_steps = self.size * 3

    def reset(self):
        self.agent_pos = 0
        self.orders = [self.size - 1]
        self.done = False
        self.steps = 0
        return self.state()

    def step(self, action):
        if self.done:
            return self.state(), 0, True

        self.steps += 1
        reward = -1

        if action == 0:  # move right
            if self.agent_pos < self.size - 1:
                self.agent_pos += 1
        elif action == 1:  # move left
            if self.agent_pos > 0:
                self.agent_pos -= 1
        elif action == 2:  # deliver
            if self.agent_pos in self.orders:
                self.orders.remove(self.agent_pos)
                reward = 10

        if len(self.orders) == 0:
            self.done = True
            reward += 20

        if self.steps >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done

    def state(self):
        return {
            "difficulty": self.difficulty,
            "grid_size": self.size,
            "agent_pos": self.agent_pos,
            "pending_orders": self.orders,
            "steps_taken": self.steps,
            "max_steps": self.max_steps,
        }
