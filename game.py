import gymnasium as gym

class LunarLanderEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()