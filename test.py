import time
import numpy as np
from game import LunarLanderEnv
from agent import DQNAgent
from train import HYPERPARAMS  # Import configuration to ensure architecture matches

def test(render: bool = True, test_episodes: int = 10):
    # Set render mode: 'human' for visual window, None for fast execution
    render_mode = "human" if render else None
    env = LunarLanderEnv(render_mode=render_mode)
    
    # Initialize agent using central hyperparameters from train.py
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=HYPERPARAMS["hidden_dim"],
        lr=HYPERPARAMS["lr"],
        gamma=HYPERPARAMS["gamma"],
        tau=HYPERPARAMS["tau"],
        buffer_capacity=HYPERPARAMS["buffer_capacity"]
    )
    
    # Load trained model weights from disk
    try:
        agent.load_weights("weights.pth")
        print("Successfully loaded trained weights from 'weights.pth'.")
    except FileNotFoundError:
        print("Error: 'weights.pth' not found! Please run train.py first.")
        return

    # Set policy network to evaluation mode
    agent.policy_net.eval()
    print(f"\n--- Starting Randomized Evaluation ({test_episodes} Episodes) ---")
    
    scores = []
    success_count = 0

    for ep in range(1, test_episodes + 1):
        # Reset environment without a fixed seed to get random initial positions each time
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Always select the best action greedily (epsilon = 0.0)
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            state = next_state
            
            # Small delay for smooth rendering animation
            if render:
                time.sleep(0.01)
                
        scores.append(total_reward)
        if total_reward >= 200.0:
            success_count += 1
            
        status = "SUCCESS" if total_reward >= 200 else "FAILED"
        print(f"Test Episode {ep:2d} | Reward: {total_reward:6.2f} | {status}")

    # Calculate final statistical metrics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    success_rate = (success_count / test_episodes) * 100

    print("\n--- Final Evaluation Metrics ---")
    print(f"Average Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Success Rate:  {success_rate:.1f}% ({success_count}/{test_episodes})")
    
    env.close()

if __name__ == "__main__":
    # Set render=True to watch the landing, or render=False for fast headless evaluation
    test(render=True, test_episodes=10)