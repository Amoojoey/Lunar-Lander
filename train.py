import numpy as np
from game import LunarLanderEnv
from agent import DQNAgent

HYPERPARAMS = {
    "total_episodes": 2500,          
    "target_moving_avg": 240.0,      
    "max_steps_per_episode": 500,    

    "hidden_dim": 128,               

    "lr": 3e-4,                     
    "gamma": 0.99,                  
    "tau": 1e-3,                     
    "batch_size": 128,            
    "buffer_capacity": 100000,      

    "epsilon_start": 1.0,            
    "epsilon_min": 0.01,             
    "epsilon_decay": 0.995           
}

def train():
    #Main training loop for DQN agent.

    # Create environment 
    env = LunarLanderEnv(render_mode=None) 
    
    # Initialize agent using hyperparameters dictionary
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=HYPERPARAMS["hidden_dim"],
        lr=HYPERPARAMS["lr"],
        gamma=HYPERPARAMS["gamma"],
        tau=HYPERPARAMS["tau"],
        buffer_capacity=HYPERPARAMS["buffer_capacity"]
    )

    epsilon = HYPERPARAMS["epsilon_start"]
    scores_history = []
    
    print(f"--- Starting DQN Training on device: {agent.device} ---")
    print(f"Target 100-Episode Moving Average: {HYPERPARAMS['target_moving_avg']}")
    print(f"Max Steps per Episode: {HYPERPARAMS['max_steps_per_episode']}\n")

    for episode in range(1, HYPERPARAMS["total_episodes"] + 1):
        state = env.reset(seed=42 + episode)
        total_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # 1. Select action 
            action = agent.select_action(state, epsilon)
            
            # 2. Step environment with selected action
            next_state, reward, done = env.step(action)
            step_count += 1
            
            # 3. Truncate episode if maximum step limit is reached
            if step_count >= HYPERPARAMS["max_steps_per_episode"]:
                done = True
            
            # 4. Store transition in replay memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # 5.gradient descent update step
            agent.update(batch_size=HYPERPARAMS["batch_size"])
            
            # 6. Soft update target network parameters
            agent.soft_update_target_network()
            
            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(HYPERPARAMS["epsilon_min"], epsilon * HYPERPARAMS["epsilon_decay"])
        scores_history.append(total_reward)
        
        # Calculate moving average
        moving_avg = np.mean(scores_history[-100:])

        # Print progress every 10 episodes
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode:4d}/{HYPERPARAMS['total_episodes']} | "
                  f"Score: {total_reward:6.2f} | "
                  f"100-Ep Moving Avg: {moving_avg:6.2f} | "
                  f"Epsilon: {epsilon:.3f}")

        # stopping condition
        if moving_avg >= HYPERPARAMS["target_moving_avg"] and episode >= 100:
            print(f"\nTarget achieved! High performance model converged in {episode} episodes!")
            print(f"Final 100-Episode Moving Average: {moving_avg:.2f}")
            break

    # Save final model weights
    agent.save_weights("weights.pth")
    print("\nTraining completed successfully. Weights saved to 'weights.pth'.")
    env.close()

if __name__ == "__main__":
    train()