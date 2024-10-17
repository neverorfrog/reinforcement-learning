from gridworld import GridWorld
from policy_iteration import policy_iteration
from value_iteration import value_iteration
import time

if __name__ == "__main__":
    world = GridWorld(width=100, height=100)
    # world.render()
    now = time.time()
    # policy = policy_iteration(world)
    policy = value_iteration(world)
    elapsed = time.time() - now
    print(elapsed)
    # world.render_policy(policy)
        