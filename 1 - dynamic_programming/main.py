from gridworld import GridWorld
from policy_iteration import policy_iteration
from value_iteration import value_iteration

if __name__ == "__main__":
    world = GridWorld(width=5, height=5)
    world.render()
    # policy = policy_iteration(world)
    policy = value_iteration(world)
    world.render_policy(policy)
        