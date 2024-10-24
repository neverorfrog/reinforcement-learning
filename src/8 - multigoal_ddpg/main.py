from fetch_agent import FetchAgent, launch, plot_tasks, Type 

if __name__ == "__main__":
    reach = 'FetchReach-v2'
    push = 'FetchPush-v2'
    pickandplace = 'FetchPickAndPlace-v2'
    launch(reach, Type.HER)
    plot_tasks(reach)