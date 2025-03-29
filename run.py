from model import OpinionModel
import matplotlib.pyplot as plt
import pandas as pd

model = OpinionModel(num_agents=100)

for i in range(50):
    model.step()

results = model.datacollector.get_model_vars_dataframe()
results.plot()
plt.title("Opinion Dynamics Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.grid(True)
plt.show()