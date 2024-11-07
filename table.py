import matplotlib.pyplot as plt
import pandas as pd

# Define the data for the table
data = {
    'Method': ['NSGA-III', 'Tabu Search', 'Simulated Annealing'],
    'F1_Mean': [2594.69, 1898.08, 3969.07],
    'F1_Std': [467.4, 460.81, 925.0],
    'F2_Mean': [344.01, 344.02, 344.02],
    'F2_Std': [0.0, 0.0, 0.01]
}

# Create a DataFrame
table_df = pd.DataFrame(data)

# Set the Method column as the index
table_df.set_index('Method', inplace=True)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=table_df.values,
                 rowLabels=table_df.index,
                 colLabels=table_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)  # Set font size
table.scale(1, 1.5)  # Adjust scaling as necessary

# Title for the table
plt.title('Table 1: Mean and Standard Deviation of Objective Functions', fontsize=12)
plt.savefig('Objective_Functions_Metrics.png', bbox_inches='tight', dpi=300)  # Save the table as an image
plt.show()
