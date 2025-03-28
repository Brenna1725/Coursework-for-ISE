import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tabulate import tabulate

# Assume the CSV has exactly 10 rows (no header), each row is one algorithm's result (Random or Tabu) for a feature.
csv_file = 'raw data.csv'
df = pd.read_csv(csv_file, header=None)

# Make sure we have exactly 10 rows.
if len(df) != 10:
    raise ValueError(f"Expected 10 rows of data, but found {len(df)}.")

# Define names for 5 groups (3 for Adult, 2 for COMPAS).
group_names = [
    "ADULT - gender",
    "ADULT - race",
    "ADULT - age",
    "COMPAS - Race",
    "COMPAS - Sex"
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Prepare a table to store the test results.
results_table = [["Group", "U statistic", "p-value"]]

for i in range(5):
    # Row 2*i is for Random Search.
    data_random = df.iloc[2*i].dropna().values
    # Row 2*i + 1 is for Tabu Search.
    data_tabu = df.iloc[2*i + 1].dropna().values

    # Convert data to float.
    data_random = data_random.astype(float)
    data_tabu = data_tabu.astype(float)

    # Perform the two-sided Mann-Whitney U test.
    u_statistic, p_value = mannwhitneyu(data_random, data_tabu, alternative='two-sided')

    # Save the test results.
    results_table.append([group_names[i], f"{u_statistic:.3f}", f"{p_value:.3g}"])

    # Print results (optional).
    print(f"{group_names[i]}:")
    print(f"  U statistic = {u_statistic:.3f}")
    print(f"  p-value = {p_value:.3g}\n")

    # Plot boxplots for the two methods.
    axes[i].boxplot([data_random, data_tabu], labels=["Random Search", "Tabu Search"])
    axes[i].set_title(group_names[i])
    axes[i].set_ylabel("IDI Ratio")

plt.tight_layout()
plt.show()

# Print a summary table of the test results.
print("\nSummary of Mann-Whitney U Test:")
print(tabulate(results_table, headers="firstrow", tablefmt="pretty"))
