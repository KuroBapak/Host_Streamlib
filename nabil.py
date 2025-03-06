import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "healthcare_dataset.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Convert column names to strings (in case of unexpected spaces)
df.columns = df.columns.str.strip()

# Convert categorical columns to string type
categorical_columns = ["Medical Condition", "Gender", "Admission Type", "Test Results"]
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Set seaborn style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 14))
axes = axes.flatten()

# 1. Age Distribution 
if "Age" in df.columns:
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Age Distribution of Patients")

# 2. Gender vs. Medical Condition
if "Medical Condition" in df.columns and "Gender" in df.columns:
    sns.countplot(x="Medical Condition", hue="Gender", data=df, ax=axes[1], palette="coolwarm")
    axes[1].set_title("Medical Conditions by Gender")
    axes[1].tick_params(axis='x', rotation=45)

# 3. Billing Amount Distribution
if "Billing Amount" in df.columns:
    sns.boxplot(y="Billing Amount", data=df, ax=axes[2], color="green")
    axes[2].set_title("Billing Amount Distribution")

# 4. Admission Type Probability
if "Admission Type" in df.columns:
    admission_probs = df["Admission Type"].value_counts(normalize=True)
    sns.barplot(x=admission_probs.index, y=admission_probs.values, ax=axes[3], palette="magma")
    axes[3].set_title("Probability Distribution of Admission Types")
    axes[3].set_ylabel("Probability")
    axes[3].tick_params(axis='x', rotation=45)  # ✅ FIXED rotation without warning

# 5. Test Result Probabilities
if "Test Results" in df.columns:
    test_results_probs = df["Test Results"].value_counts(normalize=True)
    sns.barplot(x=test_results_probs.index, y=test_results_probs.values, ax=axes[4], palette="viridis")
    axes[4].set_title("Probability Distribution of Test Results")
    axes[4].set_ylabel("Probability")
    axes[4].tick_params(axis='x', rotation=45)  # ✅ FIXED rotation without warning

# Remove extra subplot if not needed
fig.delaxes(axes[5])

# Show all visualizations
plt.tight_layout()
plt.show()