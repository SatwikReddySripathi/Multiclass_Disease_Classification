import json
import pandas as pd
import matplotlib.pyplot as plt
import os
file_path = os.path.abspath('monitoring/feedback.jsonl')
print(file_path)

# Load JSONL File
file_path = r"D:\MS\Sem3 - Fall 2024\MLOps\Multiclass_Disease_Classification\monitoring\monitoring\feedback.jsonl"
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Expand Predicted Labels for Per-Disease Analysis
df_exploded = df.explode('predicted_labels')
df_exploded['is_correct'] = df_exploded['feedback_type'] == 'correct'

# Disease-Level Metrics
metrics = df_exploded.groupby('predicted_labels').agg(
    correct_predictions=('is_correct', 'sum'),
    total_predictions=('predicted_labels', 'count')
).reset_index()
metrics['accuracy'] = metrics['correct_predictions'] / metrics['total_predictions']

# Display Disease-Level Metrics
import ace_tools as tools
tools.display_dataframe_to_user(name="Disease Prediction Metrics", dataframe=metrics)

# Visualization 1: Correct vs. Incorrect Predictions per Disease
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(metrics['predicted_labels'], metrics['correct_predictions'], label='Correct Predictions')
ax.bar(metrics['predicted_labels'], metrics['total_predictions'] - metrics['correct_predictions'],
       bottom=metrics['correct_predictions'], label='Incorrect Predictions')
ax.set_title('Correct and Incorrect Predictions per Disease')
ax.set_xlabel('Disease')
ax.set_ylabel('Number of Predictions')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Accuracy per Disease
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(metrics['predicted_labels'], metrics['accuracy'], marker='o', label='Accuracy', linestyle='-')
ax.set_title('Accuracy per Disease')
ax.set_xlabel('Disease')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='r', linestyle='--', label='Baseline (50%)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Additional Analysis 1: Gender-Based Analysis
gender_metrics = df_exploded.groupby(['gender', 'predicted_labels']).agg(
    correct_predictions=('is_correct', 'sum'),
    total_predictions=('predicted_labels', 'count')
).reset_index()
gender_metrics['accuracy'] = gender_metrics['correct_predictions'] / gender_metrics['total_predictions']
tools.display_dataframe_to_user(name="Gender-Based Disease Metrics", dataframe=gender_metrics)

# Visualization 3: Gender Accuracy by Disease
fig, ax = plt.subplots(figsize=(10, 6))
for gender, group in gender_metrics.groupby('gender'):
    ax.plot(group['predicted_labels'], group['accuracy'], marker='o', label=gender)
ax.set_title('Accuracy by Gender and Disease')
ax.set_xlabel('Disease')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.legend(title='Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Additional Analysis 2: Age Group Trends
# Categorize patients into age groups
bins = [0, 30, 50, 70, 100]
labels = ['0-30', '31-50', '51-70', '71-100']
df_exploded['age_group'] = pd.cut(df_exploded['age'], bins=bins, labels=labels, right=True)

# Calculate accuracy by age group
age_group_metrics = df_exploded.groupby('age_group').agg(
    correct_predictions=('is_correct', 'sum'),
    total_predictions=('age_group', 'count')
).reset_index()
age_group_metrics['accuracy'] = age_group_metrics['correct_predictions'] / age_group_metrics['total_predictions']
tools.display_dataframe_to_user(name="Age Group Metrics", dataframe=age_group_metrics)

# Visualization 4: Accuracy by Age Group
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(age_group_metrics['age_group'], age_group_metrics['accuracy'], color='skyblue', edgecolor='black')
ax.set_title('Accuracy by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
