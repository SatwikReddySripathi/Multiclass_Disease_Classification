import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google.cloud import storage
import json
import glob
import io
import pytz
import os

def get_access_token():
    SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
    credentials = Credentials.from_service_account_file('D:/MS/Sem3 - Fall 2024/MLOps/Multiclass_Disease_Classification/application_deployed/secret_key.json', scopes=SCOPES)
    credentials.refresh(Request())
    return credentials.token
access_token = get_access_token()


def initialize_storage(bucket_name):
    """Initialize GCS client and get bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket

bucket_name = "nih-dataset-mlops"
bucket = initialize_storage(bucket_name)

# List all blobs (files) in the bucket
blobs = bucket.list_blobs()

# Print the file names (blobs) in the bucket
# for blob in blobs:
#     print(blob.name)
file_path = 'feedback/feedback.jsonl'

# Get the blob (file) from the bucket
blob = bucket.blob(file_path)

# Download the file content as a string
file_content = blob.download_as_text()





# import json
# Read the content of the JSONL file
def process_jsonl_data(file_content):
    data = []
    for line in file_content.splitlines():
        data.append(json.loads(line))
    return data

# Process the data
data = process_jsonl_data(file_content)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)
print(df)

# Expand the predicted labels, true labels, and confidence scores into separate rows for easier analysis
expanded_data = []
for index, row in df.iterrows():
    for i, label in enumerate(row['predicted_labels']):

        # if not isinstance(row['true_labels'][i], list):
        #     true_label = [row['true_labels'][i]]
        # else:
        #     true_label = row['true_labels'][i]

        expanded_data.append({
            'image_name': row['image_name'],
            'age': row['age'],
            'gender': row['gender'],
            'predicted_label': label,
            # 'true_label': true_label,
            'confidence_score': row['confidence_score'][i],
            'feedback_type': row['feedback_type']
        })

expanded_df = pd.DataFrame(expanded_data)
print(expanded_df)

# Additional Metrics Calculation
# 1. Overall Inference Count and Count for Every Disease
overall_inference_count = expanded_df['predicted_label'].value_counts()
print(overall_inference_count)
print('Overall Inference: ',len(expanded_df))

# 2. Correct and Incorrect Inferences for Every Disease
# expanded_df['is_correct'] = expanded_df[expanded_df['predicted_label'] == expanded_df['true_label']]
# Group by predicted_label and feedback_type, and count the occurrences
correct_inference_count = expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'].groupby('predicted_label').size()
incorrect_inference_count = expanded_df[expanded_df['feedback_type'] == 'Incorrect_Predictions'].groupby('predicted_label').size()

print("Correct Inference Count:")
print(correct_inference_count)

print("\nIncorrect Inference Count:")
print(incorrect_inference_count)


# 3. Confidence Scores for Correct Inferences
correct_confidence_scores = expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'].groupby('predicted_label')['confidence_score'].mean()





# 1. Create a function to calculate precision, recall, and F1 score
def calculate_metrics(df):
    metrics = {}

    # Get the unique predicted labels (diseases)
    predicted_labels = df['predicted_label'].unique()

    for label in predicted_labels:
        # Get True Positives (TP), False Positives (FP), False Negatives (FN)
        tp = len(df[(df['predicted_label'] == label) & (df['feedback_type'] == 'Correct_Predictions')])
        fp = len(df[(df['predicted_label'] == label) & (df['feedback_type'] == 'Incorrect_Predictions')])
        fn = len(df[(df['predicted_label'] != label) & (df['feedback_type'] == 'Incorrect_Predictions')])

        # Compute precision, recall, and F1 score for each label
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        overall_inference_count = len(df[df['predicted_label'] == label])
        # print(overall_inference_count)
        # print('Overall Inference: ',len(expanded_df))


        correct_inference_count = len(df[(df['feedback_type'] == 'Correct_Predictions') & (df['predicted_label'] == label)])
        incorrect_inference_count = len(df[(df['feedback_type'] == 'Incorrect_Predictions') & (df['predicted_label'] == label)])
        avg_correct_inference_confidence = df[(df['feedback_type'] == 'Correct_Predictions') & (df['predicted_label'] == label)].confidence_score.mean()
        # avg_incorrect_inference_count = df[(df['feedback_type'] == 'Incorrect_Predictions') & (df['predicted_label'] == label)].confidence_score.mean()

        # print("Correct Inference Count:")
        # print(correct_inference_count)

        # print("\nIncorrect Inference Count:")
        # print(incorrect_inference_count)

        # Store the metrics
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'overall_inference_count': overall_inference_count,
            'correct_inference_count': correct_inference_count,
            'incorrect_inference_count': incorrect_inference_count,
            'avg_correct_inference_confidence': avg_correct_inference_confidence,
            # 'avg_incorrect_inference_count': avg_incorrect_inference_count
        }

    metrics['total'] = {
        'precision': None,
        'recall': None,
        'f1_score': None,
        'overall_inference_count': len(df),
        'correct_inference_count': len(df[df['feedback_type'] == 'Correct_Predictions']),
        'incorrect_inference_count': len(df[df['feedback_type'] == 'Incorrect_Predictions']),
        'avg_correct_inference_confidence': None,
        # 'avg_incorrect_inference_count': avg_incorrect_inference_count
    }

    return metrics

# Calculate metrics
metrics = calculate_metrics(expanded_df)

# Convert metrics to DataFrame for easier visualization
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

# Display the metrics for each disease
print(metrics_df)



from datetime import datetime
import pytz
est_timezone = pytz.timezone('US/Eastern')
timestamp = datetime.now(est_timezone).strftime("%Y-%m-%d_%H-%M-%S")
# print(timestamp)
metrics_df.to_csv(f'D:/MS/Sem3 - Fall 2024/MLOps/Multiclass_Disease_Classification/monitoring/metrics_{timestamp}.csv')

bucket_name = 'nih-dataset-mlops'
destination_blob_name = f'model_monitoring/metrics_{timestamp}.csv'


csv_buffer = io.StringIO()
metrics_df.to_csv(csv_buffer, index=False)
# Rewind the buffer's cursor to the start
csv_buffer.seek(0)

# Create a blob (file object) and upload the in-memory CSV to GCS
blob = bucket.blob(destination_blob_name)
blob.upload_from_file(csv_buffer, content_type='text/csv')

print(f"File uploaded to gs://{bucket_name}/{destination_blob_name}")






# Create Dash app
app = dash.Dash(__name__)

# Layout with Tabs
app.layout = html.Div([
    html.H1("Model Performance Dashboards"),

    # Tabs
    dcc.Tabs([
        # First Tab: Overall Inference and Correct/Incorrect Inferences
        dcc.Tab(label='Overall Inference Metrics', children=[
            html.H2("Overall Model Inference Metrics"),
            dcc.Graph(
                id='disease-inference-count',
                figure=px.bar(overall_inference_count, title="Overall Inference Count", labels={'index': 'Disease', 'value': 'Inference Count'})
            ),
            dcc.Graph(
                id='correct-incorrect-inferences',
                figure=px.bar(
                    pd.DataFrame({
                        'correct': correct_inference_count,
                        'incorrect': incorrect_inference_count
                    }).reset_index().melt(id_vars='predicted_label'),
                    x='predicted_label', y='value', color='variable', title="Correct vs Incorrect Inferences (Based on Feedback)"
                )
            ),
            dcc.Graph(
                id='confidence-scores',
                figure=px.box(
                    expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'], x='predicted_label', y='confidence_score',
                    title="Confidence Scores for Correct Inferences"
                )
            )
        ]),

        # Second Tab: Disease-Specific Metrics
        dcc.Tab(label='Disease Performance', children=[
            html.H2("Select Disease to View Metrics"),
            dcc.Dropdown(
                id='disease-dropdown',
                options=[{'label': disease, 'value': disease} for disease in metrics_df.index[:-1]],
                value='Pleural_Thickening',  # Default value
                style={'width': '50%'}
            ),
            html.Div(id='disease-dashboard')
        ])
    ])
])

# Callback to update the disease dashboard based on selected disease
@app.callback(
    Output('disease-dashboard', 'children'),
    [Input('disease-dropdown', 'value')]
)
def update_dashboard(disease):
    disease_data = metrics_df.loc[disease]

    # Create the disease-specific dashboard
    return html.Div([
        html.H2(f"Metrics for {disease}"),
        dash_table.DataTable(
            columns=[
                {"name": "Metric", "id": "metric"},
                {"name": "Value", "id": "value"}
            ],
            data=[
                {"metric": "Precision", "value": disease_data['precision']},
                {"metric": "Recall", "value": disease_data['recall']},
                {"metric": "F1 Score", "value": disease_data['f1_score']},
                {"metric": "Overall Inference Count", "value": disease_data['overall_inference_count']},
                {"metric": "Correct Inference Count", "value": disease_data['correct_inference_count']},
                {"metric": "Incorrect Inference Count", "value": disease_data['incorrect_inference_count']}
            ],
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        # Bar chart for Correct vs Incorrect Inferences
        dcc.Graph(
            id=f'correct-incorrect-{disease}',
            figure={
                'data': [
                    go.Bar(
                        x=['Correct', 'Incorrect'],
                        y=[disease_data['correct_inference_count'], disease_data['incorrect_inference_count']],
                        name=f'Inferences for {disease}',
                        marker=dict(color=['green', 'red'])
                    )
                ],
                'layout': go.Layout(
                    title=f'Correct and Incorrect Inferences for {disease}',
                    xaxis={'title': 'Inference Type'},
                    yaxis={'title': 'Count'}
                )
            }
        )
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

