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





"""
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

"""


import glob
import io

# Function to extract timestamp from the filename
def extract_timestamp_from_filename(filename):
    # Assuming timestamp is part of the filename in the format "filename_YYYY-MM-DD_HH-MM-SS.csv"
    # print(filename)
    base_name = os.path.basename(filename)
    timestamp_str = base_name.split('_')[1] + '_' + base_name.split('_')[-1].replace('.csv', '')  # Assuming the timestamp is the last part before ".csv"

    print(filename)
    print(base_name)
    print(timestamp_str)
    print(datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S"))
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

# Path to the directory containing your CSV files
directory_path = '/content/drive/MyDrive/MLOps_testing/monitoring'

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
print(csv_files)



access_token = get_access_token()
bucket_name = 'nih-dataset-mlops'
bucket = initialize_storage(bucket_name)

# List all blobs (files) in the bucket
path_prefix = 'model_monitoring'
blobs = bucket.list_blobs(prefix=path_prefix)

# Initialize an empty list to store DataFrames
dfs = []
df = pd.DataFrame()

# Print the file names (blobs) in the bucket
for blob in blobs:
    if blob.name.endswith('.csv'):
      print(blob.name)  # Check if it's a CSV file
      # Download the content of the file as text
      file_content = blob.download_as_text()

      # Use StringIO to read the content into a pandas DataFrame
      data = pd.read_csv(io.StringIO(file_content))

      # Optionally, add a column for the filename
      data['filename'] = blob.name
      timestamp = extract_timestamp_from_filename(blob.name)
      data['timestamp'] = timestamp
      df = pd.concat([df,data], ignore_index=True)
      dfs.append(df)

      # Now you can work with the DataFrame `data` directly, for example:
      print(f"Data from {blob.name}:")
      print(data.head())
# print(dfs)
# # Concatenate all DataFrames into a single DataFrame
# final_df = pd.concat(dfs, ignore_index=True)

# # Optionally, you can save the combined DataFrame to a new CSV file
# # final_df.to_csv('/path/to/save/combined_data.csv', index=False)

# # Display the first few rows of the combined DataFrame
# final_df.columns = ['predicted_disease', 'precision', 'recall', 'f1_score',
#        'overall_inference_count', 'correct_inference_count',
#        'incorrect_inference_count','avg_correct_inference_confidence', 'filename', 'timestamp']
# # print(final_df.head())
# print(final_df)


final_df = df.copy(deep=True)

final_df = final_df.rename(columns={'Unnamed: 0': 'predicted_disease'})
print(final_df)



from datetime import timedelta

# Assuming 'final_df' is the DataFrame containing your combined data

# Ensure the timestamp column is in datetime format
final_df1 = final_df.copy(deep=True)
final_df1['timestamp'] = pd.to_datetime(final_df1['timestamp'], format='%Y-%m-%d %H:%M:%S')

# Sort by timestamp to ensure the data is in chronological order
final_df1 = final_df1.sort_values(by='timestamp')

# Get the unique diseases
diseases = final_df1['predicted_disease'].unique()
print(final_df1)


# Loop through each disease and plot the graph
for disease in diseases:
    # Filter the data for the current disease
    disease_data = final_df1[final_df1['predicted_disease'] == disease]
    print(disease_data)

# import pandas as pd
# import plotly.express as px
from datetime import timedelta

# Assuming 'final_df' is the DataFrame containing your combined data

# Ensure the timestamp column is in datetime format
final_df1 = final_df.copy(deep=True)
final_df1['timestamp'] = pd.to_datetime(final_df1['timestamp'], format='%Y-%m-%d %H:%M:%S')

# Sort by timestamp to ensure the data is in chronological order
final_df1 = final_df1.sort_values(by='timestamp')

# Get the unique diseases
diseases = final_df1['predicted_disease'].unique()

# Loop through each disease and plot the graph
for disease in diseases:
    # Filter the data for the current disease
    disease_data = final_df1[final_df1['predicted_disease'] == disease]

    # Plotting recall and f1_score over time for the current disease
    fig = px.line(disease_data,
                  x='timestamp',
                  y=['recall', 'f1_score'],
                  title=f"{disease} - Recall and F1 Score Improvement Over Time",
                  labels={'timestamp': 'Time', 'value': 'Score'},
                  line_shape='linear')

    min_timestamp = disease_data['timestamp'].min()
    max_timestamp = disease_data['timestamp'].max()

    extended_min_timestamp = min_timestamp - timedelta(hours=0.5)  # 30 days in the past
    extended_max_timestamp = max_timestamp + timedelta(hours=0.5)

    # Update x-axis to show bigger scale by spacing out the ticks
    fig.update_xaxes(
        range=[extended_min_timestamp, extended_max_timestamp],
        tickmode='auto',  # Auto mode will dynamically determine the best ticks
        tickformat='%Y-%m-%d',  # Format the timestamp as 'YYYY-MM-DD'
        dtick="W1",  # Use weekly ticks for a bigger scale
        tickangle=45,  # Rotate the ticks for better readability
    )

    # Show the plot
    fig.show()
    final_df1['timestamp']





import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from datetime import timedelta

# Assuming 'final_df' is the DataFrame containing your combined data and 'metrics_df' is available
# Also assuming final_df contains 'predicted_disease', 'timestamp', 'recall', and 'f1_score'

# Create a Dash app
app = dash.Dash(__name__)

# Ensure the timestamp column is in datetime format for final_df
final_df1 = final_df.copy(deep=True)
final_df1['timestamp'] = pd.to_datetime(final_df1['timestamp'])
final_df1 = final_df1.sort_values(by='timestamp')  # Sort by timestamp

# Get unique diseases from final_df
diseases = final_df1['predicted_disease'].unique()
# diseases = [disease for disease in final_df1['predicted_disease'].unique() if disease is not None and disease != '']

# Layout for the app
app.layout = html.Div([
    html.H1("Model Performance Dashboards"),

    # Tabs
    dcc.Tabs([
        # First Tab: Overall Inference and Correct/Incorrect Inferences
        dcc.Tab(label='Overall Inference Metrics', children=[
            html.H2("Overall Model Inference Metrics"),
            # Add your existing charts for overall metrics here
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
        # dcc.Tab(label='Disease Performance', children=[
        #     html.H2("Select Disease to View Metrics"),
        #     dcc.Dropdown(
        #         id='disease-dropdown',
        #         options=[{'label': disease, 'value': disease} for disease in diseases],
        #         value=diseases[0],  # Default value (first disease)
        #         style={'width': '50%'}
        #     ),
        #     html.Div(id='disease-dashboard')
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

    # Filter the data for the selected disease for recall and f1 score over time graph
    disease_time_data = final_df1[final_df1['predicted_disease'] == disease]

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
        ),
        # Add the Recall and F1 Score Improvement Over Time Graph
        dcc.Graph(
            id=f'disease-recall-f1-{disease}',
            figure=px.line(
                disease_time_data,
                x='timestamp',
                y=['recall', 'f1_score'],
                title=f"{disease} - Recall and F1 Score Improvement Over Time",
                labels={'timestamp': 'Time', 'value': 'Score'},
                line_shape='linear'
            ).update_xaxes(
                range=[disease_time_data['timestamp'].min() - timedelta(hours=0.5), disease_time_data['timestamp'].max() + timedelta(hours=0.5)],
                tickmode='auto',
                tickformat='%Y-%m-%d',
                dtick="W1",
                tickangle=45,
            )
        )
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# 3. Confidence Scores for Correct Inferences
correct_confidence_scores = expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'].groupby('predicted_label')['confidence_score'].mean()

# import plotly.express as px
# fig = px.box(data, x='predicted_label', y='confidence_score', title="Confidence Scores for Correct Predictions")

figure=px.box(
    expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'], x='predicted_label', y='confidence_score',
    title="Confidence Scores for Correct Inferences"
)
figure.show()

fig = px.histogram(expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'], x='predicted_label', nbins=20, title="Distribution of Confidence Scores")
fig.show()

# fig = px.density_contour(expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions'], x='predicted_label', title="Confidence Score Density")
data = expanded_df[expanded_df['feedback_type'] == 'Correct_Predictions']
fig = px.violin(data, x='predicted_label', y='confidence_score', box=True, points="all", title="Violin Plot of Confidence Scores")

fig.show()

fig = px.scatter(data, x='predicted_label', y='confidence_score', color='feedback_type', title="Confidence Score vs Predicted Label")
fig.show()
