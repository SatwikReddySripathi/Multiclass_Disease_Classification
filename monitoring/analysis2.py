
from datetime import datetime
import os
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
directory_path = 'D:\MS\Sem3 - Fall 2024\MLOps\Multiclass_Disease_Classification\monitoring'

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
print(csv_files)



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





# Create a Dash app
app = dash.Dash(__name__)

# Ensure the timestamp column is in datetime format
final_df1 = final_df.copy(deep=True)
final_df1['timestamp'] = pd.to_datetime(final_df1['timestamp'])

# Sort by timestamp to ensure the data is in chronological order
final_df1 = final_df1.sort_values(by='timestamp')

# Get the unique diseases
diseases = final_df1['predicted_disease'].unique()

# Layout for the dashboard
app.layout = html.Div([
    html.H1("Disease Recall and F1 Score Improvement Over Time"),

    # Dropdown to select a disease
    dcc.Dropdown(
        id='disease-dropdown',
        options=[{'label': disease, 'value': disease} for disease in diseases],
        value=diseases[0],  # Default value (first disease)
        style={'width': '50%'}
    ),

    # Graph to show recall and F1 score
    dcc.Graph(id='disease-graph')
])

# Callback to update the graph based on selected disease
@app.callback(
    Output('disease-graph', 'figure'),
    [Input('disease-dropdown', 'value')]
)
def update_graph(selected_disease):
    # Filter the data for the selected disease
    disease_data = final_df1[final_df1['predicted_disease'] == selected_disease]

    # Plotting recall and f1_score over time for the selected disease
    fig = px.line(disease_data,
                  x='timestamp',
                  y=['recall', 'f1_score'],
                  title=f"{selected_disease} - Recall and F1 Score Improvement Over Time",
                  labels={'timestamp': 'Time', 'value': 'Score'},
                  line_shape='linear')

    # Get the min and max timestamp for the selected disease
    min_timestamp = disease_data['timestamp'].min()
    max_timestamp = disease_data['timestamp'].max()

    # Extend the timeline by 0.5 hours before and after the data range
    extended_min_timestamp = min_timestamp - timedelta(hours=0.5)
    extended_max_timestamp = max_timestamp + timedelta(hours=0.5)

    # Update x-axis to show a bigger scale with weekly ticks
    fig.update_xaxes(
        range=[extended_min_timestamp, extended_max_timestamp],
        tickmode='auto',  # Auto mode will dynamically determine the best ticks
        tickformat='%Y-%m-%d',  # Format the timestamp as 'YYYY-MM-DD'
        dtick="W1",  # Use weekly ticks for a bigger scale
        tickangle=45,  # Rotate the ticks for better readability
    )

    # Return the figure to be displayed
    return fig

# Run the Dash app
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