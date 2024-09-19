from flask import Flask, render_template, request, redirect, url_for, make_response, send_file, session
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from datetime import timedelta
from pandas import Timedelta
import numpy as np
import datetime
import os
from collections import defaultdict
from flask_caching import Cache
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Random secret key for session management

# Configure file-based cache
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache-directory'  # Directory for storing cache files
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds

cache = Cache(app)

def cache_key():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique ID for the session
    return f"user_{session['user_id']}"

def parse_time(time_obj):
    if isinstance(time_obj, float) or isinstance(time_obj, int):
        return float(time_obj) * 24 * 60
    elif isinstance(time_obj, pd.Timestamp):
        base_date = pd.Timestamp('1899-12-31')
        delta = time_obj - base_date
        return delta.total_seconds() / 60
    elif isinstance(time_obj, datetime.datetime):
        base_date = datetime.datetime(1899, 12, 31)
        delta = time_obj - base_date
        return delta.total_seconds() / 60
    elif isinstance(time_obj, datetime.time):
        return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
    elif isinstance(time_obj, Timedelta):
        return time_obj.total_seconds() / 60
    elif isinstance(time_obj, str):
        try:
            parts = time_obj.split(':')
            if len(parts) == 3:
                return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2])).total_seconds() / 60
            elif len(parts) == 2:
                return timedelta(hours=int(parts[0]), minutes=int(parts[1])).total_seconds() / 60
            else:
                raise ValueError(f"Unsupported time format: {time_obj}")
        except ValueError:
            raise ValueError(f"Unsupported string format for time_str: {time_obj}")
    else:
        raise ValueError(f"Unsupported type for time_str: {type(time_obj)}")

@app.route('/', methods=['GET'])
def index():
    cache.delete(cache_key())  # Clear cache for the user
    return render_template('index.html')

@app.route('/plate_selection', methods=['POST'])
def plate_selection():
    uploaded_file = request.files['csv_file']
    plate_type = request.form['plateType']
    file_extension = uploaded_file.filename.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, header=0)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file, header=0)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", 400

        # Initialize a session-specific col_tracker
        col_tracker = defaultdict(int)

        # Apply the function to handle duplicate columns using this session-specific tracker
        df.columns = df.columns.to_series().apply(lambda col: handle_duplicates(col, col_tracker))
        df.columns = [remove_suffixes(col) for col in df.columns]

        if 'Time' not in df.columns:
            return "The uploaded file must contain a 'Time' column.", 400

        time_column = df['Time'].apply(parse_time)
        time_column = pd.to_numeric(time_column, errors='raise')

        df.drop(columns=['Time'], inplace=True)
        df.insert(0, 'Time', time_column)

        # Store the data using a user-specific cache key
        cache.set(f'data_{cache_key()}', df)
        cache.set(f'plate_type_{cache_key()}', plate_type)

        return redirect(url_for('edit', plate_type=plate_type))
    except (pd.errors.EmptyDataError, ValueError) as e:
        return f"Error processing file: {str(e)}. Please upload a valid CSV or Excel file.", 400


# Initialize a dictionary to track duplicate columns
col_tracker = defaultdict(int)

# Function to handle duplicate column names with a session-specific tracker
def handle_duplicates(col, col_tracker):
    if col == "Time":
        return col
    col_tracker[col] += 1
    if col_tracker[col] > 1:
        return f"{col}_{col_tracker[col] - 1}"
    return col

def remove_suffixes(col_name):
    import re
    return re.sub(r'\.\d+$', '', col_name)

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    plate_type = request.args.get('plate_type', default='96')
    use_labels = 'no'

    df = cache.get(f'data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    if request.method == 'POST':
        form_data = request.form.to_dict()
        use_labels = form_data.pop('use_labels', 'no')

        form_data.pop('Time', None)
        well_name_map = {k: v for k, v in form_data.items() if v.strip() != ''}

        if use_labels == 'yes':
            df = df.copy()
            well_name_map = None
        else:
            df = df.rename(columns=well_name_map)
            if 'Time' not in df.columns:
                raise ValueError("Time column not found in DataFrame after renaming.")
            columns_to_keep = ['Time'] + list(well_name_map.values())
            df = df[columns_to_keep]

        cache.set(f'renamed_data_{cache_key()}', df)
        cache.set(f'use_labels_{cache_key()}', use_labels)
        cache.set(f'well_name_map_{cache_key()}', well_name_map)

        return redirect(url_for('results'))

    well_data = df.iloc[:, 1:]
    max_values = well_data.max()

    if plate_type == '96':
        rows, cols = 8, 12
    elif plate_type == '384':
        rows, cols = 16, 24

    if len(max_values) != rows * cols:
        raise ValueError(f"Expected {rows * cols} wells, but got {len(max_values)} columns.")

    heatmap_data = max_values.values.reshape((rows, cols))
    x_labels = [f"{i+1}" for i in range(cols)]
    y_labels = [f"{chr(65+i)}" for i in range(rows)]

    if plate_type == '96':
        heatmap = go.Heatmap(
            z=heatmap_data,
            x=x_labels,
            y=y_labels,
            colorscale='Sunsetdark',
            xgap=0.5,
            ygap=0.5,
            hovertemplate='Column: %{x}<br>Row: %{y}<br>Value: %{z}<extra></extra>'
        )
        heatmap_fig = go.Figure(data=[heatmap])
        heatmap_fig.update_layout(
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=50, r=50, t=50, b=50),
            width=600,
            height=400
        )
    elif plate_type == '384':
        heatmap = go.Heatmap(
            z=heatmap_data,
            x=x_labels,
            y=y_labels,
            colorscale='Sunsetdark',
            xgap=1,
            ygap=1,
            hovertemplate='Column: %{x}<br>Row: %{y}<br>Value: %{z}<extra></extra>'
        )
        heatmap_fig = go.Figure(data=[heatmap])
        heatmap_fig.update_layout(
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=50, r=50, t=50, b=50),
            width=800,
            height=600
        )

    heatmap_filename = 'heatmap.html'
    pio.write_html(heatmap_fig, file=heatmap_filename, full_html=True)

    heatmap_div = pyo.plot(heatmap_fig, output_type='div', include_plotlyjs=True)

    well_names = list(df.columns[1:])
    if plate_type == '96':
        return render_template('edit_96.html', well_names=well_names, heatmap_graph=heatmap_div, use_labels=use_labels)
    elif plate_type == '384':
        return render_template('edit_384.html', well_names=well_names, heatmap_graph=heatmap_div, use_labels=use_labels)

@app.route('/results')
def results():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    named_wells = [col for col in df.columns if col.strip()]
    df = df[named_wells]
    graph = create_plot(df)
    graph_div = pyo.plot(graph, output_type='div', include_plotlyjs=True)
    download_filename = "interactive_optical_density_plot.html"
    return render_template('results.html', plotly_graph=graph_div, csv_filename="renamed_data.csv", download_filename=download_filename)

def string_to_pastel_color(sample_name):
    hash_val = 0
    for char in sample_name:
        hash_val = hash_val * 31 + ord(char)
        hash_val = hash_val ^ (hash_val << 10)
        hash_val = hash_val ^ (hash_val >> 15)
    hash_val = hash_val & 0xfffff
    hue = abs(hash_val) % 360
    saturation = 60
    lightness = 85
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def calculate_mean_std(df):
    df_melted = df.melt(id_vars=['Time'], var_name='Sample', value_name='OD')
    df_melted['Time'] = df_melted['Time'] / 60
    df_grouped = df_melted.groupby(['Sample', 'Time']).agg(
        mean_OD=('OD', 'mean'),
        std_OD=('OD', 'std')
    ).reset_index()
    return df_grouped

def create_plot(data):
    plot_data = calculate_mean_std(data)
    traces = []
    for sample in plot_data['Sample'].unique():
        sample_data = plot_data[plot_data['Sample'] == sample]
        color = string_to_pastel_color(sample)
        trace = go.Scatter(
            x=sample_data['Time']/60,
            y=sample_data['mean_OD'],
            error_y=dict(type='data', array=sample_data['std_OD'], visible=True, thickness=1),
            mode='lines+markers',
            name=sample,
            line=dict(color=color, width=1)
        )
        traces.append(trace)

    layout = go.Layout(
        title='Optical Density Readings',
        xaxis_title='time (hours)',
        yaxis_title='OD600nm',
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1000,
        height=750,
        xaxis=dict(
            tickvals=np.arange(0, max(plot_data['Time']) / 60 + 1, 2),
            ticktext=[int(hour) for hour in np.arange(0, max(plot_data['Time']) / 60 + 1, 2)]
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(plot_bgcolor='white',
                      xaxis=dict(color='black', tickcolor='black', showgrid=False, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black'),
                      yaxis=dict(color='black', tickcolor='black', showgrid=False, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black'))
    return fig

@app.route('/download_report')
def download_report():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    max_growth_rates = calculate_max_growth_rate_per_hour(df.copy())
    summary = group_replicates_and_calculate_mean_std(max_growth_rates)
    max_growth_rate_fig = plot_max_growth_rate(summary)
    max_growth_rate_html = pio.to_html(max_growth_rate_fig, full_html=False)

    report_html = f"""
    <html>
    <head>
        <title>Growth Analysis Report</title>
    </head>
    <body>
        <h1>Growth Analysis Report</h1>
        <h2>Max Growth Rate (Mean ± Std)</h2>
        {max_growth_rate_html}
    </body>
    </html>
    """

    report_filename = 'growth_analysis_report.html'
    with open(report_filename, 'w') as file:
        file.write(report_html)

    return send_file(report_filename, as_attachment=True)

@app.route('/download_heatmap/<string:filename>')
def download_heatmap(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return "File not found.", 404

@app.route('/download_interactive_plot/<string:filename>')
def download_interactive_plot(filename):
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400
    graph = create_plot(df)
    pio.write_html(graph, file=filename, full_html=True)
    return send_file(filename, as_attachment=True, download_name=filename)

@app.route('/download/<string:csv_filename>')
def download(csv_filename):
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    csv_data = df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename={csv_filename}'
    response.headers['Content-Type'] = 'text/csv'
    return response

#############################################################################################################
############################################################################################################
# Step 1: Assign standard well labels without suffixes
def assign_well_labels(df, plate_type):
    if 'Time' not in df.columns:
        raise ValueError("Time column not found in DataFrame.")
    
    # Exclude 'Time' column
    original_labels = [col for col in df.columns if col != 'Time']
    # Generate well labels
    if plate_type == '96':
        well_labels = [f"{chr(65 + i)}{j + 1}" for i in range(8) for j in range(12)]
    elif plate_type == '384':
        well_labels = [f"{chr(65 + i)}{j + 1}" for i in range(16) for j in range(24)]
    else:
        raise ValueError("Invalid plate type")
    
    # Map well labels to original labels
    label_mapping = dict(zip(well_labels[:len(original_labels)], original_labels))
    
    # Assign new columns
    new_columns = ['Time'] + well_labels[:len(original_labels)]
    if len(new_columns) != len(df.columns):
        raise ValueError(f"Length mismatch: DataFrame has {len(df.columns)} columns, new labels have {len(new_columns)} elements")
    
    df.columns = new_columns
    return df, label_mapping

# Step 2: Get sample labels from CSV or user input and map them to the well labels
def get_labels(df, use_labels, well_name_map=None):
    if use_labels == 'yes':
        labels = {well: sample for well, sample in zip(df.columns[1:], df.columns[1:])}
    else:
        labels = well_name_map or {}
    return labels

# Step 3: Replace well labels (A1, A2, etc.) with sample labels (e.g., sample1, sample2)
def replace_well_labels_with_sample_labels(df, labels):
    if labels:
        new_columns = ['Time'] + [labels.get(col, col) for col in df.columns[1:]]
        df.columns = new_columns
    return df

# Step 4: Calculate max growth rate per hour for each well/sample
def calculate_max_growth_rate_per_hour(df):
    col_tracker = defaultdict(int)

    def handle_duplicates(col):
        col_tracker[col] += 1
        if col_tracker[col] > 1:
            return f"{col}_{col_tracker[col] - 1}"
        return col

    df.columns = df.columns.to_series().apply(handle_duplicates)

    df_melted = pd.melt(df, id_vars=['Time'], var_name='Sample', value_name='OD')
    df_melted['Time'] = df_melted['Time'] / 60

    df_melted['OD_diff'] = df_melted.groupby('Sample')['OD'].diff()
    df_melted['Time_diff'] = df_melted.groupby('Sample')['Time'].diff()

    df_melted['Growth_rate'] = df_melted['OD_diff'] / df_melted['Time_diff']
    df_melted['Growth_rate'] = df_melted['Growth_rate'].replace([np.inf, -np.inf], np.nan).clip(lower=0)

    df_melted['Original_Sample'] = df_melted['Sample'].str.replace(r'_\d+$', '', regex=True)
    max_growth_rate_df = df_melted.groupby(['Original_Sample', 'Sample'])['Growth_rate'].max().reset_index()

    return max_growth_rate_df

# Step 5: Group replicates, calculate mean and std
def group_replicates_and_calculate_mean_std(max_growth_rate_df):
    summary = max_growth_rate_df.groupby('Original_Sample')['Growth_rate'].agg(['mean', 'std']).reset_index()
    summary.columns = ['Sample', 'Mean_growth_rate', 'Std_growth_rate']
    return summary

# Step 6: Plot the max growth rates (mean ± std)
def plot_max_growth_rate(summary):
    fig = go.Figure([go.Bar(
        x=summary['Sample'],
        y=summary['Mean_growth_rate'],
        error_y=dict(type='data', array=summary['Std_growth_rate'], visible=True),
        text=summary['Mean_growth_rate'],
        textposition='auto'
    )])

    fig.update_layout(
        title='Max Growth Rate (Mean ± Std)',
        xaxis_title='Sample',
        yaxis_title='Max Growth Rate (OD/hour)',
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=800,
        height=600
    )

    return fig

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
