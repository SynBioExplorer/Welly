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
from scipy.integrate import trapz
import hashlib
import re
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Random secret key for session management

# Configure file-based cache
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache-directory'  # Directory for storing cache files
app.config['CACHE_DEFAULT_TIMEOUT'] = 1000  # Cache timeout in seconds

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
            width=800,
            height=600
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

# app.py


@app.route('/results', methods=['GET', 'POST'])
def results():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return redirect(url_for('upload'))

    # Get all sample names (excluding 'Time')
    samples = [col for col in df.columns if col != 'Time']

    # Create a sorted list of unique sample names
    unique_samples = sorted(set(samples))

    # Retrieve existing sample colors or initialize with pastel colors
    sample_colors = cache.get(f'sample_colors_{cache_key()}') or {}

    # Ensure all unique samples have colors assigned
    for sample in unique_samples:
        if sample not in sample_colors or not sample_colors[sample]:
            sample_colors[sample] = string_to_pastel_color(sample)
    cache.set(f'sample_colors_{cache_key()}', sample_colors)

    # Map sanitized sample names to original sample names
    sanitized_to_original = {}
    for sample in unique_samples:
        sanitized_sample = re.sub(r'[^\w]', '_', sample)
        sanitized_to_original[f'sample_color_{sanitized_sample}'] = sample

    if request.method == 'POST':
        # Process color selections
        form_data = request.form.to_dict()
        for key, value in form_data.items():
            if key in sanitized_to_original:
                sample = sanitized_to_original[key]
                sample_colors[sample] = value.strip()
        # Update the cache with new colors
        cache.set(f'sample_colors_{cache_key()}', sample_colors)

    # Generate the Plotly graph with the current colors
    fig = create_plot(df, sample_colors)
    plotly_graph = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('results.html', plotly_graph=plotly_graph, samples=samples,
                           unique_samples=unique_samples, sample_colors=sample_colors)


def string_to_pastel_color(s):
    # Generate a hash of the string
    hash_object = hashlib.md5(s.encode())
    hash_int = int(hash_object.hexdigest()[:6], 16)
    # Extract RGB components
    r = (hash_int & 0xFF0000) >> 16
    g = (hash_int & 0x00FF00) >> 8
    b = (hash_int & 0x0000FF)
    # Mix with white to get pastel colors
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    # Return as hex color code
    return f'#{r:02X}{g:02X}{b:02X}'

def calculate_mean_std(df):
    df_melted = df.melt(id_vars=['Time'], var_name='Sample', value_name='OD')
    df_grouped = df_melted.groupby(['Sample', 'Time']).agg(
        mean_OD=('OD', 'mean'),
        std_OD=('OD', 'std')
    ).reset_index()
    return df_grouped

def create_plot(df, sample_colors):
    data_columns = [col for col in df.columns if col != 'Time']
    time_values = df['Time'] / 60  # Convert time to hours if necessary

    melted_df = df.melt(id_vars=['Time'], value_vars=data_columns, var_name='Sample', value_name='OD')
    melted_df['Time'] = melted_df['Time'] / 60  # Convert time to hours

    group_df = melted_df.groupby(['Time', 'Sample']).agg({'OD': ['mean', 'std']}).reset_index()
    group_df.columns = ['Time', 'Sample', 'mean_OD', 'std_OD']

    traces = []
    for sample in group_df['Sample'].unique():
        sample_data = group_df[group_df['Sample'] == sample]
        color = sample_colors.get(sample, string_to_pastel_color(sample))

        trace = go.Scatter(
            x=sample_data['Time'],
            y=sample_data['mean_OD'],
            error_y=dict(type='data', array=sample_data['std_OD'], visible=True, thickness=1),
            mode='lines+markers',
            name=sample,
            line=dict(color=color, width=1),
            marker=dict(color=color),
            text=[f"Time: {t:.3f} hours<br>OD: {val:.3f}" for t, val in zip(sample_data['Time'], sample_data['mean_OD'])],
            hoverinfo='text'
        )
        traces.append(trace)

    # Layout settings with tick marks
    layout = go.Layout(
        title='Optical Density Readings',
        xaxis_title='Time (hours)',
        yaxis_title='OD600nm',
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1000,
        height=750,
        xaxis=dict(
            color='black',
            tickcolor='black',
            ticks='outside',   # Display tick marks outside the axis
            ticklen=5,         # Length of the tick marks
            tickwidth=2,       # Width of the tick marks
            tickfont=dict(size=12),  # Font size for tick labels
            showgrid=False,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True        # Mirror axis lines and ticks on opposite side
        ),
        yaxis=dict(
            color='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5,
            tickwidth=2,
            tickfont=dict(size=12),
            showgrid=False,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig





####################### Max growth rate calculation #######################

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
def plot_max_growth_rate(summary, sample_colors):
    summary['Mean_growth_rate'] = summary['Mean_growth_rate'].round(3)
    summary['Std_growth_rate'] = summary['Std_growth_rate'].round(3)
    
    # Create a list of colors for the samples
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Sample']]
    
    fig = go.Figure([go.Bar(
        x=summary['Sample'],
        y=summary['Mean_growth_rate'],
        error_y=dict(type='data', array=summary['Std_growth_rate'], visible=True, thickness=1.5, color='black'),
        text=summary['Mean_growth_rate'],
        textposition='auto',
        marker=dict(color=colors)
    )])

    fig.update_layout(
        title=dict(
            text='Max Growth Rate (Mean ± Std)',
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, family='Times New Roman')
        ),
        xaxis=dict(
            title='Sample',
            titlefont=dict(size=16, family='Times New Roman'),
            tickfont=dict(size=14, family='Times New Roman'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        ),
        yaxis=dict(
            title='Max Growth Rate (OD/hour)',
            titlefont=dict(size=16, family='Times New Roman'),
            tickfont=dict(size=14, family='Times New Roman'),
            linecolor='black',
            linewidth=2,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig





####################### Area under curve calculation #######################

def calculate_auc(df):
    """
    Calculate the Area Under the Curve (AUC) for each well in the DataFrame.
    """
    auc_results = {}
    time_values = df['Time'].values  # Get time values once

    for idx in range(len(df.columns)):
        column = df.columns[idx]
        if column == 'Time':
            continue

        # Access the column by position to avoid duplicate column name issues
        od_values = df.iloc[:, idx].values

        # Ensure time_values and od_values are the same length
        if len(time_values) != len(od_values):
            print(f"Length mismatch in column {column}. Skipping this well.")
            continue

        # Remove NaNs from both arrays
        valid_indices = ~np.isnan(od_values) & ~np.isnan(time_values)
        od_values_clean = od_values[valid_indices]
        time_values_clean = time_values[valid_indices]

        # Ensure arrays are one-dimensional
        od_values_clean = od_values_clean.flatten()
        time_values_clean = time_values_clean.flatten()

        # Calculate AUC using the trapezoidal rule
        try:
            auc = trapz(od_values_clean, time_values_clean)
            # Since column names may not be unique, create a unique key for each column
            unique_col_name = f"{column}_{idx}"
            auc_results[unique_col_name] = {'Sample': column, 'AUC': auc}
        except ValueError as e:
            print(f"Error calculating AUC for column {column}: {e}")
            continue

    return auc_results




def group_auc_by_sample(df, auc_results):
    """
    Group AUC by sample names and calculate mean and standard deviation.
    """
    # Convert AUC results into a DataFrame
    auc_df = pd.DataFrame.from_dict(auc_results, orient='index')

    # Now, 'Sample' and 'AUC' are columns in auc_df
    # Use 'Sample' for grouping

    # Use the sample names directly for grouping
    auc_df['Original_Sample'] = auc_df['Sample']

    # Proceed with grouping
    summary = auc_df.groupby('Original_Sample').agg(
        Mean_AUC=('AUC', 'mean'),
        Std_AUC=('AUC', 'std'),
        Replicate_Count=('AUC', 'count')
    ).reset_index()

    # Round Mean_AUC and Std_AUC
    summary['Mean_AUC'] = summary['Mean_AUC'].round(3)
    summary['Std_AUC'] = summary['Std_AUC'].fillna(0).round(3)

    return summary





def plot_auc(summary, sample_colors):
    summary['Mean_AUC'] = summary['Mean_AUC'].round(3)
    summary['Std_AUC'] = summary['Std_AUC'].round(3)

    # Create a list of colors for the samples
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Original_Sample']]

    fig = go.Figure([go.Bar(
        x=summary['Original_Sample'],
        y=summary['Mean_AUC'],
        error_y=dict(type='data', array=summary['Std_AUC'], visible=True, thickness=1.5, color='black'),
        text=summary['Mean_AUC'],
        textposition='auto',
        marker=dict(color=colors)
    )])

    fig.update_layout(
        title=dict(
            text='Area Under the Curve (Mean ± Std)',
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, family='Times New Roman')
        ),
        xaxis=dict(
            title='Sample',
            titlefont=dict(size=16, family='Times New Roman'),
            tickfont=dict(size=14, family='Times New Roman'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        ),
        yaxis=dict(
            title='AUC',
            titlefont=dict(size=16, family='Times New Roman'),
            tickfont=dict(size=14, family='Times New Roman'),
            linecolor='black',
            linewidth=2,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig






####################### download stuff #######################


@app.route('/download_report')
def download_report():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    # Retrieve sample_colors
    sample_colors = cache.get(f'sample_colors_{cache_key()}') or {}

    # Calculate the AUC for each well
    auc_results = calculate_auc(df)

    # Group AUC by sample names
    auc_summary = group_auc_by_sample(df, auc_results)

    # Generate the AUC plot with sample colors
    auc_fig = plot_auc(auc_summary, sample_colors)
    auc_html = pio.to_html(auc_fig, full_html=False)

    # Generate the line graph with sample colors
    line_graph = create_plot(df, sample_colors)
    line_graph_html = pio.to_html(line_graph, full_html=False)

    # Read the heatmap HTML content
    heatmap_filename = 'heatmap.html'
    if os.path.exists(heatmap_filename):
        with open(heatmap_filename, 'r') as file:
            heatmap_html = file.read()
    else:
        heatmap_html = "<p>Heatmap not available.</p>"

    # Calculate the max growth rate per hour for each well
    max_growth_rates = calculate_max_growth_rate_per_hour(df.copy())
    summary = group_replicates_and_calculate_mean_std(max_growth_rates)

    # Generate the max growth rate plot with sample colors
    max_growth_rate_fig = plot_max_growth_rate(summary, sample_colors)
    max_growth_rate_html = pio.to_html(max_growth_rate_fig, full_html=False)

    # Combine everything into the report HTML
    report_html = f"""
    <html>
    <head>
        <title>Growth Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Growth Analysis Report</h1>
        <h2>Optical Density Line Graph</h2>
        {line_graph_html}
        <h2>Heatmap of Maximum OD Values</h2>
        {heatmap_html}
        <h2>Max Growth Rate (Mean ± Std)</h2>
        {max_growth_rate_html}
        <h2>Area Under the Curve (Mean ± Std)</h2>
        {auc_html}
    </body>
    </html>
    """

    # Return the report as an HTML file
    return send_file(
        io.BytesIO(report_html.encode('utf-8')),
        download_name='growth_analysis_report.html',  # Updated parameter name
        as_attachment=True,
        mimetype='text/html'
    )


@app.route('/download_heatmap/<string:filename>')
def download_heatmap(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return "File not found.", 404

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

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
