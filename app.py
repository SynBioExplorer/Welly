from flask import Flask, render_template, request, redirect, url_for, make_response, send_file
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


app = Flask(__name__)


from pandas import Timedelta

def parse_time(time_obj):
    if isinstance(time_obj, float) or isinstance(time_obj, int):
        # Assume it's from Excel, representing days. Convert to minutes.
        return float(time_obj) * 24 * 60
    elif isinstance(time_obj, pd.Timestamp):
        # Subtract base date to get total seconds
        base_date = pd.Timestamp('1899-12-31')  # Excel's base date for 1900 date system
        delta = time_obj - base_date
        return delta.total_seconds() / 60
    elif isinstance(time_obj, datetime.datetime):
        # Subtract base date to get total seconds
        base_date = datetime.datetime(1899, 12, 31)
        delta = time_obj - base_date
        return delta.total_seconds() / 60
    elif isinstance(time_obj, datetime.time):
        # Only time part, no days
        return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
    elif isinstance(time_obj, Timedelta):
        # Handle pandas Timedelta objects (common in Excel uploads)
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
        print(f"Encountered unsupported type: {type(time_obj)} with value: {time_obj}")
        raise ValueError(f"Unsupported type for time_str: {type(time_obj)}")


@app.route('/', methods=['GET'])
def index():
    global data, renamed_data, use_labels_global, well_name_map_global

    # Reset global variables to ensure no leftover data from previous uploads
    data = None
    renamed_data = None
    use_labels_global = None
    well_name_map_global = None

    return render_template('index.html')


@app.route('/plate_selection', methods=['POST'])
def plate_selection():
    global data, renamed_data, use_labels_global, well_name_map_global

    # Reset global variables
    data = None
    renamed_data = None
    use_labels_global = None
    well_name_map_global = None

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

        print("Initial DataFrame loaded:", df.head())  # Debugging

        # Handle duplicate columns
        df.columns = df.columns.to_series().apply(handle_duplicates)
        
        # Remove suffixes from column names
        import re
        def remove_suffixes(col_name):
            return re.sub(r'\.\d+$', '', col_name)
        df.columns = [remove_suffixes(col) for col in df.columns]

        print("DataFrame after removing suffixes:", df.head())  # Debugging

        # Ensure 'Time' column exists and handle it properly
        if 'Time' not in df.columns:
            return "The uploaded file must contain a 'Time' column.", 400

        # Extract and process the 'Time' column
        time_column = df['Time'].apply(parse_time)
        time_column = pd.to_numeric(time_column, errors='raise')

        # Insert the processed 'Time' column back into the dataframe
        df.drop(columns=['Time'], inplace=True)
        df.insert(0, 'Time', time_column)

        print("Final DataFrame before saving:", df.head())  # Debugging

        data = df
        return redirect(url_for('edit', plate_type=plate_type))
    except (pd.errors.EmptyDataError, ValueError) as e:
        return f"Error processing file: {str(e)}. Please upload a valid CSV or Excel file.", 400


# Initialize a dictionary to track duplicate columns
col_tracker = defaultdict(int)

def handle_duplicates(col):
    # Do not apply renaming to the "Time" column
    if col == "Time":
        return col
    col_tracker[col] += 1
    if col_tracker[col] > 1:
        return f"{col}_{col_tracker[col] - 1}"
    return col
  
@app.route('/edit', methods=['GET', 'POST'])
def edit():
    plate_type = request.args.get('plate_type', default='96')
    use_labels = 'no'  # Default value

    if request.method == 'POST':
        form_data = request.form.to_dict()
        use_labels = form_data.pop('use_labels', 'no')

        # **Step 1: Exclude 'Time' and empty sample labels from form_data**
        form_data.pop('Time', None)  # Remove 'Time' if it's in form_data
        # Filter out entries where the sample label is empty
        well_name_map = {k: v for k, v in form_data.items() if v.strip() != ''}

        if use_labels == 'yes':
            # Use labels from the CSV/Excel file
            df = data.copy()
            well_name_map = None  # No renaming needed
        else:
            # Rename columns in 'data' based on the mapping
            df = data.rename(columns=well_name_map)

            # Ensure 'Time' is in columns
            if 'Time' not in df.columns:
                raise ValueError("Time column not found in DataFrame after renaming.")

            # **Step 1 Continued: Build list of columns to keep, ensuring 'Time' is first**
            columns_to_keep = ['Time'] + list(well_name_map.values())
            df = df[columns_to_keep]

        global renamed_data, use_labels_global, well_name_map_global
        renamed_data = df
        use_labels_global = use_labels
        well_name_map_global = well_name_map

        # Print statements for debugging
        print("Renamed DataFrame shape:", renamed_data.shape)
        print("Renamed DataFrame columns:", renamed_data.columns)
        print("Use Labels Global:", use_labels_global)
        print("Well Name Map Global:", well_name_map_global)

        return redirect(url_for('results'))
    else:
        use_labels = 'no'
    # Exclude the 'Time' column from the max_values calculation
    well_data = data.iloc[:, 1:]  # Exclude the 'Time' column
    max_values = well_data.max()  # Calculate the maximum values across all time points

    # Reshape the data based on the plate type
    if plate_type == '96':
        rows, cols = 8, 12  # For 96 well plates (8 rows, 12 columns)
    elif plate_type == '384':
        rows, cols = 16, 24  # For 384 well plates (16 rows, 24 columns)

    if len(max_values) != rows * cols:
        raise ValueError(f"Expected {rows * cols} wells, but got {len(max_values)} columns.")

    # Reshape the max_values into a 2D array that matches the plate layout
    heatmap_data = max_values.values.reshape((rows, cols))

    # Generate the x and y labels
    x_labels = [f"{i+1}" for i in range(cols)]
    y_labels = [f"{chr(65+i)}" for i in range(rows)]  # 'A' to 'H' or 'A' to 'P'

        # Generate the heatmap with gaps and specific sizing for each plate type
    if plate_type == '96':
        heatmap = go.Heatmap(
            z=heatmap_data,
            x=x_labels,
            y=y_labels,
            colorscale='Sunsetdark',
            xgap=0.5,  # Smaller gap for 96 well plate
            ygap=0.5,  # Smaller gap for 96 well plate
            hovertemplate='Column: %{x}<br>Row: %{y}<br>Value: %{z}<extra></extra>'
        )
        # Initialize the figure before calling update_layout
        heatmap_fig = go.Figure(data=[heatmap])
        heatmap_fig.update_layout(
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=50, r=50, t=50, b=50),
            width=600,  # Adjust width for 96 well plate
            height=400  # Adjust height for 96 well plate
        )
    elif plate_type == '384':
        heatmap = go.Heatmap(
            z=heatmap_data,
            x=x_labels,
            y=y_labels,
            colorscale='Sunsetdark',
            xgap=1,  # Larger gap for 384 well plate
            ygap=1,  # Larger gap for 384 well plate
            hovertemplate='Column: %{x}<br>Row: %{y}<br>Value: %{z}<extra></extra>'
        )
        # Initialize the figure before calling update_layout
        heatmap_fig = go.Figure(data=[heatmap])
        heatmap_fig.update_layout(
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=50, r=50, t=50, b=50),
            width=800,  # Adjust width for 384 well plate
            height=600  # Adjust height for 384 well plate
        )


    # Save the heatmap as an HTML file
    heatmap_filename = 'heatmap.html'
    pio.write_html(heatmap_fig, file=heatmap_filename, full_html=True)

    heatmap_div = pyo.plot(heatmap_fig, output_type='div', include_plotlyjs=True)

    well_names = list(data.columns[1:])  # Exclude the 'Time' column
    if plate_type == '96':
        return render_template('edit_96.html', well_names=well_names, heatmap_graph=heatmap_div, use_labels=use_labels)
    elif plate_type == '384':
        return render_template('edit_384.html', well_names=well_names, heatmap_graph=heatmap_div, use_labels=use_labels)



@app.route('/results')
def results():
    df = renamed_data
     # Filter out unnamed wells
    named_wells = [col for col in df.columns if col.strip()]  # Exclude columns with empty names
    df = df[named_wells]  # Keep only columns with names
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
    # Convert 'Time' from minutes to hours
    df_melted['Time'] = df_melted['Time'] / 60
    # Group by 'Sample' and 'Time' to calculate mean and sample standard deviation
    df_grouped = df_melted.groupby(['Sample', 'Time']).agg(
        mean_OD=('OD', 'mean'),
        std_OD=('OD', 'std')  # Sample standard deviation (ddof=1 by default)
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
            tickvals=np.arange(0, max(plot_data['Time']) / 60 + 1, 2),  # Tick every hour
            ticktext=[int(hour) for hour in np.arange(0, max(plot_data['Time']) / 60 + 1, 2)]  # Format tick labels without 'h'
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(plot_bgcolor='white',
                      xaxis=dict(color='black', tickcolor='black', showgrid=False, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black'),
                      yaxis=dict(color='black', tickcolor='black', showgrid=False, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black'))
    return fig

#############################################################################################################
############################################################################################################
# Step 1: Assign standard well labels without suffixes
# Step 1: Assign standard well labels (A1-H12 for 96 wells, A1-P24 for 384 wells)
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
    """
    Get labels for the samples based on user input (CSV or GUI) and map them to the well labels.
    """
    if use_labels == 'yes':
        # Use the labels from the CSV, mapped to well labels
        labels = {well: sample for well, sample in zip(df.columns[1:], df.columns[1:])}
    else:
        # Use the labels provided by the user via the GUI
        labels = well_name_map or {}
    return labels


# Step 3: Replace well labels (A1, A2, etc.) with sample labels (e.g., sample1, sample2)
def replace_well_labels_with_sample_labels(df, labels):
    """
    Replace well labels with actual sample labels (e.g., A1 -> sample1).
    """
    if labels:
        new_columns = ['Time'] + [labels.get(col, col) for col in df.columns[1:]]
        df.columns = new_columns
    return df


# Step 4: Calculate max growth rate per hour for each well/sample
def calculate_max_growth_rate_per_hour(df):
    # Handle duplicate columns (replicates) without renaming them in the original DataFrame
    col_tracker = defaultdict(int)

    def handle_duplicates(col):
        col_tracker[col] += 1
        if col_tracker[col] > 1:
            return f"{col}_{col_tracker[col] - 1}"
        return col

    # Apply the duplicate handler function to the column names
    df.columns = df.columns.to_series().apply(handle_duplicates)

    # Melt the DataFrame to transform it into a long format
    df_melted = pd.melt(df, id_vars=['Time'], var_name='Sample', value_name='OD')

    # Convert Time to hours (if not already in hours)
    df_melted['Time'] = df_melted['Time'] / 60  # Assuming Time is in minutes

    # Calculate OD differences and Time differences for each replicate
    df_melted['OD_diff'] = df_melted.groupby('Sample')['OD'].diff()
    df_melted['Time_diff'] = df_melted.groupby('Sample')['Time'].diff()

    # Calculate growth rates (OD/hour)
    df_melted['Growth_rate'] = df_melted['OD_diff'] / df_melted['Time_diff']

    # Handle cases with infinite or undefined growth rates
    df_melted['Growth_rate'] = df_melted['Growth_rate'].replace([np.inf, -np.inf], np.nan).clip(lower=0)

    # Aggregate by the original sample name (ignoring the _1, _2 suffixes) and find the max growth rate for each sample
    df_melted['Original_Sample'] = df_melted['Sample'].str.replace(r'_\d+$', '', regex=True)
    max_growth_rate_df = df_melted.groupby(['Original_Sample', 'Sample'])['Growth_rate'].max().reset_index()

    return max_growth_rate_df



# Step 5: Group replicates, calculate mean and std
def group_replicates_and_calculate_mean_std(max_growth_rate_df):
    """
    Group replicates and calculate the mean and standard deviation of the maximum growth rates.
    """
    # Aggregating maximum growth rates for each original sample (across replicates)
    summary = max_growth_rate_df.groupby('Original_Sample')['Growth_rate'].agg(['mean', 'std']).reset_index()

    # Rename columns for clarity
    summary.columns = ['Sample', 'Mean_growth_rate', 'Std_growth_rate']

    return summary




# Step 5: Plot the max growth rates (mean ± std)
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


# Step 7: Download report with growth rate analysis
@app.route('/download_report')
def download_report():
    global renamed_data, use_labels_global, well_name_map_global

    use_labels = use_labels_global
    well_name_map = well_name_map_global
    plate_type = request.args.get('plate_type', '96')

    # Calculate the max growth rate per hour for each well
    max_growth_rates = calculate_max_growth_rate_per_hour(renamed_data.copy())

    # Group replicates and calculate mean and std of the max growth rates
    summary = group_replicates_and_calculate_mean_std(max_growth_rates)

    # Plot the max growth rate (mean ± std)
    max_growth_rate_fig = plot_max_growth_rate(summary)
    max_growth_rate_html = pio.to_html(max_growth_rate_fig, full_html=False)

    # Build the HTML content for the report
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

    # Save and download the report as an HTML file
    report_filename = 'growth_analysis_report.html'
    with open(report_filename, 'w') as file:
        file.write(report_html)

    return send_file(report_filename, as_attachment=True)





############################################################################################################
############################################################################################################

@app.route('/download_heatmap/<string:filename>')
def download_heatmap(filename):
    # The file is located in the same directory as app.py
    file_path = os.path.join(os.getcwd(), filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return "File not found.", 404


@app.route('/download_interactive_plot/<string:filename>')
def download_interactive_plot(filename):
    df = renamed_data
    graph = create_plot(df)
    pio.write_html(graph, file=filename, full_html=True)
    return send_file(filename, as_attachment=True, download_name=filename)

@app.route('/download/<string:csv_filename>')
def download(csv_filename):
    # Debug: print the shape of the dataframe
    print("Renamed data shape:", renamed_data.shape)
    print("Renamed data columns:", renamed_data.columns)

    csv_data = renamed_data.to_csv(index=False)
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename={csv_filename}'
    response.headers['Content-Type'] = 'text/csv'
    return response

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
