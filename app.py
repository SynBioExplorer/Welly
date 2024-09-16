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

        # Remove suffixes from column names
        import re

        def remove_suffixes(col_name):
            return re.sub(r'\.\d+$', '', col_name)

        df.columns = [remove_suffixes(col) for col in df.columns]

        # Ensure 'Time' column exists and handle it properly
        if 'Time' not in df.columns:
            return "The uploaded file must contain a 'Time' column.", 400

        # Extract and process the 'Time' column
        time_column = df['Time'].apply(parse_time)
        # Explicitly convert to float
        time_column = pd.to_numeric(time_column, errors='raise')

        # Insert the processed 'Time' column back into the dataframe
        df.drop(columns=['Time'], inplace=True)
        df.insert(0, 'Time', time_column)

        # Removed redundant processing below
        # if not isinstance(time_column.iloc[0], float):
        #     time_column = time_column.apply(lambda x: x.total_seconds() / 60 if isinstance(x, timedelta) else x)

        global data
        data = df
        return redirect(url_for('edit', plate_type=plate_type))
    except (pd.errors.EmptyDataError, ValueError) as e:
        return f"Error processing file: {str(e)}. Please upload a valid CSV or Excel file.", 400


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    plate_type = request.args.get('plate_type', default='96')
    use_labels = 'no'  # Default value

    if request.method == 'POST':
        form_data = request.form.to_dict()
        use_labels = form_data.pop('use_labels', 'no')

        if use_labels == 'yes':
            # If "Use CSV/Excel Labels" checkbox is checked, keep the original column names
            df = data.copy()
        else:
            # Rename columns based on the form input
            well_name_map = form_data
            # Ensure we only keep columns that were renamed via the GUI, discard originals
            df = data.rename(columns=well_name_map).loc[:, list(well_name_map.values()) + ['Time']]

        global renamed_data
        renamed_data = df
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
    df_mean = df_melted.groupby(['Sample', 'Time'])['OD'].mean().reset_index()
    df_std = df_melted.groupby(['Sample', 'Time'])['OD'].std().reset_index()

    df_mean.rename(columns={'OD': 'mean_OD'}, inplace=True)
    df_std.rename(columns={'OD': 'std_OD'}, inplace=True)

    df_plot = pd.merge(df_mean, df_std, on=['Sample', 'Time'])
    return df_plot


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