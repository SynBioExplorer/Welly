from flask import Flask, render_template, request, redirect, url_for, make_response, send_file
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from datetime import timedelta
import numpy as np
import datetime
import os
app = Flask(__name__)

def parse_time(time_str):
    if isinstance(time_str, float) or isinstance(time_str, int):
        return float(time_str)
    elif isinstance(time_str, pd.Timestamp):
        return time_str.hour * 60 + time_str.minute + time_str.second / 60
    elif isinstance(time_str, datetime.datetime):
        # Handling datetime.datetime objects directly by extracting time
        return time_str.hour * 60 + time_str.minute + time_str.second / 60
    elif isinstance(time_str, datetime.time):
        return time_str.hour * 60 + time_str.minute + time_str.second / 60
    elif isinstance(time_str, str):
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2])).total_seconds() / 60
            elif len(parts) == 2:
                return timedelta(hours=int(parts[0]), minutes=int(parts[1])).total_seconds() / 60
        except ValueError:
            raise ValueError(f"Unsupported string format for time_str: {time_str}")
    else:
        print(f"Encountered unsupported type: {type(time_str)} with value: {time_str}")
        raise ValueError(f"Unsupported type for time_str: {type(time_str)}")

    raise ValueError(f"Failed to parse time_str: {time_str} of type: {type(time_str)}")


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
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", 400

        df['Time'] = df['Time'].apply(parse_time)
        
        # Ensure all values are in minutes
        if not isinstance(df['Time'].iloc[0], float):
            df['Time'] = df['Time'].apply(lambda x: x.total_seconds() / 60 if isinstance(x, timedelta) else x)
        
        global data
        data = df
        return redirect(url_for('edit', plate_type=plate_type))
    except (pd.errors.EmptyDataError, ValueError) as e:
        return f"Error processing file: {str(e)}. Please upload a valid CSV or Excel file.", 400

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    plate_type = request.args.get('plate_type', default='96')
    if request.method == 'POST':
        well_name_map = request.form.to_dict()
        df = data.rename(columns=well_name_map)
        global renamed_data
        renamed_data = df
        return redirect(url_for('results'))

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

    # Generate the heatmap with gaps
    heatmap = go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=y_labels,  # No reverse here; labels should correspond to data directly
        colorscale='Sunsetdark',
        xgap=1,  # Add gaps between columns
        ygap=1,   # Add gaps between rows
        hovertemplate='Column: %{x}<br>Row: %{y}<br>Value: %{z}<extra></extra>'  
    )
    heatmap_fig = go.Figure(data=[heatmap])

    # Adjust layout to place the column labels at the top
    heatmap_fig.update_layout(
        xaxis=dict(side="top"),  # Move the x-axis labels to the top
        yaxis=dict(autorange="reversed"),  # Reverse the labels to match standard heatmap orientation
        margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins if needed
    )
    # Save the heatmap as an HTML file
    heatmap_filename = 'heatmap.html'
    pio.write_html(heatmap_fig, file=heatmap_filename, full_html=True)

    heatmap_div = pyo.plot(heatmap_fig, output_type='div', include_plotlyjs=True)



    well_names = list(data.columns[1:])  # Exclude the 'Time' column
    if plate_type == '96':
        return render_template('edit_96.html', well_names=well_names, heatmap_graph=heatmap_div)
    elif plate_type == '384':
        return render_template('edit_384.html', well_names=well_names, heatmap_graph=heatmap_div)



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
    df = df.melt(id_vars=['Time'], var_name='Sample', value_name='OD')
    df_mean = df.groupby(['Sample', 'Time']).mean().reset_index()
    df_std = df.groupby(['Sample', 'Time']).std().reset_index()

    df_mean.rename(columns={'OD': 'mean_OD'}, inplace=True)
    df_std.rename(columns={'OD': 'std_OD'}, inplace=True)

    df_plot = df_mean.merge(df_std, on=['Sample', 'Time'])
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
    csv_data = renamed_data.to_csv(index=False)
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename={csv_filename}'
    response.headers['Content-Type'] = 'text/csv'
    return response

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
