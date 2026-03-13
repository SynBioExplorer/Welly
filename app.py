from flask import Flask, render_template, request, redirect, url_for, make_response, send_file, session
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
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

# =============================================================================
# Publication-quality styling (Nature/Science style)
# =============================================================================

PUB_AXIS = dict(
    tickcolor='black', showline=True, linewidth=2, linecolor='black',
    ticks='outside', ticklen=5, tickwidth=1.5,
    tickfont=dict(size=11, family='Arial', color='black'),
    titlefont=dict(size=13, family='Arial', color='black'),
)

def apply_pub_style(fig, width=520, height=400, y_grid=True, show_legend=True, panel_label=None):
    """Apply publication-quality styling to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        width=width, height=height,
        font=dict(family='Arial', size=11, color='black'),
        margin=dict(l=60, r=20, t=40, b=60),
        showlegend=show_legend,
        legend=dict(font=dict(size=10, family='Arial'),
                    borderwidth=0, bgcolor='rgba(255,255,255,0.8)'),
        title=None,
    )
    grid_kw = dict(showgrid=True, gridcolor='#e0e0e0', gridwidth=0.5) if y_grid else dict(showgrid=False)
    fig.update_xaxes(**PUB_AXIS, showgrid=False, mirror=True)
    fig.update_yaxes(**PUB_AXIS, mirror=True, **grid_kw)
    if panel_label:
        fig.add_annotation(
            text=f'<b>{panel_label}</b>',
            xref='paper', yref='paper', x=0, y=1.08,
            showarrow=False, font=dict(size=14, family='Arial', color='black'),
            xanchor='left', yanchor='top'
        )
    return fig
app.config['SECRET_KEY'] = os.urandom(24)  # Random secret key for session management

# Configure file-based cache
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache-directory'  # Directory for storing cache files
app.config['CACHE_DEFAULT_TIMEOUT'] = 4000  # Cache timeout in seconds

cache = Cache(app)

def cache_key():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique ID for the session
    return f"user_{session['user_id']}"

def parse_time(time_obj, time_format='auto'):
    if time_format == 'hours' and (isinstance(time_obj, (float, int)) or (isinstance(time_obj, str) and time_obj.replace('.', '', 1).isdigit())):
        return float(time_obj) * 60  # Hours to minutes
    elif time_format == 'minutes' and (isinstance(time_obj, (float, int)) or (isinstance(time_obj, str) and time_obj.replace('.', '', 1).isdigit())):
        return float(time_obj)  # Already in minutes

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

def natural_sort_key(label):
    """
    Generate a sorting key that orders well labels in natural order (e.g., A1, A2, ..., A10).
    Always returns a tuple so mixed label types can be compared.
    """
    label = str(label)
    match = re.match(r"([A-Z]+)(\d+)$", label)
    if match:
        letters, numbers = match.groups()
        return (0, letters, int(numbers))
    return (1, label, 0)

@app.route('/robots.txt')
def robots():
    return 'User-agent: *\nAllow: /\n', 200, {'Content-Type': 'text/plain'}

@app.route('/favicon.ico')
def favicon():
    return '', 204

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
            # Try standard CSV first; fall back to European format (semicolon delimiter, comma decimal)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=0)
            # Detect European comma-decimal format: if non-Time columns are strings containing commas
            non_time_cols = [c for c in df.columns if str(c) != 'Time']
            if non_time_cols and df[non_time_cols[0]].dtype == object:
                sample_val = str(df[non_time_cols[0]].iloc[0])
                if ',' in sample_val and ';' not in sample_val:
                    # Comma-decimal CSV (but comma delimiter) — replace commas in values
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, decimal=',')
                elif ';' in open(uploaded_file.name).read(500) if hasattr(uploaded_file, 'name') else False:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, sep=';', decimal=',')
            # Final fallback: try to convert any remaining comma-decimal strings
            for col in df.columns:
                if df[col].dtype == object and str(col) != 'Time':
                    try:
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
                    except (ValueError, TypeError):
                        pass
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file, header=0)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", 400

        # Initialize a session-specific col_tracker
        col_tracker = defaultdict(int)

        # Ensure all column names are strings
        df.columns = [str(c) for c in df.columns]
        df.columns = df.columns.to_series().apply(lambda col: handle_duplicates(col, col_tracker))
        df.columns = [remove_suffixes(col) for col in df.columns]

        if 'Time' not in df.columns:
            return "The uploaded file must contain a 'Time' column.", 400

        time_format = request.form.get('timeFormat', 'auto')
        time_column = df['Time'].apply(lambda t: parse_time(t, time_format))
        time_column = pd.to_numeric(time_column, errors='raise')

        df.drop(columns=['Time'], inplace=True)
        df.insert(0, 'Time', time_column)

        # Store the data using a user-specific cache key
        cache.set(f'data_{cache_key()}', df)
        cache.set(f'plate_type_{cache_key()}', plate_type)

        return redirect(url_for('edit', plate_type=plate_type))
    except (pd.errors.EmptyDataError, ValueError, TypeError) as e:
        return f"Error processing file: {str(e)}. Please upload a valid CSV or Excel file.", 400


# Function to handle duplicate column names with a session-specific tracker
def handle_duplicates(col, col_tracker):
    if col == "Time":
        return col
    col_tracker[col] += 1
    if col_tracker[col] > 1:
        return f"{col}_{col_tracker[col] - 1}"
    return col

def remove_suffixes(col_name):
    return re.sub(r'\.\d+$', '', str(col_name))

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
            df = df.copy()
            original_columns = list(df.columns)
            cols_to_keep_indices = [0]  # Time column
            new_names = ['Time']
            for orig_well, new_name in well_name_map.items():
                if orig_well in original_columns:
                    idx = original_columns.index(orig_well)
                    cols_to_keep_indices.append(idx)
                    new_names.append(new_name)
            df = df.iloc[:, cols_to_keep_indices]
            df.columns = new_names

        cache.set(f'renamed_data_{cache_key()}', df)
        cache.set(f'use_labels_{cache_key()}', use_labels)
        cache.set(f'well_name_map_{cache_key()}', well_name_map)

        return redirect(url_for('results'))

    # Generate individual well growth curves as a subplot grid matching the plate layout
    if plate_type == '96':
        n_rows, n_cols = 8, 12
    else:
        n_rows, n_cols = 16, 24

    row_labels = [chr(65 + i) for i in range(n_rows)]
    col_labels = [str(j + 1) for j in range(n_cols)]
    subplot_titles = [f"{row_labels[r]}{col_labels[c]}" for r in range(n_rows) for c in range(n_cols)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.03
    )

    time_hours = df['Time'] / 60
    well_columns = list(df.columns[1:])

    # For 384-well plates, downsample to every nth point for performance
    if plate_type == '384' and len(time_hours) > 50:
        step = max(1, len(time_hours) // 50)
        time_ds = time_hours.iloc[::step]
    else:
        step = 1
        time_ds = time_hours

    trace_type = go.Scattergl if plate_type == '384' else go.Scatter

    # Map wells to grid positions: standard labels (A1-P24) go to correct position,
    # non-standard labels fill sequentially into remaining slots
    well_re = re.compile(r'^([A-P])(\d{1,2})$')
    used_positions = set()
    well_positions = {}
    for col_name in well_columns:
        m = well_re.match(col_name)
        if m:
            r = ord(m.group(1)) - 65  # A=0, B=1, ...
            c = int(m.group(2)) - 1   # 1-based to 0-based
            if 0 <= r < n_rows and 0 <= c < n_cols:
                well_positions[col_name] = (r, c)
                used_positions.add((r, c))
    # Non-standard labels fill sequentially into unused slots
    next_slot = 0
    for col_name in well_columns:
        if col_name not in well_positions:
            while next_slot < n_rows * n_cols and (next_slot // n_cols, next_slot % n_cols) in used_positions:
                next_slot += 1
            if next_slot < n_rows * n_cols:
                pos = (next_slot // n_cols, next_slot % n_cols)
                well_positions[col_name] = pos
                used_positions.add(pos)
                next_slot += 1

    for col_name in well_columns:
        if col_name not in well_positions:
            continue  # more wells than grid slots (shouldn't happen)
        r, c = well_positions[col_name]
        y_data = df[col_name].iloc[::step] if step > 1 else df[col_name]
        fig.add_trace(trace_type(
            x=time_ds, y=y_data,
            mode='lines', name=col_name,
            line=dict(width=1, color='#1f77b4'),
            showlegend=False
        ), row=r + 1, col=c + 1)

    # Match width to the Bootstrap container (~1100px) for both plate types
    plot_width = 1100
    cell_h = plot_width // n_cols
    plot_height = n_rows * cell_h + 60
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=20)
    )

    # Uniform y-axis range across all subplots
    all_od = df.iloc[:, 1:]
    y_min = float(all_od.min().min())
    y_max = float(all_od.max().max())
    y_pad = (y_max - y_min) * 0.05
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='#ddd')
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='#ddd',
                     range=[y_min - y_pad, y_max + y_pad])

    # Make subplot titles smaller
    for ann in fig.layout.annotations:
        ann.font.size = 7 if plate_type == '384' else 9

    wells_div = pyo.plot(fig, output_type='div', include_plotlyjs='cdn')

    well_names = list(df.columns[1:])
    if plate_type == '96':
        return render_template('edit_96.html', well_names=well_names, heatmap_graph=wells_div, use_labels=use_labels)
    elif plate_type == '384':
        return render_template('edit_384.html', well_names=well_names, heatmap_graph=wells_div, use_labels=use_labels)

# app.py


@app.route('/results', methods=['GET', 'POST'])
def results():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return redirect(url_for('index'))

    # Get all sample names (excluding 'Time')
    samples = [col for col in df.columns if col != 'Time']

    # Create a sorted list of unique sample names
    unique_samples = sorted(set(samples), key=natural_sort_key)

    # Retrieve existing sample colors or initialize with pastel colors
    sample_colors = cache.get(f'sample_colors_{cache_key()}') or {}

    # Ensure all unique samples have colors assigned
    for sample in unique_samples:
        if sample not in sample_colors or not sample_colors[sample]:
            sample_colors[sample] = string_to_pastel_color(sample)
    cache.set(f'sample_colors_{cache_key()}', sample_colors)

    # Map sanitized sample names to original sample names
    sample_to_field_name = {}
    sanitized_to_original = {}
    for sample in unique_samples:
        sanitized_sample = re.sub(r'[^\w]', '_', sample)
        field_name = f'sample_color_{sanitized_sample}'
        sample_to_field_name[sample] = field_name
        sanitized_to_original[field_name] = sample

    # Retrieve cached axis settings or defaults
    axis_settings = cache.get(f'axis_settings_{cache_key()}') or {}

    if request.method == 'POST':
        # Process color selections
        form_data = request.form.to_dict()
        for key, value in form_data.items():
            if key in sanitized_to_original:
                sample = sanitized_to_original[key]
                sample_colors[sample] = value.strip()
        # Update the cache with new colors
        cache.set(f'sample_colors_{cache_key()}', sample_colors)

        # Process axis settings
        for field in ['x_min', 'x_max', 'y_min', 'y_max']:
            val = form_data.get(field, '').strip()
            axis_settings[field] = float(val) if val else None
        for field in ['x_label', 'y_label']:
            val = form_data.get(field, '').strip()
            axis_settings[field] = val if val else None
        cache.set(f'axis_settings_{cache_key()}', axis_settings)

    # Generate the Plotly graph with the current colors
    fig = create_plot(df, sample_colors, axis_settings=axis_settings)
    plotly_graph = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('results.html', plotly_graph=plotly_graph, samples=samples,
                           unique_samples=unique_samples, sample_colors=sample_colors,
                           sample_to_field_name=sample_to_field_name,
                           axis_settings=axis_settings)


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

def create_plot(df, sample_colors, axis_settings=None, for_report=False):
    data_columns = [col for col in df.columns if col != 'Time']

    melted_df = df.melt(id_vars=['Time'], value_vars=data_columns, var_name='Sample', value_name='OD')
    melted_df['Time'] = melted_df['Time'] / 60  # Convert time to hours

    group_df = melted_df.groupby(['Time', 'Sample']).agg({'OD': ['mean', 'std']}).reset_index()
    group_df.columns = ['Time', 'Sample', 'mean_OD', 'std_OD']

    traces = []
    sorted_samples = sorted(group_df['Sample'].unique(), key=natural_sort_key)
    for sample in sorted_samples:
        sample_data = group_df[group_df['Sample'] == sample]
        color = sample_colors.get(sample, string_to_pastel_color(sample))

        trace = go.Scatter(
            x=sample_data['Time'],
            y=sample_data['mean_OD'],
            error_y=dict(type='data', array=sample_data['std_OD'], visible=True, thickness=1),
            mode='lines+markers',
            name=sample,
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=3),
            text=[f"Time: {t:.3f} hours<br>OD: {val:.3f}" for t, val in zip(sample_data['Time'], sample_data['mean_OD'])],
            hoverinfo='text'
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_xaxes(title_text='Time (hours)')
    fig.update_yaxes(title_text='OD600nm')

    if for_report:
        apply_pub_style(fig, width=1100, height=500)
    else:
        fig.update_layout(
            title='Optical Density Readings',
            paper_bgcolor='white', plot_bgcolor='white', width=1000, height=750,
            xaxis=dict(color='black', tickcolor='black', ticks='outside', ticklen=5, tickwidth=2,
                       tickfont=dict(size=12), showgrid=False, showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(color='black', tickcolor='black', ticks='outside', ticklen=5, tickwidth=2,
                       tickfont=dict(size=12), showgrid=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        )

    if axis_settings:
        if axis_settings.get('x_min') is not None or axis_settings.get('x_max') is not None:
            fig.update_xaxes(range=[axis_settings.get('x_min'), axis_settings.get('x_max')])
        if axis_settings.get('y_min') is not None or axis_settings.get('y_max') is not None:
            fig.update_yaxes(range=[axis_settings.get('y_min'), axis_settings.get('y_max')])
        if axis_settings.get('x_label'):
            fig.update_xaxes(title_text=axis_settings['x_label'])
        if axis_settings.get('y_label'):
            fig.update_yaxes(title_text=axis_settings['y_label'])

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

# Step 4: Calculate µmax (maximum specific growth rate) for each well/sample
def calculate_umax(df):
    col_tracker = defaultdict(int)

    def handle_duplicates(col):
        col_tracker[col] += 1
        if col_tracker[col] > 1:
            return f"{col}_{col_tracker[col] - 1}"
        return col

    df.columns = df.columns.to_series().apply(handle_duplicates)

    df_melted = pd.melt(df, id_vars=['Time'], var_name='Sample', value_name='OD')
    df_melted['Time'] = df_melted['Time'] / 60  # minutes to hours

    # Compute ln(OD), clipping non-positive values
    df_melted['ln_OD'] = np.log(df_melted['OD'].clip(lower=1e-10))

    umax_results = []
    window = 5
    for sample, group in df_melted.groupby('Sample'):
        group = group.sort_values('Time').reset_index(drop=True)
        time_vals = group['Time'].values
        ln_od_vals = group['ln_OD'].values

        max_slope = 0.0
        best_i = 0
        for i in range(len(time_vals) - window + 1):
            t_win = time_vals[i:i + window]
            ln_win = ln_od_vals[i:i + window]
            # Linear regression slope: Σ((t-t̄)(y-ȳ)) / Σ((t-t̄)²)
            t_mean = t_win.mean()
            ln_mean = ln_win.mean()
            numerator = np.sum((t_win - t_mean) * (ln_win - ln_mean))
            denominator = np.sum((t_win - t_mean) ** 2)
            if denominator > 0:
                slope = numerator / denominator
                if slope > max_slope:
                    max_slope = slope
                    best_i = i

        # Doubling time: t_d = ln(2) / µmax
        doubling_time = np.log(2) / max_slope if max_slope > 0 else np.nan

        # Lag phase via tangent intercept method
        if max_slope > 0:
            mid = best_i + window // 2
            t_umax = time_vals[mid]
            ln_od_umax = ln_od_vals[mid]
            ln_od_0 = ln_od_vals[0]
            lag_time = t_umax - (ln_od_umax - ln_od_0) / max_slope
            lag_time = max(lag_time, 0.0)
        else:
            lag_time = np.nan

        original_sample = re.sub(r'_\d+$', '', sample)
        umax_results.append({
            'Original_Sample': original_sample,
            'Sample': sample,
            'Growth_rate': max_slope,
            'Doubling_time': doubling_time,
            'Lag_phase': lag_time
        })

    return pd.DataFrame(umax_results)

# Step 5: Group replicates, calculate mean and std
def group_replicates_and_calculate_mean_std(max_growth_rate_df):
    summary = max_growth_rate_df.groupby('Original_Sample').agg(
        Mean_umax=('Growth_rate', 'mean'),
        Std_umax=('Growth_rate', 'std'),
        Mean_doubling_time=('Doubling_time', 'mean'),
        Std_doubling_time=('Doubling_time', 'std'),
        Mean_lag_phase=('Lag_phase', 'mean'),
        Std_lag_phase=('Lag_phase', 'std')
    ).reset_index()
    summary.columns = ['Sample', 'Mean_umax', 'Std_umax', 'Mean_doubling_time', 'Std_doubling_time', 'Mean_lag_phase', 'Std_lag_phase']
    summary = summary.sort_values('Sample', key=lambda x: x.map(natural_sort_key))

    return summary

# Step 6: Plot the max growth rates (mean ± std)
def plot_max_growth_rate(summary, sample_colors, panel_label=None):
    summary = summary.copy()
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Sample']]
    fig = go.Figure([go.Bar(
        x=summary['Sample'],
        y=summary['Mean_umax'].round(3),
        error_y=dict(type='data', array=summary['Std_umax'].fillna(0).round(3), visible=True, thickness=1.5, color='black'),
        marker=dict(color=colors),
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    )])
    fig.update_xaxes(title_text='Sample')
    fig.update_yaxes(title_text='\u00b5max (h\u207b\u00b9)')
    apply_pub_style(fig, show_legend=False, panel_label=panel_label)
    return fig





####################### Area under curve calculation #######################

def calculate_auc(df):
    """
    Calculate the Area Under the Curve (AUC) for each well in the DataFrame.
    """
    auc_results = {}
    time_values = df['Time'].values / 60  # Convert minutes to hours

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
    summary = summary.sort_values('Original_Sample', key=lambda x: x.map(natural_sort_key))

    return summary





def plot_auc(summary, sample_colors, panel_label=None):
    summary = summary.copy()
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Original_Sample']]
    fig = go.Figure([go.Bar(
        x=summary['Original_Sample'],
        y=summary['Mean_AUC'].round(3),
        error_y=dict(type='data', array=summary['Std_AUC'].fillna(0).round(3), visible=True, thickness=1.5, color='black'),
        marker=dict(color=colors),
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    )])
    fig.update_xaxes(title_text='Sample')
    fig.update_yaxes(title_text='AUC (OD\u00b7h)')
    apply_pub_style(fig, show_legend=False, panel_label=panel_label)
    return fig






####################### Log-transformed growth curves #######################

def plot_log_growth_curves(df, sample_colors):
    data_columns = [col for col in df.columns if col != 'Time']
    melted_df = df.melt(id_vars=['Time'], value_vars=data_columns, var_name='Sample', value_name='OD')
    melted_df['Time'] = melted_df['Time'] / 60
    melted_df['ln_OD'] = np.log(melted_df['OD'].clip(lower=1e-10))

    group_df = melted_df.groupby(['Time', 'Sample']).agg(
        mean_ln_OD=('ln_OD', 'mean'),
        std_ln_OD=('ln_OD', 'std')
    ).reset_index()

    traces = []
    sorted_samples = sorted(group_df['Sample'].unique(), key=natural_sort_key)
    for sample in sorted_samples:
        sample_data = group_df[group_df['Sample'] == sample]
        color = sample_colors.get(sample, string_to_pastel_color(sample))
        traces.append(go.Scatter(
            x=sample_data['Time'], y=sample_data['mean_ln_OD'],
            error_y=dict(type='data', array=sample_data['std_ln_OD'].fillna(0), visible=True, thickness=1),
            mode='lines+markers', name=sample,
            line=dict(color=color, width=1),
            marker=dict(color=color)
        ))

    fig = go.Figure(data=traces)
    fig.update_xaxes(title_text='Time (hours)')
    fig.update_yaxes(title_text='ln(OD)')
    apply_pub_style(fig, width=1100, height=500)
    return fig


####################### Growth rate kinetics #######################

def plot_growth_rate_kinetics(df, sample_colors):
    col_tracker = defaultdict(int)

    def handle_dup(col):
        col_tracker[col] += 1
        if col_tracker[col] > 1:
            return f"{col}_{col_tracker[col] - 1}"
        return col

    df = df.copy()
    df.columns = df.columns.to_series().apply(handle_dup)

    df_melted = pd.melt(df, id_vars=['Time'], var_name='Sample', value_name='OD')
    df_melted['Time'] = df_melted['Time'] / 60
    df_melted['ln_OD'] = np.log(df_melted['OD'].clip(lower=1e-10))

    window = 5
    kinetics_rows = []
    for sample, group in df_melted.groupby('Sample'):
        group = group.sort_values('Time').reset_index(drop=True)
        time_vals = group['Time'].values
        ln_od_vals = group['ln_OD'].values

        for i in range(len(time_vals) - window + 1):
            t_win = time_vals[i:i + window]
            ln_win = ln_od_vals[i:i + window]
            t_mean = t_win.mean()
            ln_mean = ln_win.mean()
            numerator = np.sum((t_win - t_mean) * (ln_win - ln_mean))
            denominator = np.sum((t_win - t_mean) ** 2)
            slope = max(numerator / denominator, 0.0) if denominator > 0 else 0.0
            mid_time = t_win[window // 2]
            original_sample = re.sub(r'_\d+$', '', sample)
            kinetics_rows.append({
                'Time': mid_time,
                'Sample': original_sample,
                'mu': slope
            })

    kinetics_df = pd.DataFrame(kinetics_rows)
    grouped = kinetics_df.groupby(['Time', 'Sample']).agg(
        mean_mu=('mu', 'mean'),
        std_mu=('mu', 'std')
    ).reset_index()

    traces = []
    sorted_samples = sorted(grouped['Sample'].unique(), key=natural_sort_key)
    for sample in sorted_samples:
        sdata = grouped[grouped['Sample'] == sample]
        color = sample_colors.get(sample, string_to_pastel_color(sample))
        traces.append(go.Scatter(
            x=sdata['Time'], y=sdata['mean_mu'],
            error_y=dict(type='data', array=sdata['std_mu'].fillna(0), visible=True, thickness=1),
            mode='lines+markers', name=sample,
            line=dict(color=color, width=1),
            marker=dict(color=color)
        ))

    fig = go.Figure(data=traces)
    fig.update_xaxes(title_text='Time (hours)')
    fig.update_yaxes(title_text='\u00b5 (h\u207b\u00b9)')
    apply_pub_style(fig, width=1100, height=500)
    return fig


####################### Doubling time and lag phase bar charts #######################

def plot_doubling_time(summary, sample_colors, panel_label=None):
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Sample']]
    fig = go.Figure([go.Bar(
        x=summary['Sample'],
        y=summary['Mean_doubling_time'].round(3),
        error_y=dict(type='data', array=summary['Std_doubling_time'].fillna(0).round(3), visible=True, thickness=1.5, color='black'),
        marker=dict(color=colors),
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    )])
    fig.update_xaxes(title_text='Sample')
    fig.update_yaxes(title_text='Doubling time (h)')
    apply_pub_style(fig, show_legend=False, panel_label=panel_label)
    return fig


def plot_lag_phase(summary, sample_colors, panel_label=None):
    colors = [sample_colors.get(sample, '#37738F') for sample in summary['Sample']]
    fig = go.Figure([go.Bar(
        x=summary['Sample'],
        y=summary['Mean_lag_phase'].round(3),
        error_y=dict(type='data', array=summary['Std_lag_phase'].fillna(0).round(3), visible=True, thickness=1.5, color='black'),
        marker=dict(color=colors),
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    )])
    fig.update_xaxes(title_text='Sample')
    fig.update_yaxes(title_text='Lag phase (h)')
    apply_pub_style(fig, show_legend=False, panel_label=panel_label)
    return fig


####################### Report helper: sparkline grid + heatmap #######################

def generate_report_sparklines(df_raw, plate_type, well_name_map=None, sample_colors=None):
    """Generate the well sparkline subplot grid for the report using raw (pre-rename) data."""
    if plate_type == '96':
        n_rows, n_cols = 8, 12
    else:
        n_rows, n_cols = 16, 24

    well_name_map = well_name_map or {}
    sample_colors = sample_colors or {}

    row_labels = [chr(65 + i) for i in range(n_rows)]
    col_labels = [str(j + 1) for j in range(n_cols)]
    subplot_titles = [f"{row_labels[r]}{col_labels[c]}" for r in range(n_rows) for c in range(n_cols)]

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
                        horizontal_spacing=0.008, vertical_spacing=0.025)

    time_hours = df_raw['Time'] / 60
    well_columns = list(df_raw.columns[1:])

    # Map wells to grid positions (same logic as edit page)
    well_re = re.compile(r'^([A-P])(\d{1,2})$')
    used_positions = set()
    well_positions = {}
    for col_name in well_columns:
        m = well_re.match(col_name)
        if m:
            r = ord(m.group(1)) - 65
            c = int(m.group(2)) - 1
            if 0 <= r < n_rows and 0 <= c < n_cols:
                well_positions[col_name] = (r, c)
                used_positions.add((r, c))
    next_slot = 0
    for col_name in well_columns:
        if col_name not in well_positions:
            while next_slot < n_rows * n_cols and (next_slot // n_cols, next_slot % n_cols) in used_positions:
                next_slot += 1
            if next_slot < n_rows * n_cols:
                pos = (next_slot // n_cols, next_slot % n_cols)
                well_positions[col_name] = pos
                used_positions.add(pos)
                next_slot += 1

    for col_name in well_columns:
        if col_name not in well_positions:
            continue
        r, c = well_positions[col_name]
        sample_name = well_name_map.get(col_name)
        color = sample_colors.get(sample_name, '#1f77b4') if sample_name else '#1f77b4'
        fig.add_trace(go.Scatter(
            x=time_hours, y=df_raw[col_name],
            mode='lines', showlegend=False,
            line=dict(width=1, color=color)
        ), row=r + 1, col=c + 1)

    all_od = df_raw.iloc[:, 1:]
    y_min = float(all_od.min().min())
    y_max = float(all_od.max().max())
    y_pad = (y_max - y_min) * 0.05

    plot_width = 1100
    cell_h = plot_width // n_cols
    fig.update_layout(
        width=plot_width, height=n_rows * cell_h + 40,
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False, margin=dict(l=20, r=20, t=30, b=10)
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='#ddd')
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='#ddd',
                     range=[y_min - y_pad, y_max + y_pad])
    font_size = 6 if plate_type == '384' else 8
    for ann in fig.layout.annotations:
        ann.font.size = font_size
    return fig


def generate_report_heatmap(df_raw, plate_type):
    """Generate a max-OD heatmap of the plate for the report."""
    well_data = df_raw.iloc[:, 1:]
    max_values = well_data.max()
    well_columns = list(well_data.columns)

    if plate_type == '96':
        rows, cols = 8, 12
    else:
        rows, cols = 16, 24

    # Build heatmap grid, placing wells at correct positions (NaN for empty slots)
    heatmap_data = np.full((rows, cols), np.nan)
    well_re = re.compile(r'^([A-P])(\d{1,2})$')
    used_positions = set()
    well_positions = {}
    for col_name in well_columns:
        m = well_re.match(col_name)
        if m:
            r = ord(m.group(1)) - 65
            c = int(m.group(2)) - 1
            if 0 <= r < rows and 0 <= c < cols:
                well_positions[col_name] = (r, c)
                used_positions.add((r, c))
    next_slot = 0
    for col_name in well_columns:
        if col_name not in well_positions:
            while next_slot < rows * cols and (next_slot // cols, next_slot % cols) in used_positions:
                next_slot += 1
            if next_slot < rows * cols:
                pos = (next_slot // cols, next_slot % cols)
                well_positions[col_name] = pos
                used_positions.add(pos)
                next_slot += 1
    for col_name in well_columns:
        if col_name in well_positions:
            r, c = well_positions[col_name]
            heatmap_data[r, c] = max_values[col_name]

    x_labels = [str(i + 1) for i in range(cols)]
    y_labels = [chr(65 + i) for i in range(rows)]

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data, x=x_labels, y=y_labels,
        colorscale='YlOrRd', xgap=1, ygap=1,
        colorbar=dict(title=dict(text='Max OD', font=dict(size=11, family='Arial')),
                      tickfont=dict(size=10, family='Arial')),
        hovertemplate='Well: %{y}%{x}<br>Max OD: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis=dict(side='top', tickfont=dict(size=10, family='Arial')),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10, family='Arial')),
        paper_bgcolor='white', plot_bgcolor='white',
        width=1100, height=500 if plate_type == '96' else 700,
        margin=dict(l=30, r=80, t=30, b=10)
    )
    return fig


####################### download stuff #######################


@app.route('/download_report')
def download_report():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    df_raw = cache.get(f'data_{cache_key()}')
    plate_type = cache.get(f'plate_type_{cache_key()}') or '96'
    sample_colors = cache.get(f'sample_colors_{cache_key()}') or {}
    well_name_map = cache.get(f'well_name_map_{cache_key()}') or {}

    # --- Generate all figures ---
    line_graph = create_plot(df, sample_colors, for_report=True)
    line_graph_html = pio.to_html(line_graph, full_html=False, include_plotlyjs=False)

    log_growth_fig = plot_log_growth_curves(df, sample_colors)
    log_growth_html = pio.to_html(log_growth_fig, full_html=False, include_plotlyjs=False)

    growth_rate_kinetics_fig = plot_growth_rate_kinetics(df, sample_colors)
    growth_rate_kinetics_html = pio.to_html(growth_rate_kinetics_fig, full_html=False, include_plotlyjs=False)

    max_growth_rates = calculate_umax(df.copy())
    summary = group_replicates_and_calculate_mean_std(max_growth_rates)

    max_growth_rate_fig = plot_max_growth_rate(summary, sample_colors, panel_label='a) Maximum Specific Growth Rate (\u00b5max)')
    max_growth_rate_html = pio.to_html(max_growth_rate_fig, full_html=False, include_plotlyjs=False)

    doubling_time_fig = plot_doubling_time(summary, sample_colors, panel_label='b) Doubling Time')
    doubling_time_html = pio.to_html(doubling_time_fig, full_html=False, include_plotlyjs=False)

    lag_phase_fig = plot_lag_phase(summary, sample_colors, panel_label='c) Lag Phase')
    lag_phase_html = pio.to_html(lag_phase_fig, full_html=False, include_plotlyjs=False)

    auc_results = calculate_auc(df)
    auc_summary = group_auc_by_sample(df, auc_results)
    auc_fig = plot_auc(auc_summary, sample_colors, panel_label='d) Area Under the Curve')
    auc_html = pio.to_html(auc_fig, full_html=False, include_plotlyjs=False)

    # Sparkline grid and heatmap (use raw data with well labels)
    if df_raw is not None:
        sparkline_fig = generate_report_sparklines(df_raw, plate_type, well_name_map, sample_colors)
        sparkline_html = pio.to_html(sparkline_fig, full_html=False, include_plotlyjs=False)
        heatmap_fig = generate_report_heatmap(df_raw, plate_type)
        heatmap_html = pio.to_html(heatmap_fig, full_html=False, include_plotlyjs=False)
    else:
        sparkline_html = '<p>Raw data not available.</p>'
        heatmap_html = '<p>Raw data not available.</p>'

    # Build the report
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Growth Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  /* ---- Nature-style report ---- */
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 30px;
    background: #fff;
  }}
  header {{
    border-bottom: 3px solid #1a1a1a;
    padding-bottom: 12px;
    margin-bottom: 32px;
  }}
  header h1 {{
    font-size: 22pt;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
  }}
  header .subtitle {{
    font-size: 10pt;
    color: #555;
  }}
  h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #1a1a1a;
    margin: 28px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #ccc;
  }}
  h2 .fig-label {{
    font-weight: 400;
    color: #666;
    font-size: 10pt;
  }}
  .caption {{
    font-size: 9.5pt;
    color: #444;
    margin: 2px 0 16px 0;
    line-height: 1.45;
  }}
  .caption em {{ font-style: italic; }}

  /* Two-column grid */
  .row {{
    display: flex;
    gap: 24px;
    margin-bottom: 8px;
  }}
  .col-full {{ width: 100%; }}
  .col-half {{ width: calc(50% - 12px); min-width: 0; }}

  /* Plotly container */
  .plot-wrap {{
    border: 1px solid #e0e0e0;
    border-radius: 3px;
    padding: 8px;
    background: #fafafa;
    margin-bottom: 4px;
    overflow-x: auto;
  }}
  .plot-wrap .js-plotly-plot {{ margin: 0 auto; }}

  /* Section divider */
  .section-divider {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 32px 0;
  }}
  footer {{
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #ccc;
    font-size: 8.5pt;
    color: #888;
    text-align: center;
  }}
</style>
</head>
<body>

<header>
  <h1>Growth Analysis Report</h1>
  <div class="subtitle">Generated by Welly &mdash; Bacterial Growth Curve Analysis Tool</div>
</header>

<!-- ============================================================ -->
<!-- SECTION 1: Individual Well Growth Curves -->
<!-- ============================================================ -->
<h2><span class="fig-label">Figure 1</span> &nbsp;Individual Well Growth Curves</h2>
<p class="caption">
  Growth curves (OD600nm vs time) for each well, arranged in plate layout. All wells share the same y-axis scale.
</p>
<div class="row">
  <div class="col-full">
    <div class="plot-wrap">{sparkline_html}</div>
  </div>
</div>

<!-- ============================================================ -->
<!-- SECTION 2: Heatmap -->
<!-- ============================================================ -->
<h2><span class="fig-label">Figure 2</span> &nbsp;Maximum OD Heatmap</h2>
<p class="caption">
  Heatmap of the maximum OD600nm reached per well. Colour scale: YlOrRd.
</p>
<div class="row">
  <div class="col-full">
    <div class="plot-wrap">{heatmap_html}</div>
  </div>
</div>

<hr class="section-divider">

<!-- ============================================================ -->
<!-- SECTION 3: Growth Curves (OD and log-OD) -->
<!-- ============================================================ -->
<h2><span class="fig-label">Figure 3</span> &nbsp;Growth Curves (Mean &plusmn; SD)</h2>
<p class="caption">
  Optical density at 600 nm over time, grouped by sample. Error bars represent &plusmn;1 SD across replicates.
</p>
<div class="row">
  <div class="col-full">
    <div class="plot-wrap">{line_graph_html}</div>
  </div>
</div>

<h2><span class="fig-label">Figure 4</span> &nbsp;Log-Transformed Growth Curves</h2>
<p class="caption">
  Natural logarithm of OD vs time. Linear regions indicate exponential growth phase.
</p>
<div class="row">
  <div class="col-full">
    <div class="plot-wrap">{log_growth_html}</div>
  </div>
</div>

<hr class="section-divider">

<!-- ============================================================ -->
<!-- SECTION 4: Growth Rate Analysis -->
<!-- ============================================================ -->
<h2><span class="fig-label">Figure 5</span> &nbsp;Specific Growth Rate (&mu;) Kinetics</h2>
<p class="caption">
  Instantaneous specific growth rate at each time point, calculated as the slope of ln(OD) vs time
  using a 5-point sliding window linear regression. Units: h<sup>&minus;1</sup>.
</p>
<div class="row">
  <div class="col-full">
    <div class="plot-wrap">{growth_rate_kinetics_html}</div>
  </div>
</div>

<h2><span class="fig-label">Figure 6</span> &nbsp;Growth Parameters (Mean &plusmn; SD)</h2>
<div class="row">
  <div class="col-half">
    <p class="caption"><strong>a)</strong> &mu;<sub>max</sub>: max slope of ln(OD) vs time (5-pt window). Units: h<sup>&minus;1</sup>.</p>
    <div class="plot-wrap">{max_growth_rate_html}</div>
  </div>
  <div class="col-half">
    <p class="caption"><strong>b)</strong> Doubling time: t<sub>d</sub> = ln(2)/&mu;<sub>max</sub>. Units: h.</p>
    <div class="plot-wrap">{doubling_time_html}</div>
  </div>
</div>
<div class="row">
  <div class="col-half">
    <p class="caption"><strong>c)</strong> Lag phase (&lambda;): tangent intercept method at &mu;<sub>max</sub>. Units: h.</p>
    <div class="plot-wrap">{lag_phase_html}</div>
  </div>
  <div class="col-half">
    <p class="caption"><strong>d)</strong> Area under the curve (AUC): trapezoidal integration of OD vs time. Units: OD&middot;h.</p>
    <div class="plot-wrap">{auc_html}</div>
  </div>
</div>

<footer>
  Report generated by <strong>Welly</strong> &mdash; Growth Curve Analysis Tool for 96- and 384-well Microplate Readers.<br>
  For comments or feature requests, please email <a href="mailto:felix.meier@mq.edu.au">felix.meier@mq.edu.au</a>.
</footer>

</body>
</html>"""

    report_bytes = report_html.encode('utf-8')
    response = make_response(report_bytes)
    response.headers['Content-Disposition'] = 'attachment; filename=growth_analysis_report.html'
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Content-Length'] = len(report_bytes)
    return response


@app.route('/download_heatmap/<string:filename>')
def download_heatmap(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return "File not found.", 404

@app.route('/download_growth_rate_csv')
def download_growth_rate_csv():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    max_growth_rates = calculate_umax(df.copy())
    summary = group_replicates_and_calculate_mean_std(max_growth_rates)

    formula = '# umax = max slope of ln(OD) vs time (5-point window). Doubling time = ln(2)/umax. Lag phase = tangent intercept method.\n'
    csv_data = '\ufeff' + formula + summary.to_csv(index=False)
    response = make_response(csv_data.encode('utf-8'))
    response.headers['Content-Disposition'] = 'attachment; filename=umax_summary.csv'
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    return response


@app.route('/download_auc_csv')
def download_auc_csv():
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    auc_results = calculate_auc(df)
    auc_summary = group_auc_by_sample(df, auc_results)
    auc_summary = auc_summary.rename(columns={'Original_Sample': 'Sample'})

    formula = '# AUC = area under the OD vs time (hours) curve, calculated using the trapezoidal rule. Units: OD*h\n'
    csv_data = '\ufeff' + formula + auc_summary.to_csv(index=False)
    response = make_response(csv_data.encode('utf-8'))
    response.headers['Content-Disposition'] = 'attachment; filename=auc_summary.csv'
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    return response


@app.route('/download/<string:csv_filename>')
def download(csv_filename):
    df = cache.get(f'renamed_data_{cache_key()}')
    if df is None:
        return "Session expired or no data available.", 400

    csv_data = '\ufeff' + df.to_csv(index=False)
    response = make_response(csv_data.encode('utf-8'))
    response.headers['Content-Disposition'] = f'attachment; filename={csv_filename}'
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    return response

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
