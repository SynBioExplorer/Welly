
# Welly: A High-Throughput Growth Analysis Tool

## Purpose
Welly is a web-based application designed to facilitate the analysis of bacterial growth data from high-throughput experiments. It allows users to upload CSV or Excel files containing optical density (OD600nm) measurements over time from 96- or 384-well plates. The application processes the data to generate key metrics such as the maximum growth rate and area under the curve (AUC) for each well, and then aggregates these metrics across replicates. Welly provides interactive visualizations, including line graphs and heatmaps, to help users interpret their data effectively.



## Workflow
1. **Upload Data**: Upload your CSV or Excel file containing optical density readings.
2. **Plate Selection**: Choose the type of plate (96-well or 384-well) used in your experiment.
3. **Data Visualization**: View heatmaps of maximum OD values, growth curves over time, and calculate metrics like maximum growth rates and AUC.
4. **Download Reports**: Download detailed analysis reports, including graphs and data summaries.

## Demo
You can try the tool online at [Welly](http://welly.pythonanywhere.com).

## Installation Guide

### Prerequisites
- Python 3.10 or higher


### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/synbioexplorer/welly.git
   cd optical-density-analysis-tool
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## License
This project is licensed under the MIT License.
