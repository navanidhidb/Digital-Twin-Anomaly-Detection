Digital Twin Anomaly Detection System
This project was developed as part of our 5th semester coursework to explore anomaly detection techniques in IoT sensor data using machine learning algorithms.
What does it do?
We built a system that analyzes sensor data from a digital twin environment and identifies unusual patterns or anomalies. The system combines two different detection methods - Isolation Forest and DBSCAN - to get better accuracy than using just one algorithm alone.
We also created a simple GUI application so anyone can run the analysis without writing code themselves.
Why this project?
Digital twins are becoming increasingly important in manufacturing and industrial IoT. Being able to detect anomalies in real-time sensor data helps prevent equipment failures and maintain optimal operating conditions. We wanted to explore how combining different ML algorithms could improve detection accuracy.
Main Features

Pipeline-based approach with 8 distinct processing steps
GUI application for easy interaction (no coding required to run)
Visualizations for better understanding of detected anomalies
Comparative analysis between different detection methods
Feature importance analysis to understand which sensors contribute most to anomalies

Project Structure
Here's what each file does:

app.py - Main GUI application that runs everything
data_loading_exploration.py - Loads and explores the dataset
data_preprocessing.py - Cleans and normalizes the data
IsolationForest.py - Implements the Isolation Forest algorithm
DBSCAN.py - Implements DBSCAN clustering for anomaly detection
hybrid_integration.py - Combines results from both algorithms
sensor_visualization.py - Creates time-series plots of sensor data
3d_visualization.py - 3D scatter plots for visual analysis
feature_contribution.py - Analyzes which features are most important

Dataset
The dataset contains sensor readings with four features:

Temperature
Humidity
Light intensity
Sound levels (Loudness)

Total records: 6,558 data points
Getting Started
What you'll need

Python 3.12 or higher
The packages listed in requirements.txt

Installation Steps

Clone or download this repository
Install the required packages:

bashpip install -r requirements.txt
That's it!
Running the Project
Option 1: Using the GUI (Easiest)
Just run:
bashpython app.py
This opens a window where you can:

Run each step individually to see what it does
Run all steps at once
View graphs and results in real-time

Option 2: Running Scripts Individually
If you want to run specific parts:
bashpython data_preprocessing.py
python IsolationForest.py
python DBSCAN.py
And so on...
How the Detection Works
Step 1: Isolation Forest
This algorithm works by randomly isolating data points. Anomalies are easier to isolate than normal points, so they get flagged.
Step 2: DBSCAN
This algorithm groups similar points together. Points that don't fit into any group are considered anomalies.
Step 3: Hybrid Approach
We combine both methods using weighted voting:

Isolation Forest gets 60% weight
DBSCAN gets 40% weight

This hybrid approach gives us better results than either method alone.
Results
Our system detected approximately 294 anomalies (4.48% of the data) using the hybrid approach.
Key findings:

Temperature showed the highest deviation during anomalies (Z-score: 2.85)
Humidity was the second most significant indicator
The hybrid method achieved 95.84% agreement between both algorithms
Most anomalies occurred during specific time windows, suggesting event-based patterns

What We Learned

Combining multiple algorithms improves detection accuracy
Different algorithms catch different types of anomalies
Feature engineering and normalization are crucial steps
Visual analysis helps validate algorithmic results
Parameter tuning significantly affects detection performance

Technologies Used

Python 
pandas and NumPy for data handling
scikit-learn for the ML algorithms
matplotlib and seaborn for visualizations
tkinter for the GUI

