import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import sys
import os
from PIL import Image, ImageTk

class AnomalyDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Anomaly Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Get the directory where this script is located
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Title
        title = tk.Label(root, text="Hybrid Anomaly Detection in Digital Twins", 
                        font=("Arial", 18, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=20)
        
        # Main container with two panes
        main_container = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg='#2c3e50', 
                                       sashwidth=5, sashrelief=tk.RAISED)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Controls and Output
        left_panel = tk.Frame(main_container, bg='#2c3e50')
        main_container.add(left_panel, width=600)
        
        # Button Frame
        button_frame = tk.Frame(left_panel, bg='#2c3e50')
        button_frame.pack(pady=10, fill=tk.X)
        
        # Define buttons for each step
        self.buttons = [
            ("Step 1: Data Loading & Exploration", "data_loading_exploration.py"),
            ("Step 2: Data Preprocessing", "data_preprocessing.py"),
            ("Step 3: Isolation Forest", "IsolationForest.py"),
            ("Step 4: DBSCAN", "DBSCAN.py"),
            ("Step 5: Hybrid Integration", "hybrid_integration.py"),
            ("Step 6: Sensor Visualization", "sensor_visualization.py"),
            ("Step 7: 3D Visualization", "3d_visualization.py"),
            ("Step 8: Feature Contribution", "feature_contribution.py"),
        ]
        
        # Create buttons
        for i, (text, script) in enumerate(self.buttons):
            btn = tk.Button(button_frame, text=text, width=35, height=2,
                           font=("Arial", 10), bg='#3498db', fg='white',
                           activebackground='#2980b9', cursor='hand2',
                           command=lambda s=script, t=text: self.run_script(s, t))
            btn.grid(row=i, column=0, padx=10, pady=5, sticky='ew')
        
        # Run All Button
        run_all_btn = tk.Button(button_frame, text="▶ Run All Steps", width=35, height=2,
                               font=("Arial", 12, "bold"), bg='#27ae60', fg='white',
                               activebackground='#229954', cursor='hand2',
                               command=self.run_all)
        run_all_btn.grid(row=8, column=0, padx=10, pady=10, sticky='ew')
        
        # Clear Button
        clear_btn = tk.Button(button_frame, text="Clear Output", width=35, height=1,
                             font=("Arial", 10), bg='#e74c3c', fg='white',
                             activebackground='#c0392b', cursor='hand2',
                             command=self.clear_output)
        clear_btn.grid(row=9, column=0, padx=10, pady=5, sticky='ew')
        
        button_frame.columnconfigure(0, weight=1)
        
        # Output Text Area
        output_label = tk.Label(left_panel, text="Output Console:", font=("Arial", 11, "bold"),
                               bg='#2c3e50', fg='white')
        output_label.pack(pady=(10, 5))
        
        self.output_text = scrolledtext.ScrolledText(left_panel, width=70, height=25,
                                                     font=("Courier", 9), bg='#1c1c1c',
                                                     fg='#00ff00', insertbackground='white',
                                                     wrap=tk.WORD)
        self.output_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Right Panel - Graph Visualization
        right_panel = tk.Frame(main_container, bg='#34495e')
        main_container.add(right_panel, width=700)
        
        # Graph Title
        graph_title = tk.Label(right_panel, text="Visualization Panel", 
                              font=("Arial", 14, "bold"), bg='#34495e', fg='white')
        graph_title.pack(pady=10)
        
        # Notebook for multiple graphs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style for notebook
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#34495e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#3498db', foreground='white', 
                       padding=[10, 5], font=('Arial', 9, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#27ae60')])
        
        # Initial placeholder
        self.add_placeholder_tab()
        
        # Status Bar
        self.status_bar = tk.Label(root, text="Ready", font=("Arial", 10),
                                  bg='#34495e', fg='white', anchor='w', padx=10)
        self.status_bar.pack(side='bottom', fill='x')
        
        # Store graph images path
        self.graph_paths = []
        
        # Print debug info at startup
        self.print_debug_info()
        
    def print_debug_info(self):
        """Print debugging information to console"""
        self.output_text.insert('end', "=" * 70 + "\n")
        self.output_text.insert('end', "SYSTEM INFORMATION\n")
        self.output_text.insert('end', "=" * 70 + "\n")
        self.output_text.insert('end', f"Working Directory: {os.getcwd()}\n")
        self.output_text.insert('end', f"App Directory: {self.app_dir}\n")
        self.output_text.insert('end', f"Python Version: {sys.version}\n\n")
        
        self.output_text.insert('end', "Available Python Scripts:\n")
        self.output_text.insert('end', "-" * 70 + "\n")
        
        py_files = [f for f in os.listdir(self.app_dir) if f.endswith('.py')]
        if py_files:
            for py_file in sorted(py_files):
                self.output_text.insert('end', f"  ✓ {py_file}\n")
        else:
            self.output_text.insert('end', "  ⚠ No Python files found!\n")
        
        self.output_text.insert('end', "\n" + "=" * 70 + "\n")
        self.output_text.insert('end', "Ready to run scripts. Click a button to start.\n")
        self.output_text.insert('end', "=" * 70 + "\n\n")
        
    def add_placeholder_tab(self):
        """Add a placeholder tab when no graphs are available"""
        placeholder_frame = tk.Frame(self.notebook, bg='#34495e')
        self.notebook.add(placeholder_frame, text="Welcome")
        
        placeholder_label = tk.Label(placeholder_frame, 
                                     text="📊\n\nGraphs will appear here\nwhen you run the scripts",
                                     font=("Arial", 14), bg='#34495e', fg='#ecf0f1',
                                     justify=tk.CENTER)
        placeholder_label.pack(expand=True)
    
    def clear_graph_tabs(self):
        """Clear all graph tabs except placeholder"""
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.add_placeholder_tab()
        self.graph_paths = []
    
    def add_graph_tab(self, image_path, tab_name):
        """Add a new tab with a graph image"""
        # Use absolute path
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.app_dir, image_path)
            
        if not os.path.exists(image_path):
            return
        
        # Remove placeholder if this is the first real graph
        if len(self.graph_paths) == 0:
            self.notebook.forget(0)
        
        self.graph_paths.append(image_path)
        
        # Create frame for the graph
        graph_frame = tk.Frame(self.notebook, bg='white')
        
        # Create canvas with scrollbars
        canvas = tk.Canvas(graph_frame, bg='white')
        v_scrollbar = tk.Scrollbar(graph_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = tk.Scrollbar(graph_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        
        scrollable_frame = tk.Frame(canvas, bg='white')
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Load and display image
        try:
            img = Image.open(image_path)
            # Resize if too large
            max_width, max_height = 650, 650
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(scrollable_frame, image=photo, bg='white')
            label.image = photo  # Keep a reference
            label.pack(padx=10, pady=10)
            
        except Exception as e:
            error_label = tk.Label(scrollable_frame, text=f"Error loading graph:\n{str(e)}",
                                  font=("Arial", 10), bg='white', fg='red')
            error_label.pack(expand=True)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add tab
        self.notebook.add(graph_frame, text=tab_name)
        self.notebook.select(graph_frame)
    
    def scan_for_graphs(self, script_name):
        """Scan for generated graph files after running a script"""
        # Common graph output patterns
        graph_patterns = {
            "data_loading_exploration.py": ["data_exploration.png", "correlation_matrix.png"],
            "data_preprocessing.py": ["preprocessing_results.png"],
            "IsolationForest.py": ["isolation_forest_analysis.png", "isolation_forest_results.png"],
            "DBSCAN.py": ["dbscan_analysis.png", "dbscan_results.png"],
            "hybrid_integration.py": ["hybrid_integration_analysis.png", "hybrid_results.png"],
            "sensor_visualization.py": ["sensor_data_with_anomalies.png", "sensor_visualization.png"],
            "3d_visualization.py": ["3d_anomaly_visualization.png"],
            "feature_contribution.py": ["feature_contribution_analysis.png"],
        }
        
        # Scan for expected graphs
        if script_name in graph_patterns:
            for pattern in graph_patterns[script_name]:
                full_path = os.path.join(self.app_dir, pattern)
                if os.path.exists(full_path):
                    tab_name = pattern.replace('.png', '').replace('_', ' ').title()
                    self.add_graph_tab(full_path, tab_name)
        
        # Generic scan for recent PNG files
        try:
            png_files = [f for f in os.listdir(self.app_dir) if f.endswith('.png')]
            png_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.app_dir, x)), reverse=True)
            
            # Add up to 3 most recent PNG files not already added
            count = 0
            for png_file in png_files:
                full_path = os.path.join(self.app_dir, png_file)
                if full_path not in self.graph_paths and count < 3:
                    tab_name = png_file.replace('.png', '').replace('_', ' ').title()
                    self.add_graph_tab(full_path, tab_name)
                    count += 1
        except Exception as e:
            self.output_text.insert('end', f"Note: Could not scan for graphs: {str(e)}\n")
    
    def run_script(self, script_name, step_name):
        """Run a single Python script"""
        # Build full path to script
        script_path = os.path.join(self.app_dir, script_name)
        
        # Check if script exists
        if not os.path.exists(script_path):
            error_msg = f"Script '{script_name}' not found!\n\nLooking at: {script_path}\n\nAvailable scripts:\n"
            for f in os.listdir(self.app_dir):
                if f.endswith('.py'):
                    error_msg += f"  - {f}\n"
            messagebox.showerror("Error", error_msg)
            self.output_text.insert('end', f"\n✗ ERROR: {error_msg}\n")
            return
        
        self.output_text.insert('end', f"\n{'='*70}\n")
        self.output_text.insert('end', f"Running: {step_name}\n")
        self.output_text.insert('end', f"Script: {script_name}\n")
        self.output_text.insert('end', f"{'='*70}\n")
        self.output_text.see('end')
        self.status_bar.config(text=f"Running: {step_name}...")
        self.root.update()
        
        try:
            # Run the script and capture output
            result = subprocess.run(
                [sys.executable, script_path], 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=self.app_dir  # Set working directory to app directory
            )
            
            # Display output
            if result.stdout:
                self.output_text.insert('end', result.stdout + '\n')
            if result.stderr:
                self.output_text.insert('end', f"ERRORS:\n{result.stderr}\n")
            
            if result.returncode == 0:
                self.output_text.insert('end', f"\n✓ {step_name} completed successfully!\n")
                self.status_bar.config(text=f"✓ {step_name} completed")
                
                # Scan for and display generated graphs
                self.scan_for_graphs(script_name)
            else:
                self.output_text.insert('end', f"\n✗ {step_name} failed with return code {result.returncode}\n")
                self.status_bar.config(text=f"✗ {step_name} failed")
                
        except subprocess.TimeoutExpired:
            self.output_text.insert('end', f"\n⚠ {step_name} timed out (>5 minutes)\n")
            self.status_bar.config(text=f"⚠ {step_name} timed out")
        except Exception as e:
            self.output_text.insert('end', f"\n✗ Error: {str(e)}\n")
            self.status_bar.config(text="Error occurred")
        
        self.output_text.see('end')
    
    def run_all(self):
        """Run all scripts sequentially"""
        self.clear_output()
        self.output_text.insert('end', "Starting complete pipeline...\n")
        self.output_text.insert('end', f"{'='*70}\n\n")
        
        for step_name, script_name in self.buttons:
            self.run_script(script_name, step_name)
        
        self.output_text.insert('end', f"\n{'='*70}\n")
        self.output_text.insert('end', "✓ ALL STEPS COMPLETED!\n")
        self.output_text.insert('end', f"{'='*70}\n")
        self.status_bar.config(text="✓ All steps completed successfully")
        self.output_text.see('end')
    
    def clear_output(self):
        """Clear the output console and graphs"""
        self.output_text.delete('1.0', 'end')
        self.clear_graph_tabs()
        self.status_bar.config(text="Output cleared - Ready")
        self.print_debug_info()

def main():
    root = tk.Tk()
    app = AnomalyDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()