#!/usr/bin/env python3
"""
overlap_analysis_entire_experiment.py - Run complete overlap analysis for all time steps.

PURPOSE:
This script orchestrates the complete velocity overlap analysis pipeline for an entire
experiment (all time steps 1-1200). It uses multithreading to process multiple time steps
in parallel and generates a final animated HTML visualization that can play through all
time frames.

PROCESS:
For each time step t in [1, 1200]:
1. Run generate_decoded_velocity_analysis.py to create decoded velocity and position data
2. Run analysis_overlapping_velocities.py to analyze velocity distributions and compute RMSE
3. Run analysis_overlapping_velocities_mutual_information.py to analyze correlations

Finally, combine all individual HTML files into a single animated HTML with frame controls.

USAGE:
    python encoder/overlap_analysis_entire_experiment.py --dataset 7p2 --start 1 --end 1200 --threads 5
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Change to parent directory
os.chdir(PARENT_DIR)

# Configuration
SCRIPT_DIR = Path(PARENT_DIR) / "encoder"
GENERATE_SCRIPT = SCRIPT_DIR / "generate_decoded_velocity_analysis.py"
ANALYSIS_SCRIPT = SCRIPT_DIR / "analysis_overlapping_velocities.py"
MI_SCRIPT = SCRIPT_DIR / "analysis_overlapping_velocities_mutual_information.py"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize host preferences to get correct paths
from Ordered_001_Initialize import HostPreferences  # noqa: E402

try:
    host_prefs = HostPreferences()
    metadata_path = Path(host_prefs.metadata_location)
    PROJECT_ROOT = metadata_path.parent.parent

    OUTPUT_DIR = Path(host_prefs.training_data_path) / "overlap_analysis"
    FIGURE_OUTPUT_DIR = PROJECT_ROOT / "encoder" / "velocity_overlap_analysis"

    logger.info(f"Initialized paths from HostPreferences:")
    logger.info(f"  Output dir: {OUTPUT_DIR}")
    logger.info(f"  Figure output dir: {FIGURE_OUTPUT_DIR}")
except Exception as e:
    logger.warning(f"Could not load HostPreferences, using default paths: {e}")
    OUTPUT_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"
    FIGURE_OUTPUT_DIR = "/Users/kkreth/PycharmProjects/cgan/encoder/velocity_overlap_analysis"


def run_step(script_path, dataset, time, step_name):
    """
    Run a single analysis step (script) for a specific time step.

    Args:
        script_path: Path to the Python script to run
        dataset: Dataset name (e.g., "7p2")
        time: Time step
        step_name: Name of the step for logging

    Returns:
        Tuple of (success: bool, time: int, step_name: str, error_msg: str or None)
    """
    try:
        cmd = [sys.executable, str(script_path), '--dataset', dataset, '--time', str(time)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 10 minute timeout per step
        )

        if result.returncode != 0:
            error_msg = f"Step failed with return code {result.returncode}\nSTDERR:\n{result.stderr}"
            return (False, time, step_name, error_msg)

        return (True, time, step_name, None)

    except subprocess.TimeoutExpired:
        return (False, time, step_name, "Timeout after 10 minutes")
    except Exception as e:
        return (False, time, step_name, str(e))


def process_single_time(dataset, time):
    """
    Process a single time step through the entire pipeline.

    Args:
        dataset: Dataset name
        time: Time step

    Returns:
        Tuple of (success: bool, time: int, errors: list)
    """
    errors = []

    # Step 1: Generate decoded velocity data
    success, t, step, error = run_step(GENERATE_SCRIPT, dataset, time, "generate")
    if not success:
        errors.append(f"Generate step failed: {error}")
        return (False, time, errors)

    # Step 2: Analyze overlapping velocities
    success, t, step, error = run_step(ANALYSIS_SCRIPT, dataset, time, "analysis")
    if not success:
        errors.append(f"Analysis step failed: {error}")
        return (False, time, errors)

    # Step 3: Mutual information analysis
    success, t, step, error = run_step(MI_SCRIPT, dataset, time, "mi_analysis")
    if not success:
        errors.append(f"MI analysis step failed: {error}")
        return (False, time, errors)

    return (True, time, errors)


def load_rmse_data_for_time(dataset, time, analysis_dir, downsample_factor=2):
    """
    Load RMSE data for a specific time step.

    Args:
        dataset: Dataset name
        time: Time step
        analysis_dir: Directory containing RMSE CSV files
        downsample_factor: Keep every Nth point (default 2 = keep 50%)

    Returns:
        DataFrame with RMSE data or None if not found
    """
    csv_file = Path(analysis_dir) / f"rmse_per_position_{dataset}_{time:04d}.csv"

    if not csv_file.exists():
        return None

    try:
        df = pd.read_csv(csv_file)
        # Downsample to reduce memory usage
        if downsample_factor > 1:
            df = df.iloc[::downsample_factor].copy()
        return df
    except Exception as e:
        logger.error(f"Error loading RMSE data for time {time}: {e}")
        return None


def create_animated_html(dataset, time_steps, output_dir):
    """
    Create an animated HTML visualization that loads time step data on-demand.

    Args:
        dataset: Dataset name
        time_steps: List of time steps to include
        output_dir: Directory to save the HTML file and JSON data files
    """
    logger.info(f"Creating animated HTML with on-demand loading for {len(time_steps)} time steps...")

    # Get downsample factor from global args (will be set in main)
    downsample_factor = getattr(create_animated_html, 'downsample_factor', 2)

    # Create data directory for JSON files
    data_dir = Path(output_dir) / f"animation_data_{dataset}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Export each time step as a separate JSON file and collect metadata
    available_times = []
    vx_min, vx_max = float('inf'), float('-inf')
    vy_min, vy_max = float('inf'), float('-inf')
    vz_min, vz_max = float('inf'), float('-inf')

    for time in tqdm(time_steps, desc="Exporting JSON data"):
        df = load_rmse_data_for_time(dataset, time, FIGURE_OUTPUT_DIR, downsample_factor=downsample_factor)
        if df is None:
            logger.warning(f"Missing RMSE data for time {time}")
            continue

        # Update global min/max for color scales
        vx_min = min(vx_min, df['rmse_vx'].min())
        vx_max = max(vx_max, df['rmse_vx'].max())
        vy_min = min(vy_min, df['rmse_vy'].min())
        vy_max = max(vy_max, df['rmse_vy'].max())
        vz_min = min(vz_min, df['rmse_vz'].min())
        vz_max = max(vz_max, df['rmse_vz'].max())

        # Export to JSON
        json_file = data_dir / f"time_{time:04d}.json"
        json_data = {
            'time': time,
            'x': df['x'].tolist(),
            'y': df['y'].tolist(),
            'z': df['z'].tolist(),
            'rmse_vx': df['rmse_vx'].tolist(),
            'rmse_vy': df['rmse_vy'].tolist(),
            'rmse_vz': df['rmse_vz'].tolist(),
            'mean_vx': df['mean_vx'].tolist(),
            'mean_vy': df['mean_vy'].tolist(),
            'mean_vz': df['mean_vz'].tolist(),
            'sample_count': df['sample_count'].tolist()
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        available_times.append(time)

    if not available_times:
        logger.error("No RMSE data found for any time step!")
        return

    logger.info(f"Exported data for {len(available_times)} time steps")
    logger.info(f"RMSE ranges - Vx: [{vx_min:.6f}, {vx_max:.6f}], "
                f"Vy: [{vy_min:.6f}, {vy_max:.6f}], "
                f"Vz: [{vz_min:.6f}, {vz_max:.6f}]")

    # Create metadata file
    metadata = {
        'dataset': dataset,
        'available_times': available_times,
        'vx_range': [vx_min, vx_max],
        'vy_range': [vy_min, vy_max],
        'vz_range': [vz_min, vz_max],
        'data_dir': f"animation_data_{dataset}"
    }

    metadata_file = data_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    # Create lightweight HTML viewer
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Velocity RMSE Animation - {dataset}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        #container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #controls {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .control-group {{
            display: inline-block;
            margin-right: 20px;
        }}
        button {{
            padding: 8px 16px;
            margin: 0 5px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        button:disabled {{
            background-color: #cccccc;
            cursor: not-allowed;
        }}
        #timeSlider {{
            width: 500px;
            vertical-align: middle;
        }}
        #timeDisplay {{
            font-weight: bold;
            font-size: 18px;
            display: inline-block;
            min-width: 100px;
        }}
        #loading {{
            display: none;
            color: #007bff;
            font-weight: bold;
        }}
        #plot {{
            width: 100%;
            height: 700px;
        }}
        .info {{
            margin-top: 10px;
            padding: 10px;
            background-color: #e7f3ff;
            border-left: 4px solid #007bff;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>Velocity RMSE Animation - Dataset {dataset}</h1>
        <div class="info">
            <strong>Note:</strong> Data is loaded on-demand. Each time step is a separate file,
            allowing smooth navigation through {len(available_times)} time steps without memory issues.
            <br><br>
            <strong>Important:</strong> If you see "Failed to load data", you need to serve this HTML with a local web server:
            <ul style="margin: 10px 0; padding-left: 30px;">
                <li>Navigate to this directory in terminal</li>
                <li>Run: <code>python3 -m http.server 8000</code></li>
                <li>Open: <a href="http://localhost:8000/velocity_rmse_animation_{dataset}_viewer.html">http://localhost:8000/velocity_rmse_animation_{dataset}_viewer.html</a></li>
            </ul>
        </div>

        <div id="controls">
            <div class="control-group">
                <button id="playBtn">▶ Play</button>
                <button id="pauseBtn" disabled>⏸ Pause</button>
                <button id="prevBtn">⏮ Previous</button>
                <button id="nextBtn">⏭ Next</button>
            </div>
            <div class="control-group">
                <span id="timeDisplay">Time: {available_times[0]}</span>
                <input type="range" id="timeSlider" min="0" max="{len(available_times)-1}" value="0" step="1">
                <span id="loading">Loading...</span>
            </div>
            <div class="control-group">
                <label>Speed: </label>
                <input type="number" id="speedInput" value="500" min="100" max="5000" step="100" style="width: 80px;">
                <span> ms/frame</span>
            </div>
        </div>

        <div id="plot"></div>
    </div>

    <script>
        // Configuration from metadata
        const metadata = {json.dumps(metadata, indent=8)};

        const availableTimes = metadata.available_times;
        const vxRange = metadata.vx_range;
        const vyRange = metadata.vy_range;
        const vzRange = metadata.vz_range;
        const dataDir = metadata.data_dir;

        let currentIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        let currentData = null;

        // Create the initial plot structure
        const layout = {{
            title: {{
                text: 'Velocity RMSE by Position - Time: ' + availableTimes[0] + '<br>(RMSE = deviation from mean, interactive 3D)',
                font: {{ size: 16 }}
            }},
            grid: {{ rows: 1, columns: 3, pattern: 'independent' }},
            scene: {{
                xaxis: {{ title: 'X' }},
                yaxis: {{ title: 'Y' }},
                zaxis: {{ title: 'Z' }},
                domain: {{ x: [0, 0.3], y: [0, 1] }}
            }},
            scene2: {{
                xaxis: {{ title: 'X' }},
                yaxis: {{ title: 'Y' }},
                zaxis: {{ title: 'Z' }},
                domain: {{ x: [0.35, 0.65], y: [0, 1] }}
            }},
            scene3: {{
                xaxis: {{ title: 'X' }},
                yaxis: {{ title: 'Y' }},
                zaxis: {{ title: 'Z' }},
                domain: {{ x: [0.7, 1.0], y: [0, 1] }}
            }},
            showlegend: false,
            margin: {{ l: 0, r: 0, t: 100, b: 0 }}
        }};

        // Initialize empty plot
        Plotly.newPlot('plot', [], layout, {{ responsive: true }});

        // Load and display data for a specific time step
        async function loadTime(index) {{
            if (index < 0 || index >= availableTimes.length) return;

            const time = availableTimes[index];
            document.getElementById('loading').style.display = 'inline';

            try {{
                const jsonPath = `./${{dataDir}}/time_${{String(time).padStart(4, '0')}}.json`;
                console.log('Attempting to load:', jsonPath);
                const response = await fetch(jsonPath);
                if (!response.ok) throw new Error(`HTTP error! status: ${{response.status}}`);

                const data = await response.json();
                currentData = data;

                // Create hover text
                const hoverVx = data.x.map((x, i) =>
                    `Pos: (${{x}}, ${{data.y[i]}}, ${{data.z[i]}})<br>` +
                    `Mean: ${{data.mean_vx[i].toFixed(6)}}<br>` +
                    `RMSE: ${{data.rmse_vx[i].toFixed(6)}}<br>` +
                    `Samples: ${{data.sample_count[i]}}`
                );

                const hoverVy = data.x.map((x, i) =>
                    `Pos: (${{x}}, ${{data.y[i]}}, ${{data.z[i]}})<br>` +
                    `Mean: ${{data.mean_vy[i].toFixed(6)}}<br>` +
                    `RMSE: ${{data.rmse_vy[i].toFixed(6)}}<br>` +
                    `Samples: ${{data.sample_count[i]}}`
                );

                const hoverVz = data.x.map((x, i) =>
                    `Pos: (${{x}}, ${{data.y[i]}}, ${{data.z[i]}})<br>` +
                    `Mean: ${{data.mean_vz[i].toFixed(6)}}<br>` +
                    `RMSE: ${{data.rmse_vz[i].toFixed(6)}}<br>` +
                    `Samples: ${{data.sample_count[i]}}`
                );

                // Create traces
                const traces = [
                    // Vx subplot
                    {{
                        type: 'scatter3d',
                        mode: 'markers',
                        x: data.x,
                        y: data.y,
                        z: data.z,
                        marker: {{
                            size: 4,
                            color: data.rmse_vx,
                            colorscale: 'Reds',
                            opacity: 0.6,
                            cmin: vxRange[0],
                            cmax: vxRange[1],
                            colorbar: {{
                                x: 0.28,
                                len: 0.9,
                                title: 'RMSE Vx'
                            }}
                        }},
                        text: hoverVx,
                        hoverinfo: 'text',
                        scene: 'scene'
                    }},
                    // Vy subplot
                    {{
                        type: 'scatter3d',
                        mode: 'markers',
                        x: data.x,
                        y: data.y,
                        z: data.z,
                        marker: {{
                            size: 4,
                            color: data.rmse_vy,
                            colorscale: 'Greens',
                            opacity: 0.6,
                            cmin: vyRange[0],
                            cmax: vyRange[1],
                            colorbar: {{
                                x: 0.63,
                                len: 0.9,
                                title: 'RMSE Vy'
                            }}
                        }},
                        text: hoverVy,
                        hoverinfo: 'text',
                        scene: 'scene2'
                    }},
                    // Vz subplot
                    {{
                        type: 'scatter3d',
                        mode: 'markers',
                        x: data.x,
                        y: data.y,
                        z: data.z,
                        marker: {{
                            size: 4,
                            color: data.rmse_vz,
                            colorscale: 'Blues',
                            opacity: 0.6,
                            cmin: vzRange[0],
                            cmax: vzRange[1],
                            colorbar: {{
                                x: 1.0,
                                len: 0.9,
                                title: 'RMSE Vz'
                            }}
                        }},
                        text: hoverVz,
                        hoverinfo: 'text',
                        scene: 'scene3'
                    }}
                ];

                // Update plot
                const newLayout = {{
                    title: {{
                        text: `Velocity RMSE by Position - Time: ${{time}}<br>(RMSE = deviation from mean, interactive 3D)`,
                        font: {{ size: 16 }}
                    }}
                }};

                Plotly.react('plot', traces, Object.assign({{}}, layout, newLayout), {{ responsive: true }});

                // Update UI
                currentIndex = index;
                document.getElementById('timeSlider').value = index;
                document.getElementById('timeDisplay').textContent = `Time: ${{time}}`;

            }} catch (error) {{
                console.error('Error loading data:', error);
                const msg = `Failed to load data for time ${{time}}: ${{error.message}}\\n\\n` +
                           `This usually happens when opening the file directly (file://).\\n` +
                           `Please start a local web server:\\n\\n` +
                           `1. Open terminal in this directory\\n` +
                           `2. Run: python3 -m http.server 8000\\n` +
                           `3. Open: http://localhost:8000/velocity_rmse_animation_{dataset}_viewer.html`;
                alert(msg);
            }} finally {{
                document.getElementById('loading').style.display = 'none';
            }}
        }}

        // Control handlers
        document.getElementById('playBtn').addEventListener('click', () => {{
            if (!isPlaying) {{
                isPlaying = true;
                document.getElementById('playBtn').disabled = true;
                document.getElementById('pauseBtn').disabled = false;

                const speed = parseInt(document.getElementById('speedInput').value) || 500;
                playInterval = setInterval(() => {{
                    if (currentIndex < availableTimes.length - 1) {{
                        loadTime(currentIndex + 1);
                    }} else {{
                        // Loop back to start
                        loadTime(0);
                    }}
                }}, speed);
            }}
        }});

        document.getElementById('pauseBtn').addEventListener('click', () => {{
            if (isPlaying) {{
                isPlaying = false;
                clearInterval(playInterval);
                document.getElementById('playBtn').disabled = false;
                document.getElementById('pauseBtn').disabled = true;
            }}
        }});

        document.getElementById('prevBtn').addEventListener('click', () => {{
            if (currentIndex > 0) {{
                loadTime(currentIndex - 1);
            }}
        }});

        document.getElementById('nextBtn').addEventListener('click', () => {{
            if (currentIndex < availableTimes.length - 1) {{
                loadTime(currentIndex + 1);
            }}
        }});

        document.getElementById('timeSlider').addEventListener('input', (e) => {{
            loadTime(parseInt(e.target.value));
        }});

        // Load initial time step
        loadTime(0);
    </script>
</body>
</html>"""

    # Save HTML file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"velocity_rmse_animation_{dataset}_viewer.html"

    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Animated HTML viewer saved successfully: {output_file}")
    logger.info(f"Data files location: {data_dir}")
    logger.info(f"Total files created: 1 HTML + {len(available_times)} JSON files + 1 metadata file")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description='Run complete overlap analysis for entire experiment (all time steps)'
    )
    parser.add_argument('--dataset', type=str, default='7p2',
                       help='Dataset name (default: "7p2")')
    parser.add_argument('--start', type=int, default=1,
                       help='Starting time step (default: 1)')
    parser.add_argument('--end', type=int, default=1200,
                       help='Ending time step (default: 1200)')
    parser.add_argument('--threads', type=int, default=5,
                       help='Number of parallel threads (default: 5)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing, only create animated HTML from existing data')
    parser.add_argument('--downsample', type=int, default=2,
                       help='Downsample factor: keep every Nth point (default: 2 = 50%% of points)')
    args = parser.parse_args()

    time_steps = list(range(args.start, args.end + 1))

    logger.info(f"Starting overlap analysis for dataset '{args.dataset}'")
    logger.info(f"Time steps: {args.start} to {args.end} ({len(time_steps)} total)")
    logger.info(f"Parallel threads: {args.threads}")

    if not args.skip_processing:
        # Process all time steps in parallel
        failed_times = []

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # Submit all jobs
            future_to_time = {
                executor.submit(process_single_time, args.dataset, t): t
                for t in time_steps
            }

            # Track progress with tqdm
            with tqdm(total=len(time_steps), desc="Processing time steps") as pbar:
                for future in as_completed(future_to_time):
                    time_step = future_to_time[future]
                    try:
                        success, t, errors = future.result()
                        if not success:
                            logger.error(f"Time {t} failed: {errors}")
                            failed_times.append(t)
                    except Exception as e:
                        logger.error(f"Time {time_step} raised exception: {e}")
                        failed_times.append(time_step)

                    pbar.update(1)

        if failed_times:
            logger.warning(f"Failed to process {len(failed_times)} time steps: {sorted(failed_times)}")
        else:
            logger.info("All time steps processed successfully!")
    else:
        logger.info("Skipping processing, using existing data...")

    # Create animated HTML
    logger.info("Creating animated HTML visualization...")
    logger.info(f"Using downsample factor: {args.downsample} (keeping {100/args.downsample:.1f}% of points)")

    # Pass downsample factor to function via attribute
    create_animated_html.downsample_factor = args.downsample
    create_animated_html(args.dataset, time_steps, FIGURE_OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("COMPLETE! All analysis finished.")
    logger.info(f"Results available in: {FIGURE_OUTPUT_DIR}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
