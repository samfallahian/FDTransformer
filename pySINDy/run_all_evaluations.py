import numpy as np
import pysindy as ps
import scipy.ndimage as ndimage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

from pysindy_config import load_config_from_args, make_parser, output_path


def recover_enstrophy(wx, wy, wz, enstrophy, label):
    """
    Given vorticity components and enstrophy, recover the algebraic relation using SINDy.
    """
    X = np.stack([wx.flatten(), wy.flatten(), wz.flatten()], axis=-1)
    y = enstrophy.flatten().reshape(-1, 1)
    
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-12) # Reverting to very low threshold for exact recovery
    
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['wx', 'wy', 'wz'])
    
    optimizer.fit(X_poly, y)
    coefs = optimizer.coef_[0]
    
    # Validation
    # pysindy returns coefs in a shape that might be an AxesArray.
    # We convert to plain numpy for simple multiplication.
    c = np.asarray(coefs)
    Xp = np.asarray(X_poly)
    y_pred = Xp @ c
    mse = mean_squared_error(y, y_pred)
    
    results = {
        'Scenario': label,
        'MSE': mse
    }
    
    # Store coefficients for key terms
    for i, name in enumerate(poly_names):
        results[name] = coefs[i]
        
    return results

def run_evaluation_suite(config, raw_path=None, encoded_path=None, predicted_path=None, output_csv=None, figure_path=None):
    data_files = {
        'Raw': raw_path or output_path(config, "raw_grad", create_parent=False),
        'Encoded': encoded_path or output_path(config, "encoded_grad", create_parent=False),
        'Predicted': predicted_path or output_path(config, "predicted_grad", create_parent=False)
    }
    output_csv = output_csv or output_path(config, "evaluation_results")
    figure_path = figure_path or output_path(config, "evaluation_summary")
    
    all_results = []
    
    for label, path in data_files.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found.")
            continue
            
        data = np.load(path)
        
        # Case 1: No Smoothing
        wx = data['wx']
        wy = data['wy']
        wz = data['wz']
        enstrophy = data['enstrophy']
        
        res = recover_enstrophy(wx, wy, wz, enstrophy, label)
        all_results.append(res)
        
        # Case 2: Smoothing
        # Note: We need the full velocity grid 'V' for spatial smoothing,
        # which was used to calculate vorticity in the prepare_* scripts.
        if 'V' in data:
            V = data['V']
            V_smooth = ndimage.gaussian_filter(V, sigma=1.0)
            
            # Recalculate gradients from smoothed velocity
            # dx=1, dy=1, dz=1 is assumed if not provided
            # Using the same logic as test_smoothing.py
            grad_u = np.gradient(V_smooth[..., 0])
            grad_v = np.gradient(V_smooth[..., 1])
            grad_w = np.gradient(V_smooth[..., 2])
            
            # grad_u[0] is d/dx, grad_u[1] is d/dy, grad_u[2] is d/dz
            wx_s = grad_w[1] - grad_v[2]
            wy_s = grad_u[2] - grad_w[0]
            wz_s = grad_v[0] - grad_u[1]
            enstrophy_s = 0.5 * (wx_s**2 + wy_s**2 + wz_s**2)
            
            res_s = recover_enstrophy(wx_s, wy_s, wz_s, enstrophy_s, f"{label} (Smoothed)")
            all_results.append(res_s)
        else:
            print(f"Velocity field 'V' not found in {path}, skipping smoothing for {label}.")

    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    cols = ['Scenario', 'MSE', '1', 'wx', 'wy', 'wz', 'wx^2', 'wy^2', 'wz^2', 'wx wy', 'wx wz', 'wy wz']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save table
    df.to_csv(output_csv, index=False)
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    # Create Figure
    plot_coefficients(df, figure_path)
    
    return df

def plot_coefficients(df, figure_path):
    plt.figure(figsize=(12, 8))
    
    # We'll plot coefficients for wx^2, wy^2, wz^2
    terms = ['wx^2', 'wy^2', 'wz^2']
    scenarios = df['Scenario'].values
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, term in enumerate(terms):
        plt.bar(x + i*width, df[term], width, label=term)
        
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Expected (0.5)')
    
    plt.ylabel('Coefficient Value')
    plt.title('Recovered Coefficients for Enstrophy Formula (Expected = 0.5)')
    plt.xticks(x + width, scenarios, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path)
    print(f"\nFigure saved to: {figure_path}")

if __name__ == "__main__":
    parser = make_parser("Run SINDy enstrophy evaluations over raw, encoded, and predicted NPZ files.", runtime=False)
    parser.add_argument("--raw-input", help="Raw input NPZ. Defaults to outputs.raw_grad in the config.")
    parser.add_argument("--encoded-input", help="Encoded input NPZ. Defaults to outputs.encoded_grad in the config.")
    parser.add_argument("--predicted-input", help="Predicted input NPZ. Defaults to outputs.predicted_grad in the config.")
    parser.add_argument("--output", help="Output CSV. Defaults to outputs.evaluation_results in the config.")
    parser.add_argument("--figure", help="Output PNG. Defaults to outputs.evaluation_summary in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    run_evaluation_suite(
        config,
        raw_path=args.raw_input,
        encoded_path=args.encoded_input,
        predicted_path=args.predicted_input,
        output_csv=args.output,
        figure_path=args.figure,
    )
