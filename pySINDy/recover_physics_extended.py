import numpy as np
import pysindy as ps
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

from pysindy_config import load_config_from_args, make_parser, output_path


def recover_relation(X, y, feature_names, label, target_name):
    """
    Recover algebraic relation y = f(X) using SINDy.
    """
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-12)
    
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(feature_names)
    
    optimizer.fit(X_poly, y)
    coefs = optimizer.coef_[0]
    
    # Validation
    # Use numpy directly to avoid AxesArray issues
    Xp = np.asarray(X_poly)
    c = np.asarray(coefs)
    y_pred = Xp @ c
    mse = mean_squared_error(y, y_pred)
    
    equation = f"{target_name} = "
    terms = []
    for i, coeff in enumerate(coefs):
        if abs(coeff) > 1e-4:
            terms.append(f"{coeff:.4f} * {poly_names[i]}")
    equation += " + ".join(terms) if terms else "0"
    
    return {
        'Scenario': label,
        'Target': target_name,
        'Equation': equation,
        'MSE': mse,
        'Coefficients': coefs,
        'PolyNames': poly_names
    }

def main(config, raw_input=None, encoded_input=None, predicted_input=None, output_csv=None):
    scenarios = [
        ('raw', raw_input or output_path(config, "raw_extended", create_parent=False)),
        ('encoded', encoded_input or output_path(config, "encoded_extended", create_parent=False)),
        ('predicted', predicted_input or output_path(config, "predicted_extended", create_parent=False)),
    ]
    output_csv = output_csv or output_path(config, "extended_physics_results")
    results = []
    
    for s, path in scenarios:
        data = np.load(path)
        
        V = data['V']
        u, v, w = V[..., 0].flatten(), V[..., 1].flatten(), V[..., 2].flatten()
        wx, wy, wz = data['wx'].flatten(), data['wy'].flatten(), data['wz'].flatten()
        
        ke = data['ke'].flatten().reshape(-1, 1)
        helicity = data['helicity'].flatten().reshape(-1, 1)
        enstrophy = data['enstrophy'].flatten().reshape(-1, 1)
        
        # 1. Recover KE from (u, v, w)
        X_ke = np.stack([u, v, w], axis=-1)
        res_ke = recover_relation(X_ke, ke, ['u', 'v', 'w'], s.capitalize(), 'KE')
        results.append(res_ke)
        
        # 2. Recover Helicity from (u, v, w, wx, wy, wz)
        X_hel = np.stack([u, v, w, wx, wy, wz], axis=-1)
        res_hel = recover_relation(X_hel, helicity, ['u', 'v', 'w', 'wx', 'wy', 'wz'], s.capitalize(), 'Helicity')
        results.append(res_hel)
        
        # 3. Recover Enstrophy from (wx, wy, wz)
        X_ens = np.stack([wx, wy, wz], axis=-1)
        res_ens = recover_relation(X_ens, enstrophy, ['wx', 'wy', 'wz'], s.capitalize(), 'Enstrophy')
        results.append(res_ens)

    # Print Summary Table
    print(f"\n{'Scenario':<10} | {'Target':<10} | {'MSE':<10} | {'Equation'}")
    print("-" * 100)
    for r in results:
        print(f"{r['Scenario']:<10} | {r['Target']:<10} | {r['MSE']:.2e} | {r['Equation']}")

    # Save to CSV
    flat_results = []
    for r in results:
        row = {
            'Scenario': r['Scenario'],
            'Target': r['Target'],
            'MSE': r['MSE'],
            'Equation': r['Equation']
        }
        flat_results.append(row)
    
    df = pd.DataFrame(flat_results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    parser = make_parser("Recover KE, helicity, and enstrophy formulas from extended physics NPZ files.", runtime=False)
    parser.add_argument("--raw-input", help="Raw extended NPZ. Defaults to outputs.raw_extended in the config.")
    parser.add_argument("--encoded-input", help="Encoded extended NPZ. Defaults to outputs.encoded_extended in the config.")
    parser.add_argument("--predicted-input", help="Predicted extended NPZ. Defaults to outputs.predicted_extended in the config.")
    parser.add_argument("--output", help="Output CSV. Defaults to outputs.extended_physics_results in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    main(config, args.raw_input, args.encoded_input, args.predicted_input, args.output)
