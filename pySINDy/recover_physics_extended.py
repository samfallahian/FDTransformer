import numpy as np
import pysindy as ps
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

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

def main():
    scenarios = ['raw', 'encoded', 'predicted']
    results = []
    
    for s in scenarios:
        path = f"pySINDy/{s}_extended.npz"
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
    df.to_csv("pySINDy/extended_physics_results.csv", index=False)
    print("\nResults saved to pySINDy/extended_physics_results.csv")

if __name__ == "__main__":
    main()
