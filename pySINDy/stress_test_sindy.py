import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

def stress_test_sindy(data_path, label):
    print(f"\n--- Stress Testing: {label} ---")
    data = np.load(data_path)
    wx = data['wx'].flatten()
    wy = data['wy'].flatten()
    wz = data['wz'].flatten()
    enstrophy = data['enstrophy'].flatten()
    
    # Baseline (No Noise)
    X = np.stack([wx, wy, wz], axis=-1)
    y = enstrophy.reshape(-1, 1)
    
    noise_levels = [0, 1e-4, 1e-3, 1e-2, 0.1] # Relative noise levels
    results = []
    
    for noise in noise_levels:
        # Add noise to input components
        # enstrophy = 0.5 * (wx^2 + wy^2 + wz^2)
        # We don't add noise to the target y to see if the relation is still found 
        # from noisy components.
        
        X_noisy = X + noise * np.random.randn(*X.shape) * np.std(X, axis=0)
        
        library = ps.PolynomialLibrary(degree=2, include_bias=True)
        optimizer = ps.STLSQ(threshold=1e-5)
        
        model = ps.SINDy(feature_library=library, optimizer=optimizer)
        
        # SINDy for algebraic recovery: y = f(X)
        # We fit X_noisy -> y
        model.fit(X_noisy, t=1.0, x_dot=y)
        
        coefs = optimizer.coef_[0]
        y_pred = model.predict(X_noisy)
        mse = mean_squared_error(y, y_pred)
        
        # We expect coefficients for wx^2, wy^2, wz^2 to be 0.5
        # Index for wx^2, wy^2, wz^2 in PolynomialLibrary(degree=2):
        # 0: 1
        # 1: wx
        # 2: wy
        # 3: wz
        # 4: wx^2
        # 5: wx wy
        # 6: wx wz
        # 7: wy^2
        # 8: wy wz
        # 9: wz^2
        
        c_wx2 = coefs[4]
        c_wy2 = coefs[7]
        c_wz2 = coefs[9]
        
        results.append({
            'noise': noise,
            'mse': mse,
            'wx2': c_wx2,
            'wy2': c_wy2,
            'wz2': c_wz2
        })
        print(f"Noise {noise:.1e} | MSE: {mse:.4e} | wx^2: {c_wx2:.4f}, wy^2: {c_wy2:.4f}, wz^2: {c_wz2:.4f}")

    return results

if __name__ == "__main__":
    raw_results = stress_test_sindy("pySINDy/raw_data_grad.npz", "Raw Data")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    noises = [r['noise'] for r in raw_results]
    wx2 = [r['wx2'] for r in raw_results]
    wy2 = [r['wy2'] for r in raw_results]
    wz2 = [r['wz2'] for r in raw_results]
    
    plt.semilogx(noises, wx2, 'o-', label='wx^2 Coeff')
    plt.semilogx(noises, wy2, 's-', label='wy^2 Coeff')
    plt.semilogx(noises, wz2, 'd-', label='wz^2 Coeff')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Ground Truth (0.5)')
    
    plt.xlabel('Relative Noise Level')
    plt.ylabel('Recovered Coefficient')
    plt.title('SINDy Robustness to Noise in Vorticity Components')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('pySINDy/noise_robustness.png')
    print("\nSaved: pySINDy/noise_robustness.png")
