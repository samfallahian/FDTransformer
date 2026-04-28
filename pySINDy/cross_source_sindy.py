import numpy as np
import pysindy as ps
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

def cross_source_sindy():
    print("\n--- Cross-Source Recovery Testing ---")
    
    # Load data
    raw = np.load("pySINDy/raw_data_grad.npz")
    encoded = np.load("pySINDy/encoded_data_grad.npz")
    predicted = np.load("pySINDy/predicted_data_grad.npz")
    
    scenarios = [
        ("Raw", raw),
        ("Encoded", encoded),
        ("Predicted", predicted)
    ]
    
    results = []
    
    # We want to see if vorticity from one source can predict enstrophy from another.
    # If they are all consistent, they should.
    
    for input_label, input_data in scenarios:
        for target_label, target_data in scenarios:
            X = np.stack([input_data['wx'].flatten(), input_data['wy'].flatten(), input_data['wz'].flatten()], axis=-1)
            y = target_data['enstrophy'].flatten().reshape(-1, 1)
            
            library = ps.PolynomialLibrary(degree=2, include_bias=True)
            optimizer = ps.STLSQ(threshold=1e-5)
            
            model = ps.SINDy(feature_library=library, optimizer=optimizer)
            model.fit(X, t=1.0, x_dot=y)
            
            coefs = optimizer.coef_[0]
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            
            # expected coeffs at indices 4, 7, 9 are 0.5
            res = {
                'Input': input_label,
                'Target': target_label,
                'MSE': mse,
                'wx2': coefs[4],
                'wy2': coefs[7],
                'wz2': coefs[9]
            }
            results.append(res)
            print(f"Input: {input_label:9} | Target: {target_label:9} | MSE: {mse:.4e} | wx^2: {coefs[4]:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("pySINDy/cross_source_results.csv", index=False)
    print("\nSaved: pySINDy/cross_source_results.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    cross_source_sindy()
