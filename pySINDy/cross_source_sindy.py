import numpy as np
import pysindy as ps
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

from pysindy_config import load_config_from_args, make_parser, output_path


def cross_source_sindy(config, raw_path=None, encoded_path=None, predicted_path=None, output_csv=None):
    print("\n--- Cross-Source Recovery Testing ---")
    
    # Load data
    raw = np.load(raw_path or output_path(config, "raw_grad", create_parent=False))
    encoded = np.load(encoded_path or output_path(config, "encoded_grad", create_parent=False))
    predicted = np.load(predicted_path or output_path(config, "predicted_grad", create_parent=False))
    output_csv = output_csv or output_path(config, "cross_source_results")
    
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
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = make_parser("Test cross-source SINDy recovery across raw, encoded, and predicted data.", runtime=False)
    parser.add_argument("--raw-input", help="Raw input NPZ. Defaults to outputs.raw_grad in the config.")
    parser.add_argument("--encoded-input", help="Encoded input NPZ. Defaults to outputs.encoded_grad in the config.")
    parser.add_argument("--predicted-input", help="Predicted input NPZ. Defaults to outputs.predicted_grad in the config.")
    parser.add_argument("--output", help="Output CSV. Defaults to outputs.cross_source_results in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    cross_source_sindy(config, args.raw_input, args.encoded_input, args.predicted_input, args.output)
