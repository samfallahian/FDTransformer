import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error

from pysindy_config import load_config_from_args, make_parser, output_path


def recover_enstrophy_predicted(input_path):
    data = np.load(input_path)
    wx = data['wx'].flatten()
    wy = data['wy'].flatten()
    wz = data['wz'].flatten()
    enstrophy = data['enstrophy'].flatten()
    
    X = np.stack([wx, wy, wz], axis=-1)
    y = enstrophy.reshape(-1, 1)
    
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-5)
    
    feature_names = ['wx', 'wy', 'wz']
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(feature_names)
    
    optimizer.fit(X_poly, y)
    coefs = optimizer.coef_
    
    print("\nRecovered Equation for Enstrophy (Predicted):")
    equation = "Enstrophy = "
    terms = []
    for i, coeff in enumerate(coefs[0]):
        if abs(coeff) > 1e-6:
            terms.append(f"{coeff:.4f} * {poly_names[i]}")
    equation += " + ".join(terms)
    print(equation)
    
    y_pred = X_poly @ coefs.T
    mse = mean_squared_error(y, y_pred)
    print(f"MSE: {mse:.4e}")

if __name__ == "__main__":
    parser = make_parser("Recover the enstrophy relation from predicted-gradient NPZ data.", common_paths=False, runtime=False)
    parser.add_argument("--input", help="Input NPZ file. Defaults to outputs.predicted_grad in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    recover_enstrophy_predicted(args.input or output_path(config, "predicted_grad", create_parent=False))
