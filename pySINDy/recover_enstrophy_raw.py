import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error

from pysindy_config import load_config_from_args, make_parser, output_path


def recover_enstrophy_raw(input_path):
    data = np.load(input_path)
    wx = data['wx'].flatten()
    wy = data['wy'].flatten()
    wz = data['wz'].flatten()
    enstrophy = data['enstrophy'].flatten()
    
    # We want to recover enstrophy = 0.5 * (wx^2 + wy^2 + wz^2)
    # The input to SINDy will be X = [wx, wy, wz]
    # The output will be enstrophy (as a "dependent variable" if we use it in a specific way)
    # However, SINDy is usually for dX/dt = f(X).
    # To recover an algebraic relation y = f(X), we can use SINDy by treating it as a regression with a library of functions.
    
    X = np.stack([wx, wy, wz], axis=-1)
    y = enstrophy.reshape(-1, 1)
    
    # Custom library with polynomial terms up to degree 2
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    
    # Standard optimizer (STLSQ)
    optimizer = ps.STLSQ(threshold=1e-5)
    
    # Since we want y = f(X), we can use SINDy's underlying mechanism or just use a sparse regressor.
    # pysindy doesn't directly support y = f(X) as the primary interface (it's for ODEs/PDEs).
    # But we can use SINDy with a dummy "time" or just use the Library/Optimizer directly.
    
    # Using SINDy by pretending y is a state variable and derivatives are zero is not right.
    # Better to use the SINDy library + optimizer manually for algebraic recovery.
    
    feature_names = ['wx', 'wy', 'wz']
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(feature_names)
    
    optimizer.fit(X_poly, y)
    coefs = optimizer.coef_
    
    print("\nRecovered Equation for Enstrophy:")
    equation = "Enstrophy = "
    terms = []
    for i, coeff in enumerate(coefs[0]):
        if abs(coeff) > 1e-6:
            terms.append(f"{coeff:.4f} * {poly_names[i]}")
    equation += " + ".join(terms)
    print(equation)
    
    # Validation
    y_pred = X_poly @ coefs.T
    mse = mean_squared_error(y, y_pred)
    print(f"MSE: {mse:.4e}")

if __name__ == "__main__":
    parser = make_parser("Recover the enstrophy relation from raw-gradient NPZ data.", common_paths=False, runtime=False)
    parser.add_argument("--input", help="Input NPZ file. Defaults to outputs.raw_grad in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    recover_enstrophy_raw(args.input or output_path(config, "raw_grad", create_parent=False))
