import numpy as np
import pysindy as ps
import scipy.ndimage as ndimage
from sklearn.metrics import mean_squared_error

from pysindy_config import load_config_from_args, make_parser, output_path


def test_smoothing(data_path, label):
    data = np.load(data_path)
    V = data['V'] # (nx, ny, nz, 3)
    x = np.arange(V.shape[0]) # Dummy coords for gradient if not provided
    y = np.arange(V.shape[1])
    z = np.arange(V.shape[2])
    
    # Smooth the velocity field
    V_smooth = ndimage.gaussian_filter(V, sigma=1.0)
    
    # Recalculate gradients from smoothed velocity
    grad_u = np.gradient(V_smooth[..., 0])
    grad_v = np.gradient(V_smooth[..., 1])
    grad_w = np.gradient(V_smooth[..., 2])
    
    wx = grad_w[1] - grad_v[2]
    wy = grad_u[2] - grad_w[0]
    wz = grad_v[0] - grad_u[1]
    
    enstrophy_smooth = 0.5 * (wx**2 + wy**2 + wz**2)
    
    X = np.stack([wx.flatten(), wy.flatten(), wz.flatten()], axis=-1)
    y = enstrophy_smooth.flatten().reshape(-1, 1)
    
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-12)
    
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['wx', 'wy', 'wz'])
    
    optimizer.fit(X_poly, y)
    coefs = optimizer.coef_
    
    print(f"\nRecovered Equation for {label} (Smoothed):")
    equation = "Enstrophy = "
    terms = []
    for i, coeff in enumerate(coefs[0]):
        if abs(coeff) > 1e-6:
            terms.append(f"{coeff:.4f} * {poly_names[i]}")
    equation += " + ".join(terms)
    print(equation)

if __name__ == "__main__":
    parser = make_parser("Recover enstrophy after smoothing prepared velocity fields.", runtime=False)
    parser.add_argument("--raw-input", help="Raw input NPZ. Defaults to outputs.raw_grad in the config.")
    parser.add_argument("--encoded-input", help="Encoded input NPZ. Defaults to outputs.encoded_grad in the config.")
    parser.add_argument("--predicted-input", help="Predicted input NPZ. Defaults to outputs.predicted_grad in the config.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    test_smoothing(args.raw_input or output_path(config, "raw_grad", create_parent=False), "Raw")
    test_smoothing(args.encoded_input or output_path(config, "encoded_grad", create_parent=False), "Encoded")
    test_smoothing(args.predicted_input or output_path(config, "predicted_grad", create_parent=False), "Predicted")
