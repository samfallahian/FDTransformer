import pickle
import gzip
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

SOURCE_ROOT = '/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data/'
EXPERIMENTS = ["3p6", "4p6", "5p2", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"]

def load_data(exp, step):
    path = os.path.join(SOURCE_ROOT, exp, f'{step:04d}.pkl.gz')
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def find_vortex_core_region(df, threshold=0.90):
    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    zs = np.sort(df['z'].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)

    # Fast reshape if data is already a grid
    pivot_vx = df.pivot(index='x', columns=['y', 'z'], values='vx').values.reshape(nx, ny, nz)
    pivot_vy = df.pivot(index='x', columns=['y', 'z'], values='vy').values.reshape(nx, ny, nz)
    pivot_vz = df.pivot(index='x', columns=['y', 'z'], values='vz').values.reshape(nx, ny, nz)

    dvy_dx = np.gradient(pivot_vy, xs, axis=0)
    dvx_dy = np.gradient(pivot_vx, ys, axis=1)
    omega_z = dvy_dx - dvx_dy

    dvz_dy = np.gradient(pivot_vz, ys, axis=1)
    dvy_dz = np.gradient(pivot_vy, zs, axis=2)
    omega_x = dvz_dy - dvy_dz

    dvx_dz = np.gradient(pivot_vx, zs, axis=2)
    dvz_dx = np.gradient(pivot_vz, xs, axis=0)
    omega_y = dvx_dz - dvz_dx

    mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    # Restriction to interaction region (-30, 30)
    x_mask = (xs >= -30) & (xs <= 30)
    xs_sub = xs[x_mask]
    mag_sub = mag[x_mask, :, :]
    
    peak_val = np.max(mag_sub)
    core_mask = mag_sub >= threshold * peak_val
    core_idxs = np.argwhere(core_mask)
    
    core_points = []
    for ix, iy, iz in core_idxs:
        core_points.append({
            'x': xs_sub[ix], 'y': ys[iy], 'z': zs[iz],
            'vort_mag': mag_sub[ix, iy, iz]
        })
    return core_points

def process_exp_step(args):
    exp, step = args
    df = load_data(exp, step)
    if df is None: return None
    core_pts = find_vortex_core_region(df)
    for pt in core_pts:
        pt['exp'] = exp
        pt['step'] = step
    return core_pts

if __name__ == '__main__':
    # Sampling for estimation: 50 steps from each experiment
    tasks = []
    for exp in EXPERIMENTS:
        for step in range(100, 1000, 20): # Sample every 20 steps
            tasks.append((exp, step))
            
    print(f"Analyzing {len(tasks)} samples across {len(EXPERIMENTS)} experiments...")
    
    all_core_points = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_exp_step, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            res = future.result()
            if res:
                all_core_points.extend(res)
                
    df_core = pd.DataFrame(all_core_points)
    df_core.to_csv('transformer_neurIPS/vorticity_core_points_training.csv', index=False)
    print(f"Identified {len(df_core)} core points in training set.")
