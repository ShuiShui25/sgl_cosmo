#!/usr/bin/env python3
"""Report KDE marginal modes separately from posterior quantiles; never shift samples."""
import argparse
import numpy as np
from scipy.stats import gaussian_kde


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posterior", help="Posterior NPZ from infer_cosmology.py")
    parser.add_argument("--discard", type=float, default=0.0,
                        help="Fraction to discard (default 0; warmup is already excluded)")
    args = parser.parse_args()
    if not 0 <= args.discard < 1:
        parser.error("--discard must be in [0, 1)")
    with np.load(args.posterior, allow_pickle=False) as data:
        print("parameter,kde_mode,median,q16,q84,mean,std")
        for key in ("Om", "w", "w0", "wa", "gamma0", "gamma_s", "log_sig_g",
                    "delta0", "delta_s", "log_sig_d", "beta0", "log_sig_b"):
            if key not in data:
                continue
            values = data[key].reshape(-1)
            values = values[int(len(values) * args.discard):]
            if len(values) < 2 or not np.all(np.isfinite(values)):
                raise ValueError(f"{key}: need at least two finite samples")
            q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
            grid = np.linspace(values.min(), values.max(), 2048)
            mode = values[0] if np.ptp(values) == 0 else grid[np.argmax(gaussian_kde(values)(grid))]
            print(f"{key},{mode:.6f},{median:.6f},{q16:.6f},{q84:.6f},"
                  f"{values.mean():.6f},{values.std():.6f}")


if __name__ == "__main__":
    main()
