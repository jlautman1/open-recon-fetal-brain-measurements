# fetal_normative.py — CSV is the single source of truth for band, status, and predicted GA

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- load article-based medians (Normative.csv) ---
_module_dir = os.path.dirname(__file__)
_NORM_DF = pd.read_csv(os.path.join(_module_dir, 'Normative.csv'))

# Map column names and fixed SDs (mm)
_COL = {'CBD': 'cbd_mean', 'BBD': 'bbd_mean', 'TCD': 'tcd_mean'}
_SD  = {'CBD': 5.0,        'BBD': 5.0,        'TCD': 3.0}

def _interp_func(measure):
    """Linear interpolator over the CSV median curve (with gentle extrapolation)."""
    x = _NORM_DF['week'].values
    y = _NORM_DF[_COL[measure]].values
    return interp1d(x, y, kind='linear', fill_value='extrapolate'), x.min(), x.max()

def get_csv_stats(week, measure):
    """Mean & SD at a GA (from CSV medians + fixed SD per measure)."""
    f, _, _ = _interp_func(measure)
    mean = float(f(week))
    return mean, _SD[measure]

def get_normative_curve(measure, step=1.0):
    """Curve + ±2 SD band for plotting (from CSV)."""
    f, xmin, xmax = _interp_func(measure)
    weeks = np.arange(xmin, xmax + 1e-9, step)
    means = f(weeks).astype(float)
    sds = np.full_like(means, _SD[measure], dtype=float)
    return weeks, means, sds

def get_status(value, mean, sd):
    if value < mean - 2*sd:
        return "Below Norm"
    elif value > mean + 2*sd:
        return "Above Norm"
    else:
        return "Normal"

# ---------------- Predicted GA from CSV (inverse of the median curve) ----------------
def predict_ga_from_measurement(value, measure):
    """
    Estimate GA (weeks) from a measurement (mm) by inverting the CSV median curve.
    Uses piecewise-linear inversion with linear extrapolation outside the CSV range.
    Returns GA rounded to 0.1 weeks.
    """
    x = _NORM_DF['week'].values.copy()                 # weeks
    y = _NORM_DF[_COL[measure]].values.copy()          # medians (mm)

    # Ensure strictly increasing in "y" for inversion stability
    order = np.argsort(y)
    y_sorted = y[order]
    x_sorted = x[order]

    # Linear extrapolation below/above the CSV range
    if value <= y_sorted[0]:
        # use first two points
        x0, x1 = x_sorted[0], x_sorted[1]
        y0, y1 = y_sorted[0], y_sorted[1]
        ga = x0 + (x1 - x0) * (value - y0) / (y1 - y0)
    elif value >= y_sorted[-1]:
        # use last two points
        x0, x1 = x_sorted[-2], x_sorted[-1]
        y0, y1 = y_sorted[-2], y_sorted[-1]
        ga = x1 + (x1 - x0) * (value - y1) / (y1 - y0)
    else:
        # piecewise-linear inverse via interpolation
        ga = float(np.interp(value, y_sorted, x_sorted))

    return float(np.round(ga, 1))
# -------------------------------------------------------------------------------------

def plot_with_subject_point(measure, week, measured_value, outdir):
    """
    Draw curve/band from CSV at integer weeks; place the subject point at 'week'.
    Status is computed against CSV mean/SD at that 'week'.
    """
    weeks, means, sds = get_normative_curve(measure)
    upper = means + 2*sds
    lower = means - 2*sds

    mean_at_wk, sd_at_wk = get_csv_stats(week, measure)
    status = get_status(measured_value, mean_at_wk, sd_at_wk)

    plt.figure(figsize=(6, 4))
    plt.style.use('default')

    colors = {
        'primary':   '#1f4e79',
        'secondary': '#4a90e2',
        'success':   '#28a745',
        'warning':   '#ffc107',
        'danger':    '#dc3545',
        'gray':      '#6c757d'
    }

    plt.plot(weeks, means, label="Mean", color=colors['primary'], lw=3)
    plt.fill_between(weeks, lower, upper, color=colors['secondary'], alpha=0.20,
                     label="±2 SD (Normal Range)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel("Gestational Age (weeks)")
    plt.ylabel(f"{measure} (mm)")

    color_map = {"Normal": "success", "Below Norm": "warning", "Above Norm": "danger"}
    pc = colors[color_map[status]]

    plt.scatter([week], [measured_value], color=pc, edgecolor='white',
                s=120, linewidths=2, zorder=10, label="Subject Measurement")
    plt.annotate(f'{measured_value:.1f}mm',
                 xy=(week, measured_value),
                 xytext=(week + 0.7, measured_value + 2),
                 fontsize=10, fontweight='bold', color=pc,
                 arrowprops=dict(arrowstyle='->', color=pc, lw=1.2, alpha=0.8))

    status_display = {"Below Norm": "Below Normal Range",
                      "Above Norm": "Above Normal Range",
                      "Normal": "Within Normal Range"}[status]
    plt.title(f"{measure} Normative Analysis\n"
              f"{measured_value:.1f}mm at {week}w ({status_display})",
              fontsize=14, fontweight='bold', color=colors['primary'])

    legend = plt.legend(frameon=True, loc='upper left', fontsize=10,
                        fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.xlim(weeks.min() - 0.5, weeks.max() + 0.5)
    y_range = (upper.max() - lower.min())
    plt.ylim(lower.min() - 0.1*y_range, upper.max() + 0.1*y_range)

    plt.tight_layout()
    fname = os.path.join(outdir, f"{measure.lower()}_norm.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return fname, status

def normative_report_all(measured_dict, week_or_none, outdir):
    """
    measured_dict: {'CBD': mm, 'BBD': mm, 'TCD': mm}
    week_or_none: if None → (not used here) ; if float → use the SAME GA for all measures.
                  (Your fetal_measure.py passes the default GA here.)
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for measure in ['CBD', 'BBD', 'TCD']:
        val = float(measured_dict[measure])

        # GA used for the plot & status = whatever caller passes (your default GA).
        plot_ga = float(week_or_none)

        # plot & status
        fname, status = plot_with_subject_point(measure, plot_ga, val, outdir)

        # numbers we print beside: CSV mean/sd at the same GA
        mean, sd = get_csv_stats(plot_ga, measure)

        # Predicted GA (now from CSV inversion)
        pred_ga = predict_ga_from_measurement(val, measure)

        results[measure] = {
            "value": val,
            "mean": float(np.round(mean, 2)),
            "sd": float(sd),
            "status": status,
            "plot_path": fname,
            "predicted_ga": pred_ga,
            "plot_ga": float(plot_ga)
        }
    return results
