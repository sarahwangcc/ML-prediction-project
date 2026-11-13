"""SGC Project 2: GKX-style ML return prediction."""
import warnings


def quiet_warnings():
    """Silence known-spurious warnings so real ones stay visible.

    - numpy 2.0.x with Apple Accelerate (macOS arm64) emits 'divide by zero /
      overflow / invalid value encountered in matmul' on finite inputs.
      Predictions are checked for finiteness in run_backtest instead.
    - LightGBM + recent scikit-learn: 'X does not have valid feature names'.
    """
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
    warnings.filterwarnings("ignore", message=r".*eval_set.*is deprecated")
    try:
        from scipy.linalg import LinAlgWarning
        warnings.filterwarnings("ignore", category=LinAlgWarning)
    except ImportError:
        pass
