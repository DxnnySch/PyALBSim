import numpy as np


def format_value(v):
    """Nicely formats numeric values for display."""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"  # add commas for large integers
    elif isinstance(v, (float, np.floating)):
        # Format floats depending on magnitude
        if abs(v) >= 1e3 or abs(v) < 1e-3:
            return f"{v:.3e}"  # scientific notation
        else:
            return f"{v:.3f}".rstrip("0").rstrip(
                "."
            )  # up to 3 decimals, remove trailing zeros
    else:
        return str(v)
