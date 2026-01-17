"""
IEEE Paper Style Configuration for Matplotlib
=============================================
Centralized style settings for IEEE-compliant figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# IEEE figure sizes (in inches)
IEEE_SINGLE_COL = 3.5  # Single column width
IEEE_DOUBLE_COL = 7.0  # Double column width
IEEE_ASPECT = 0.75     # Height/Width ratio

# IEEE-compliant colors (grayscale-friendly)
IEEE_COLORS = {
    'primary': '#000000',      # Black
    'secondary': '#4D4D4D',    # Dark gray
    'tertiary': '#808080',     # Medium gray
    'quaternary': '#B3B3B3',   # Light gray
    'accent1': '#1F77B4',      # Blue (distinguishable)
    'accent2': '#D62728',      # Red (distinguishable)
    'accent3': '#2CA02C',      # Green
    'grid': '#CCCCCC',         # Grid color
}

# Line styles for differentiation
IEEE_LINE_STYLES = ['-', '--', '-.', ':']

# Markers for differentiation  
IEEE_MARKERS = ['o', 's', '^', 'v', 'D', 'p', 'h', '*']


def set_ieee_style():
    """
    Apply IEEE paper style to matplotlib.
    Call this at the start of your plotting code.
    """
    plt.style.use('default')
    
    params = {
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Axes settings
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'grid.color': IEEE_COLORS['grid'],
        
        # Legend settings
        'legend.fontsize': 8,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Tick settings
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # Math text
        'mathtext.fontset': 'stix',
    }
    
    mpl.rcParams.update(params)


def get_ieee_figsize(width: str = 'single', aspect: float = None) -> tuple:
    """
    Get IEEE-compliant figure size.
    
    Args:
        width: 'single' (3.5in) or 'double' (7.0in)
        aspect: Height/width ratio (default: 0.75)
    
    Returns:
        Tuple (width, height) in inches
    """
    if aspect is None:
        aspect = IEEE_ASPECT
    
    w = IEEE_SINGLE_COL if width == 'single' else IEEE_DOUBLE_COL
    return (w, w * aspect)


def get_ieee_colors(n: int) -> list:
    """
    Get n IEEE-compliant colors that are distinguishable in grayscale.
    
    Args:
        n: Number of colors needed
    
    Returns:
        List of color codes
    """
    # Primary palette (high contrast)
    palette = [
        '#000000',  # Black
        '#E69F00',  # Orange
        '#56B4E9',  # Sky blue
        '#009E73',  # Bluish green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish purple
    ]
    return palette[:n]


def get_ieee_linestyles(n: int) -> list:
    """
    Get n different line styles.
    
    Args:
        n: Number of line styles needed
    
    Returns:
        List of line style strings
    """
    styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    return styles[:n]


def get_ieee_markers(n: int) -> list:
    """
    Get n different markers.
    
    Args:
        n: Number of markers needed
    
    Returns:
        List of marker strings
    """
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'x', '+']
    return markers[:n]
