import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Polygon
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import re
import seaborn
from scipy.stats import beta

#%% Flares location 

def convert_prefix_value(s):
    # Convert to string if it's not already
    s = str(s)
    prefix = s[0]
    try:
        value = float(s[1:])
    except ValueError:
        value = None
    if prefix == 'X':
        return value * 1e-4
    elif prefix == 'M':
        return value * 1e-5
    elif prefix == 'C':
        return value * 1e-6
    else:
        return np.nan

def draw_sun_xy(x=None, y=None, figsize=(10, 10), title=None, color='steelblue', ax=None, label = None):
    """
    Trace the solar disk with the Carrington coordinate grid
    and displays points in the form of crosses.
    """
    # ─── Validation ───────────────────────────────────────────────────────────
    if (x is None) != (y is None):
        raise ValueError("The arguments 'x' and 'y' must be provided together.")

    if x is not None and len(x) != len(y):
        raise ValueError(
            f"'x' ({len(x)} éléments) et 'y' ({len(y)} éléments) "
            "doivent avoir la même longueur."
        )

    # ─── Création de la figure SEULEMENT si aucun ax fourni ───────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # ─── solar disk ───────────────────────────────────────────────────────
    from matplotlib.patches import Circle
    sun_circle = Circle((0, 0), 1, color='orange', fill=False, linewidth=2.5, zorder=5)
    ax.add_patch(sun_circle)

    # ─── Carrington Grid ────────────────────────────────────────────────────
    lat_lines = np.arange(-75, 90, 15)
    lon_lines = np.arange(0, 360, 30)

    # latitude lines
    for lat in lat_lines:
        lat_rad = np.radians(lat)
        y_grid  = np.sin(lat_rad)
        x_max   = np.sqrt(max(1 - y_grid**2, 0))
        x_arr   = np.linspace(-x_max, x_max, 300)

        lw  = 1.8 if lat == 0 else 1.0
        ls  = '-'  if lat == 0 else '--'
        col = 'red' if lat == 0 else 'gray'

        ax.plot(x_arr, np.full_like(x_arr, y_grid),
                color=col, linewidth=lw, linestyle=ls, alpha=0.6, zorder=2)
        if lat != 0:
            ax.text(-x_max - 0.02, y_grid, f'{lat:+d}°',
                    ha='right', va='center', fontsize=7.5, color='dimgray')

    # longitude lines
    for lon in lon_lines:
        lon_rad = np.radians(lon)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)

        if cos_lon < 0:
            continue

        lat_arr = np.linspace(-np.pi / 2, np.pi / 2, 300)
        x_plot  = -np.cos(lat_arr) * sin_lon
        y_arr   =  np.sin(lat_arr)

        ax.plot(x_plot, y_arr, color='gray', linewidth=1.0,
                linestyle='--', alpha=0.6, zorder=2)

        if abs(sin_lon) <= 1.0:
            ax.text(sin_lon, 0.05, f'{lon}°',
                    ha='center', va='top', fontsize=7.5, color='dimgray')

    # ─── Axis Annotations ─────────────────────────────────────────────────────
    ax.text(-1.15, 0, 'E\n(East)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text( 1.15, 0, 'W\n(West)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text(0,  1.12, 'N', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text(0, -1.12, 'S', ha='center', va='top',
            fontsize=11, fontweight='bold', color='steelblue')

    # ─── Points (x, y) in the form of crosses ────────────────────────────────────
    if x is not None:
        ax.plot(x, y,
                marker='x',
                markersize=14,
                markeredgewidth=2.5,
                color=color,
                linestyle='none',
                zorder=10,
                label = label)

    # ─── page layout ─────────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if label is not None:
        ax.legend(loc='upper right', fontsize=10)

    # ─── Affichage uniquement si on a créé la figure ──────────────────────────
    # (on ne fait ni tight_layout ni show si ax est fourni par l'extérieur)
    return ax

    
def solarcoor2xy(lat, long):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(long)

    # Vérification : le point est-il sur la face visible ?
    # Face visible : lon entre 270° et 360° ou 0° et 90°  (cos(lon) >= 0)
    cos_lon = np.cos(lon_rad)
    visible = cos_lon >= 0

    x =  np.cos(lat_rad) * np.sin(lon_rad)
    y =  np.sin(lat_rad)

    # Inversion de x (Ouest à droite)
    # x_plot = -x
    return x, y, visible 

def format_lat_long_coordinates(df, col='Location'):
    """
    Transforms a string column 'SXXWXX' into columns 'lat' and 'long'.
    
    - S/N: sign of latitude (S = negative, N = positive)
    - W/E: sign of the longitude (W = positive, E = negative) --> solar convention
    - If latitude or longitude is missing, the default value is 0.
    """
    lat_pattern = r'([NS])(\d+\.?\d*)'
    long_pattern = r'([EW])(\d+\.?\d*)'
    
    def parse(s):
        # Latitude
        lat_match = re.search(lat_pattern, s)
        if lat_match:
            lat_sign, lat_val = lat_match.groups()
            lat = float(lat_val) * (1 if lat_sign == 'N' else -1)
        else:
            lat = 0.0
        
        # Longitude
        long_match = re.search(long_pattern, s)
        if long_match:
            long_sign, long_val = long_match.groups()
            long = float(long_val) * (1 if long_sign == 'W' else -1)
        else:
            long = 0.0
        
        return pd.Series([lat, long])
    
    df[['lat', 'long']] = df[col].apply(parse)
    return df


def value_to_color(value, vmin, vmax, cmap=plt.cm.viridis):
    """Convertit une valeur en couleur selon un gradient (cmap)."""
    if vmax == vmin:
        norm_val = 0.5
    else:
        norm_val = (value - vmin) / (vmax - vmin)
    return cmap(norm_val)


def _project(lat_deg, lon_deg):
    """Projection (lat, lon) -> (x, y) sur le disque solaire (vue de face)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.sin(lon)
    y = -np.sin(lat)
    return x, y



def draw_sepflare_probability(probability_matrix,
                              grid_resolution=np.arange(-90, 91, 30),
                              figsize=(10, 10),
                              title=None,
                              ax=None,
                              label=None,
                              cmap=plt.cm.viridis,
                              vmax=None):
    """
    Trace le disque solaire, la grille (grid_resolution) et colorie chaque
    case selon la valeur correspondante de probability_matrix (gradient + légende).
    La valeur de chaque case est écrite au centre.

    Si show_uncertainty_table=True, la matrice `uncertainty` est affichée
    sous forme d'un tableau 6x6 sous le plot du soleil.

    probability_matrix : matrice (n-1) x (n-1) avec n = len(grid_resolution).
        - lignes  -> latitude (du haut -90 vers le bas +90)
        - colonnes-> longitude (de gauche -90 vers la droite +90)
    """
    # ─── Validation ───────────────────────────────────────────────────────
    n_cells = len(grid_resolution) - 1
    if probability_matrix.shape != (n_cells, n_cells):
        raise ValueError(
            f"probability_matrix doit être de taille {n_cells}x{n_cells} "
            f"(grid_resolution a {len(grid_resolution)} bornes), "
            f"mais a la forme {probability_matrix.shape}."
        )

    # ─── Figure ───────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ─── Échelle de couleurs ──────────────────────────────────────────────
    finite_vals = probability_matrix[np.isfinite(probability_matrix)]
    vmin = np.nanmin(finite_vals)
    if vmax is None:
        vmax = np.nanmax(finite_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ─── Coloriage des cases (projetées) ──────────────────────────────────
    n_pts = 30
    for i in range(n_cells):           # ligne = latitude
        lat_low = grid_resolution[i]
        lat_high = grid_resolution[i + 1]
        for j in range(n_cells):       # colonne = longitude
            lon_low = grid_resolution[j]
            lon_high = grid_resolution[j + 1]

            value = probability_matrix[i, j]

            lat_seq = np.linspace(lat_low, lat_high, n_pts)
            lon_seq = np.linspace(lon_low, lon_high, n_pts)

            x1, y1 = _project(np.full(n_pts, lat_low), lon_seq)
            x2, y2 = _project(lat_seq, np.full(n_pts, lon_high))
            x3, y3 = _project(np.full(n_pts, lat_high), lon_seq[::-1])
            x4, y4 = _project(lat_seq[::-1], np.full(n_pts, lon_low))

            xs = np.concatenate([x1, x2, x3, x4])
            ys = np.concatenate([y1, y2, y3, y4])

            verts = np.column_stack([xs, ys])

            if np.isnan(value):
                facecolor = (0.9, 0.9, 0.9, 0.3)
            else:
                facecolor = value_to_color(value, vmin, vmax, cmap)

            poly = Polygon(verts, closed=True, facecolor=facecolor,
                           edgecolor='none', alpha=0.85, zorder=1)
            ax.add_patch(poly)

            lat_c = 0.5 * (lat_low + lat_high)
            lon_c = 0.5 * (lon_low + lon_high)
            xc, yc = _project(lat_c, lon_c)

            if not np.isnan(value):
                r, g, b, _ = facecolor
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = 'white' if luminance < 0.5 else 'black'
                ax.text(xc, yc, f'{value:.1f}',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color=txt_color, zorder=6)

    # ─── Disque solaire ───────────────────────────────────────────────────
    sun_circle = Circle((0, 0), 1, color='orange', fill=False,
                        linewidth=2.5, zorder=5)
    ax.add_patch(sun_circle)

    # ─── Grille adaptée à grid_resolution ─────────────────────────────────
    for lat in grid_resolution:
        if abs(lat) >= 90:
            continue
        lat_seq = np.full(200, lat)
        lon_seq = np.linspace(grid_resolution[0], grid_resolution[-1], 200)
        xg, yg = _project(lat_seq, lon_seq)
        col = 'red' if lat == 0 else 'gray'
        lw = 1.8 if lat == 0 else 1.0
        ls = '-' if lat == 0 else '--'
        ax.plot(xg, yg, color=col, linewidth=lw, linestyle=ls,
                alpha=0.6, zorder=3)

    for lon in grid_resolution:
        lat_seq = np.linspace(grid_resolution[0], grid_resolution[-1], 200)
        lon_arr = np.full(200, lon)
        xg, yg = _project(lat_seq, lon_arr)
        ax.plot(xg, yg, color='gray', linewidth=1.0, linestyle='--',
                alpha=0.6, zorder=3)

    # ─── Annotations d'axes ───────────────────────────────────────────────
    ax.text(-1.15, 0, 'E\n(East)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text(1.15, 0, 'W\n(West)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text(0, 1.12, 'N', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='steelblue')
    ax.text(0, -1.12, 'S', ha='center', va='top',
            fontsize=11, fontweight='bold', color='steelblue')

    # ─── Légende (colorbar = gradient) ────────────────────────────────────
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_label = label if label is not None else 'Probability ratio'
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=11)

    # ─── Mise en page ─────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax





def plot_sep_over_flare_probability(lat_sep, long_sep, lat_flares, long_flares,
                                    label="Probabilty (%)", title="SEP / Flare probability ratio", cmap=plt.cm.Reds, 
                                    grid_resolution=np.arange(-90, 91, 30), vmax=100, ax=None):
    """
    Plot the ratio of Solar Energetic Particle (SEP) events to solar flare events as a probability percentage.

    This function calculates the spatial probability distribution of SEP events relative to solar flare locations
    and visualizes it as a 2D sun heatmap.

    Parameters:
    -----------
    lat_sep : array-like
        Latitude coordinates of SEP events
    long_sep : array-like
        Longitude coordinates of SEP events
    lat_flares : array-like
        Latitude coordinates of solar flare events
    long_flares : array-like
        Longitude coordinates of solar flare events
    label : str, optional
        Label for the colorbar (default: "Probabilty (%)")
    title : str, optional
        Title for the plot (default: "SEP / Flare probability ratio")
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use for the heatmap (default: plt.cm.Reds)
    grid_resolution : array-like, optional
        Grid resolution for the 2D histogram bins (default: np.arange(-90, 91, 30))
    vmax : float, optional
        Maximum value for the color scale (default: 30)
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure (default: None)

    Returns:
    --------
    tuple : (ax, probability_matrix, lat_long_sep, lat_long_flares)
        - ax: matplotlib.axes.Axes object containing the plot
        - probability_matrix: 2D array of SEP/flare probability percentages
        - lat_long_sep: 2D histogram of SEP event counts
        - lat_long_flares: 2D histogram of flare event counts
    """

    # Create 2D histograms for SEP and flare coordinates using the specified grid resolution
    lat_long_sep, _, _ = np.histogram2d(
        lat_sep, long_sep, bins=[grid_resolution, grid_resolution]
    )

    # Create 2D histogram for flare coordinates
    lat_long_flares, _, _ = np.histogram2d(
        lat_flares, long_flares, bins=[grid_resolution, grid_resolution]
    )

    # Convert counts to integers for cleaner display
    lat_long_sep = lat_long_sep.astype(int)
    lat_long_flares = lat_long_flares.astype(int)

    # Calculate the probability matrix by dividing SEP counts by flare counts and converting to percentage
    probability_matrix = (lat_long_sep/lat_long_flares)*100
    
    #Calculate the uncertainty (binomiale)
    confidence_intervals, lower_intervals, upper_intervals = binomiale_uncertainty(lat_long_sep, lat_long_flares)
    

    # Call the visualization function to create the heatmap plot
    ax = draw_sepflare_probability(probability_matrix,
                                   title=title,
                                   label=label,
                                   cmap=cmap,
                                   ax=ax,
                                   vmax=vmax)

    # If no axis was provided, show the plot and adjust layout
    if ax is None:
        plt.tight_layout()
        plt.show()

    # Return the plot axis, probability matrix, and both count matrices
    return ax, probability_matrix, lat_long_sep, lat_long_flares




def plot_sep_uncertainty(lat_sep, long_sep, lat_flares, long_flares,
                         grid_resolution=np.arange(-90, 91, 30),
                         figsize=(12, 8),
                         ax=None, 
                         type_uncertainty = 'binomiale'):
    """
    Calcule la matrice de probabilité SEP/flares ainsi que les intervalles
    de confiance binomiaux, puis affiche la matrice confidence_intervals
    (de type [lower, upper]) sous forme d'un tableau.

    Parameters
    ----------
    lat_sep, long_sep : array-like
        Coordonnées (latitude, longitude) des SEP.
    lat_flares, long_flares : array-like
        Coordonnées (latitude, longitude) des flares.
    grid_resolution : array-like
        Bornes de la grille (latitude/longitude).
    figsize : tuple
        Taille de la figure (utilisée seulement si ax est None).
    title : str
        Titre du tableau.
    ax : matplotlib.axes.Axes, optional
        Axe sur lequel dessiner. Si None, un nouvel axe est créé.

    Returns
    -------
    ax : l'axe matplotlib.
    """
    # ─── Histogrammes 2D SEP et flares ────────────────────────────────────
    lat_long_sep, _, _ = np.histogram2d(
        lat_sep, long_sep, bins=[grid_resolution, grid_resolution]
    )

    lat_long_flares, _, _ = np.histogram2d(
        lat_flares, long_flares, bins=[grid_resolution, grid_resolution]
    )

    # ─── Matrice de probabilité (en %) ────────────────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        probability_matrix = (lat_long_sep / lat_long_flares) * 100

    # ─── Intervalles de confiance (binomiale) ─────────────────────────────
    if type_uncertainty == 'binomiale': 
        confidence_intervals, lower_intervals, upper_intervals = \
            binomiale_uncertainty(lat_long_sep, lat_long_flares)
    elif type_uncertainty == 'wilson':
        confidence_intervals, lower_intervals, upper_intervals = \
            wilson_uncertainty(lat_long_sep, lat_long_flares)
    
    # ─── Axe ──────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # ─── Affichage du tableau ─────────────────────────────────────────────
    n = confidence_intervals.shape[0]

    # Libellés des lignes (latitude) et colonnes (longitude)
    row_labels = [f'[{grid_resolution[i]}, {grid_resolution[i+1]}]'
                  for i in range(n)]
    col_labels = [f'[{grid_resolution[j]}, {grid_resolution[j+1]}]'
                  for j in range(n)]

    # Contenu : affichage direct des chaînes [lower, upper]
    cell_text = [[str(confidence_intervals[i, j]) for j in range(n)]
                 for i in range(n)]

    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    
    for (row, col), cell in table.get_celld().items(): 
        if row == 0 or col == -1:
            cell.set_text_props(fontweight='bold')
    # Légende des axes du tableau
    
    if type_uncertainty == 'binomiale':  
        ax.set_title("Binomiale confidence intervals", fontsize=13, fontweight='bold')
    
    elif type_uncertainty == 'wilson':
        ax.set_title("Wilson confidence intervals", fontsize=13, fontweight='bold')

    return ax

from scipy.stats import norm

#%% Uncertainties

def binomiale_uncertainty(k_matrix, n_matrix, confidence=0.95, decimals=1):
    """
    Returns a 6x6 matrix of strings in the format"[inf, sup]".

    Settings:
    - k_matrix: Number 6x6 matrix of success counts (B with A)
    - n_matrix: Matrix numpy 6x6 of total counts (A)
    - confidence: Confidence level (default 0.95)
    - decimals: Number of decimal places for display (default is 3)

    Returns:
    - Numpy 6x6 matrix of character strings in the format "[inf, sup]
    """
    alpha = 1 - confidence

    lower = beta.ppf(alpha/2, k_matrix, n_matrix - k_matrix + 1)
    lower*=100
    upper = beta.ppf(1 - alpha/2, k_matrix + 1, n_matrix - k_matrix)
    upper*=100

    formatted = np.array([
        [f"[{round(lower[i,j], decimals)}, {round(upper[i,j], decimals)}]"
         for j in range(6)]
        for i in range(6)
    ])

    return formatted, lower, upper

def wilson_uncertainty(successes, trials, confidence=0.95, decimals=1):
    """
    Calcule les intervalles de confiance de Wilson score pour une proportion
    binomiale, élément par élément sur des matrices.

    Parameters
    ----------
    successes : array-like
        Nombre de succès (ex : SEP).
    trials : array-like
        Nombre d'essais (ex : flares).
    confidence : float
        Niveau de confiance (par défaut 0.95).
    decimals : int
        Nombre de décimales pour l'arrondi (en %).

    Returns
    -------
    confidence_intervals : ndarray of str
        Matrice de chaînes "[lower, upper]" (en %).
    lower_intervals : ndarray of float
        Bornes inférieures (en %).
    upper_intervals : ndarray of float
        Bornes supérieures (en %).
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    # Quantile de la loi normale
    z = norm.ppf(1 - (1 - confidence) / 2)

    lower_intervals = np.full(successes.shape, np.nan)
    upper_intervals = np.full(successes.shape, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_hat = successes / trials

    # Calcul de Wilson uniquement là où trials > 0
    mask = trials > 0
    n = trials
    p = p_hat

    # Terme central et demi-largeur
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt(
        (p * (1 - p) / n) + (z**2) / (4 * n**2)
    )

    lower = (center - half_width) * 100
    upper = (center + half_width) * 100

    lower_intervals[mask] = np.clip(lower[mask], 0, 100)
    upper_intervals[mask] = np.clip(upper[mask], 0, 100)

    # Construction de la matrice de chaînes "[lower, upper]"
    confidence_intervals = np.empty(successes.shape, dtype=object)
    for idx in np.ndindex(successes.shape):
        if mask[idx]:
            lo = round(lower_intervals[idx], decimals)
            up = round(upper_intervals[idx], decimals)
            confidence_intervals[idx] = f'[{lo}, {up}]'
        else:
            confidence_intervals[idx] = 'N/A'

    return confidence_intervals, lower_intervals, upper_intervals


#%% Events time

def time_mean(df1, df2, diff_max = 20):
    """
    To have a positive difference, you suppose that df1 happens earlier than df2

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.

    Returns
    -------
    mean : TYPE
        DESCRIPTION.

    """
    df1 = pd.to_datetime(df1, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df2 = pd.to_datetime(df2, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df3 = pd.DataFrame({
        'df1': df1,
        'df2': df2
    })
    df_clean = df3.dropna(subset=['df1'])
    df_clean = df_clean.dropna(subset=['df2'])
    
    df_clean['difference'] =  df_clean['df2'] - df_clean['df1'] 
    
    limite = pd.Timedelta(days=diff_max) 
    df_clean = df_clean[df_clean['difference'].abs() <= limite]
    
    mean = df_clean['difference'].mean()
    median = df_clean['difference'].median()
    
    return mean, df_clean, median

#%% Correlation

from scipy.stats import pearsonr, spearmanr

def print_corr(df, col_x, col_y, title=None):
    r_p, p_p = pearsonr(df[col_x], df[col_y])
    r_s, p_s = spearmanr(df[col_x], df[col_y])
    print(f"{title}")
    print(f"  Pearson  r = {r_p:+.3f} (p={p_p:.2e})")
    print(f"  Spearman r = {r_s:+.3f} (p={p_s:.2e})")
    print("---------------------------------------------------")
    return r_p, p_p, r_s, p_s


def correlation_matrix(df, columns, method='pearson', plot=False, interactive = True, cr = True, annotations = True):
    """
    Generates a correlation matrix (Pearson or Spearman) for the specified columns,
    by automatically ignoring null values (NaN, NaT, etc.).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): A list of column names to include in the matrix.
        method (str): 'pearson' or 'spearman'.
        plot (bool): If True, displays the lower triangular matrix with a heatmap.

    Returns:
        pd.DataFrame: Correlation matrix as a DataFrame.
    """
    n = len(columns)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

    for i, col_x in enumerate(columns):
        for j, col_y in enumerate(columns):
            if i <= j:  # On ne calcule que la moitié supérieure (la matrice est symétrique)
                # Filtrer les paires de valeurs non nulles
                mask = df[col_x].notna() & df[col_y].notna()
                x = df.loc[mask, col_x]
                y = df.loc[mask, col_y]

                if len(x) < 2 or len(y) < 2:
                    r = np.nan
                else:
                    if method == 'pearson':
                        r, _ = pearsonr(x, y)
                    elif method == 'spearman':
                        r, _ = spearmanr(x, y)
                    else:
                        raise ValueError("The method must be 'pearson' or 'spearman'.")
                    r *= 100  # Conversion en pourcentage

                corr_matrix.loc[col_x, col_y] = r
                corr_matrix.loc[col_y, col_x] = r  # Symétrie

    if plot:
        # Extraire la partie en bas à gauche (exclut la diagonale)
        lower_triangle = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))

        # Prendre la valeur absolue pour la couleur
        abs_lower_triangle = lower_triangle.abs()
        fig, ax = plt.subplots(figsize=(15, 10))
        im = ax.imshow(
            abs_lower_triangle,
            cmap='Purples',
            aspect='auto',
            alpha=0.6,
            vmin=0, vmax=75
        )
        fig.colorbar(im, ax=ax, label='|Correlation (%)|')
        
        # Annotations with sign
        if annotations:
            for i in range(n):
                for j in range(n):
                    if i > j:
                        value = lower_triangle.iloc[i, j]
                        ax.text(j, i, f"{value:+.0f}",
                                ha='center', va='center', color='black', fontsize=8)
            
        ax.set_xticks(range(n))
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(columns)
        ax.set_title(f"correlation matrix ({method}) - GSEP parameters")
        plt.tight_layout()

 
        if interactive:
            def on_click(event):
                # Ignore clicks outside the axes
                if event.inaxes != ax:
                    return
                # Round click coordinates to the nearest cell
                j = int(round(event.xdata))
                i = int(round(event.ydata))
                # Only react to valid lower-triangle cells
                if 0 <= i < n and 0 <= j < n and i > j:
                    col_x = columns[j]  # x-axis parameter
                    col_y = columns[i]  # y-axis parameter
                    scatter_parameters(
                        df[col_x], df[col_y],
                        name_p1=col_x,
                        name_p2=col_y, 
                        cr = cr, 
                    )
 
            fig.canvas.mpl_connect('button_press_event', on_click)
 
        plt.show()
 
    return corr_matrix


def scatter_parameters(p1, p2, name_p1='name_p1', name_p2='name_p2', cr = True):
    """
    Plots two scatter plots (linear and log-log) of p2 vs p1, each with a
    polynomial fit, in a NEW figure.
 
    Args:
        cr (bool): if True, min-max normalize p1/p2 to [0, 1] using
            nan-aware min/max (a single NaN in the raw data must NOT be
            allowed to poison every value: plain np.min/np.max return NaN
            if the array contains even one NaN, which would turn the whole
            normalized array into NaN).
 
    Returns:
        (a, b): fit coefficients from the linear-scale fit (computed on the
        normalized data if cr=True).
    """
    p1_raw = np.asarray(p1, dtype=float)
    p2_raw = np.asarray(p2, dtype=float)
 
    if cr:
        p1_min, p1_max = np.nanmin(p1_raw), np.nanmax(p1_raw)
        p2_min, p2_max = np.nanmin(p2_raw), np.nanmax(p2_raw)
        p1_raw = (p1_raw - p1_min) / (p1_max - p1_min)
        p2_raw = (p2_raw - p2_min) / (p2_max - p2_min)
 
    p1_lin, p2_lin = p1_raw, p2_raw
 
    # --- Clean data for the linear fit: 
    finite_mask = np.isfinite(p1_lin) & np.isfinite(p2_lin)
    p1_clean = p1_lin[finite_mask]
    p2_clean = p2_lin[finite_mask]
 
    order = np.argsort(p1_clean)
    p1_sorted = p1_clean[order]
    p2_sorted = p2_clean[order]
    
    # --- Clean data for the log fit 
    finite_mask_raw = np.isfinite(p1_raw) & np.isfinite(p2_raw)
    positive_mask = finite_mask_raw & (p1_raw > 0) & (p2_raw > 0)
    p1_pos = p1_raw[positive_mask]
    p2_pos = p2_raw[positive_mask]
    order_log = np.argsort(p1_pos)
    p1_pos_sorted = p1_pos[order_log]
    p2_pos_sorted = p2_pos[order_log]
 
    # Polyfit Linear 
    a, b = (np.nan, np.nan)
    a, b = np.polyfit(p1_sorted, p2_sorted, 1)
    fit_vals = a * p1_sorted + b
    
    # Polyfit 2nd polynomial
    c,d,e = (np.nan, np.nan, np.nan)
    c,d,e = np.polyfit(p1_sorted, p2_sorted, 2)
    fit_vals_2 = c * p1_sorted**2 + d * p1_sorted + e
    
    # Polyfit Log
    if len(p1_pos_sorted) > 2 and np.ptp(np.log10(p1_pos_sorted)) > 0:
        a_log, b_log = np.polyfit(np.log10(p1_pos_sorted), np.log10(p2_pos_sorted), 1)
        fit_vals_log = 10 ** (np.polyval([a_log, b_log], np.log10(p1_pos_sorted)))  
    
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    
    # Left plot
    ax[0].scatter(p1_clean, p2_clean, marker='x', color='purple', alpha=0.5)
    ax[0].plot(p1_sorted, fit_vals, '-.', color='blue', alpha=0.9, label=f'DEG 1: {a:.2e} x + {b:.2e}')
    ax[0].plot(p1_sorted, fit_vals_2, '-.', color='red', alpha=0.9, label=f'DEG 2: {c:.2e} x² + {d:.2e} x + {e:.2e}')
    ax[0].plot(p1_pos_sorted, fit_vals_log, '-', color='green', alpha=0.9, label=f'LOG: $10^{{{b_log:.2f}}} \\cdot x^{{{a_log:.2f}}}$')
    ax[0].set_xlabel(f'{name_p1}')
    ax[0].set_ylabel(f'{name_p2}')
    ax[0].legend()
    ax[0].grid()
    
    # Right plot
    ax[1].scatter(p1_pos, p2_pos, marker='x', color='purple', alpha=0.5)
    ax[1].plot(p1_sorted, fit_vals, '-.', color='blue', alpha=0.9, label=f'DEG 1:{a:.2e} x + {b:.2e}')
    ax[1].plot(p1_sorted, fit_vals_2, '-.', color='red', alpha=0.9, label=f'DEG 2:{c:.2e} x² + {d:.2e} x + {e:.2e}')
    ax[1].plot(p1_pos_sorted, fit_vals_log, '-', color='green', alpha=0.9, label=f'LOG: $10^{{{b_log:.2f}}} \\cdot x^{{{a_log:.2f}}}$')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel(f'log({name_p1})')
    ax[1].set_ylabel(f'log({name_p2})')
    ax[1].legend()
    ax[1].grid(which='both')
    
    
    fig.suptitle(f'{name_p2} = f({name_p1})')
    plt.tight_layout()
    plt.show()
 
    return a, b, c, d, e



#%% Flux Timeseries



def plot_SEP_event(sep_dictionary, index_sep=None, date_time_sep=None, 
                   int_channel = False, channels=[1],
                   log_xray = False, log_diff_channels = False, log_int_channel = False, 
                   colors_channels=None, color_xray='red', color_int = 'darkgreen', 
                   save_fig_path=None, ax=None):   
    
    
    # ------------ Understanding the request -------------------------------------
    if index_sep is None and date_time_sep is None:
        print("Insert an SEP event indicator (index or date)")
        return
    elif index_sep is not None and date_time_sep is not None:
        print("One SEP indicator is enough (index or date)")
        return

    if index_sep is not None:
        keys = list(sep_dictionary.keys())
        SEP_event = sep_dictionary[keys[index_sep]]
    else:
        SEP_event = sep_dictionary[f'{date_time_sep}']

    # --- Create the figure only if no ax is provided ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))
    else:
        fig = ax.figure
        
    #------------ X-ray flux -------------------------------------------
    ax.plot(SEP_event['xrayl'], color=color_xray, alpha=0.6, label='X-ray flux (GOES) [5\']')

    ax.set_ylim(0, SEP_event['xrayl'].max() * 1.2)
    
    if log_xray == True: 
        ax.set_yscale('log')  
        
    ax.set_ylabel('X-ray flux (W/m²)', color=color_xray)
    ax.tick_params(axis='y', labelcolor=color_xray)
    
    classification_flares = ['C', 'M', 'X']
    
    for threshold, label in zip((1e-6, 1e-5, 1e-4), classification_flares):
        ax.axhline(y=threshold, color=color_xray, linestyle='--', linewidth=0.8, alpha=0.5)
        
        if ax.get_ylim()[0] <= threshold <= ax.get_ylim()[1]:  
            ax.text(
                ax.get_xlim()[0]+0.05,
                threshold,
                label,
                color=color_xray,
                fontsize=7,
                va='bottom',
                ha='left',
                alpha=0.7,
            )

        
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

    
    #------------ Differential Channels -------------------------------------------
    ax2 = ax.twinx()
    
    label_diff_channels = ['5.00-7.23 MeV [5\']', '7.23-10.46 MeV [5\']', '10.46-15.12 MeV [5\']', 
                           '15.12-21.87 MeV [5\']', '21.87-31.62 MeV  [5\']', '45.73-66.13 MeV [5\']',
                           '66.13-95.64 MeV [5\']', '95.64-138.3 MeV [5\']', '138.3-200.0 MeV [5\']',
                           '200.0-289.2 MeV [5\']']
    
    if colors_channels is None:
        colors_channels = [f'C{idx}' for idx in range(len(channels))]
    
    for i, color_channel in zip(channels, colors_channels):
        ax2.plot(SEP_event[f'F{i}'], color=color_channel, label= label_diff_channels[i])
    
     
    smallest_channel = min(channels)
    ref_data = SEP_event[f'F{smallest_channel}']
    scale_max = ref_data.max() * 1.2
    
    ax2.set_ylim(0, scale_max)
    
    if log_diff_channels == True: 
        ax2.set_yscale('log')
        
        
    ax2.set_ylabel(f'Differential channels F{channels} (cm-2.s-1.sr-1.MeV-1)', color=colors_channels[0])
    ax2.tick_params(axis='y', labelcolor=colors_channels[0])
    
    #------------ Integral channel -------------------------------------------
    if int_channel == True:
        fig.subplots_adjust(right=0.80)
        
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.10))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.plot(SEP_event['ZPGT10W_CORR'], color=color_int, alpha = 0.9, label='>10MeV [5\']')
        
        ax3.set_ylim(0, SEP_event['ZPGT10W_CORR'].max()*1.2)
        
        if log_int_channel == True: 
            ax3.set_yscale('log')
        
        ax3.set_ylabel('Integral channel >10 MeV (pfu = cm-2.s-1.sr-1)', color=color_int)
        ax3.tick_params(axis='y', labelcolor=color_int)
        
        ax3.axhline(y=10, color=color_int, linestyle='--', linewidth=0.8, alpha=0.5)
        
        if ax3.get_ylim()[0] <= 10 <= ax3.get_ylim()[1]:  
            ax3.text(
                    ax3.get_xlim()[-1]-0.2,
                    10,
                    'SWPC threshold',              
                    color=color_int,
                    fontsize=10,
                    va='bottom',
                    ha='left',
                    alpha=0.7,
                ) 
           
    
    #--------------------time events------------------------------------------------------
    # Tracé des trois traits verticaux pour les événements temporels
    time_events = [
        ('GSEP_timestamp', 'GSEP timestamp', 'k', '-'),
        ('cdaw_start_time', 'CDAW start time', 'olive', '-'),
        ('cdaw_max_time', 'CDAW max time', 'darkgreen', '-'),
        ('cme_1st_app_time', 'CME 1st app time', 'sienna', '--'), 
        ('cme_launch_time', 'CME launch time', 'brown', '--'), 
        ('fl_start_time', 'Flare start time', 'red', '-.'), 
        ('fl_peak_time', 'Flare peak time', 'darkred', '-.'), 
    ]
    
    for key, label_event, color_event, linestyle in time_events:
        value = SEP_event[key].iloc[0] 
       
        if pd.isna(value):
            continue
        
        time_value = pd.to_datetime(value)  
        ax.axvline(x=time_value, color=color_event, linestyle=linestyle, 
                   linewidth=2.5, alpha=0.25, label=label_event)
        
        
    # Récupération des handles et labels APRÈS avoir tracé les axvlines
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    if int_channel == True:
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        plt.setp(ax3.get_xticklabels(), visible=False) 
    else:
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.setp(ax2.get_xticklabels(), visible=False) 

    
    #--------------------titles------------------------------------------------------

    if int_channel == True: 
        ax.set_title(f'X ray flux, F{channels} & >10MeV ')
        
    else:
        ax.set_title(f'X ray flux & F{channels}')
    
    fig.tight_layout()
    
    if save_fig_path is not None:
        fig.savefig(f'{save_fig_path}')

    return ax  


def plot_SEP_event_interactive_keys(sep_dictionary, channels=[1], colors_channels=None, log=False, int_channel=False, 
                                    log_xray = False, log_diff_channels = False, log_int_channel = False, color_int = 'darkgreen',
                                     color_xray='red', start_index=0, max_index=423):
    keys = list(sep_dictionary.keys())
    state = {'index': start_index}
    selected_indexes = []

    fig, ax = plt.subplots(figsize=(15, 8))

    def draw(j):
        ax.clear()

        for extra_ax in fig.axes:
            if extra_ax is not ax:
                extra_ax.remove()

        plot_SEP_event(sep_dictionary, 
                       index_sep=j,
                       int_channel=int_channel,
                       channels=channels,
                       log_xray=log_xray, 
                       log_diff_channels=log_diff_channels, 
                       log_int_channel=log_int_channel,
                       colors_channels=colors_channels, 
                       color_xray=color_xray, 
                       color_int=color_int,
                       ax=ax
                       )

        ax.set_title(
            f"X-ray flux & F{channels} & >10MeV channel  |  Index {j}/{max_index-1}  "
            f"({keys[j]})  -  SPACE: skip  |  ENTER: select"
        )
        fig.tight_layout()
        fig.canvas.draw_idle()

    def next_image():
        state['index'] += 1
        if state['index'] > max_index:
            print("Last index reached. Closed.")
            print("Selected indexes:", selected_indexes)
            plt.close(fig)
            return
        draw(state['index'])

    def on_key(event):
        if event.key == ' ':            
            next_image()
        elif event.key == 'enter':      # 
            selected_indexes.append(state['index'])
            print(f"Index {state['index']} saved ({keys[state['index']]})")
            next_image()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(state['index'])
    plt.show()

    return selected_indexes

#%% Flare time density 

def compute_event_counts(flares_time_peak,
                         start='1976-03-14',
                         end='2025-01-31',
                         window_hours=1,    # <-- fenêtre paramétrable
                         step_hours=1):     # <-- pas d'échantillonnage
    """
    Counts the number of events in a preceding sliding window
    each sampling moment.

    Settings
    ----------
    flares_time_peak: pd.Series or array of Timestamps (event dates)
    start, end  : boundaries of the period
    window_hours  : width of calculation window (in hours)
    step_hours  : interval between two evaluated times (in hours)

    Returns
    --------
    DataFrame indexed by (date, hour) with a 'count_events' column
    """

    # --- 1. Préparer et trier les événements (nécessaire pour searchsorted) ---
    events = pd.to_datetime(pd.Series(flares_time_peak)).sort_values()
    events_ns = events.values.astype('datetime64[ns]')
    print("fin étape 1")

    # --- 2. Générer tous les instants d'échantillonnage ---
    sample_times = pd.date_range(start=start,
                                 end=pd.Timestamp(end) + pd.Timedelta(days=1),
                                 freq=f'{step_hours}h',
                                 inclusive='left')
    print("fin étape 2")

    # --- 3. Calcul vectorisé du nombre d'événements par fenêtre ---
    end_window = sample_times.values.astype('datetime64[ns]')
    start_window = (sample_times - pd.Timedelta(hours=window_hours)).values.astype('datetime64[ns]')

    # searchsorted donne la position d'insertion -> nombre d'événements
    # avant chaque borne, en O(log n) au lieu d'un filtrage complet
    idx_end = np.searchsorted(events_ns, end_window, side='left')    # < end_window
    idx_start = np.searchsorted(events_ns, start_window, side='left') # >= start_window

    counts = idx_end - idx_start
    print("fin étape 3")

    # --- 4. Construire le DataFrame résultat ---
    result = pd.DataFrame({
        'datetime': sample_times,
        'count_events': counts
    })
    result['date'] = result['datetime'].dt.normalize()
    result['hour'] = result['datetime'].dt.hour
    result = result.set_index(['date', 'hour'])[['count_events']]
    print("fin étape 4")

    return result


def plot_timeseries(df, resample=None, figsize=(15, 5)):
    """
    Trace count_events en fonction du temps.

    resample : None, 'D', 'W', 'ME', 'YE'... pour agréger
               (utile car 428k points = illisible et lourd)
    """
    # Reconstruire un index datetime à partir de (date, hour)
    s = df['count_events'].copy()
    dt_index = (s.index.get_level_values('date')
                + pd.to_timedelta(s.index.get_level_values('hour'), unit='h'))
    s.index = dt_index
    s = s.sort_index()

    # Agrégation optionnelle
    if resample:
        s = s.resample(resample).sum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=0.7, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel("Nombre d'événements")
    title = "Événements au cours du temps"
    if resample:
        title += f" (agrégé par '{resample}')"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import seaborn as sns

def plot_heatmap(df, figsize=(14, 8), cmap='inferno', log_scale=False):
    """
    Heatmap : axe X = heures, axe Y = jours (ou inversement).
    df doit être indexé par (date, hour).
    """
    # Passer de l'index multi à une matrice date × hour
    pivot = df['count_events'].unstack(level='hour')  # lignes=date, colonnes=hour

    # Optionnel : échelle logarithmique si valeurs très étalées
    data = np.log1p(pivot) if log_scale else pivot

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        cmap=cmap,
        cbar_kws={'label': 'log(count+1)' if log_scale else "Nombre d'événements"},
        ax=ax
    )
    ax.set_xlabel('Heure de la journée')
    ax.set_ylabel('Date')
    ax.set_title('Nombre d\'événements par heure')

    # Alléger l'axe Y (sinon 17855 labels illisibles)
    n_ticks = 15
    yticks = np.linspace(0, len(pivot) - 1, n_ticks, dtype=int)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels([pivot.index[i].strftime('%Y-%m') for i in yticks], rotation=0)

    plt.tight_layout()
    plt.show()
    
    
def plot_timeseries_overlay(df, resamples=('D', 'W', 'ME', 'YE'),
                            figsize=(15, 6), normalize=True):
    """
    Overlays multiple aggregations on a single axis.
    normalize=True -> reports each curve to a comparable daily average.
    """
    s = df['count_events'].copy()
    dt_index = (s.index.get_level_values('date')
                + pd.to_timedelta(s.index.get_level_values('hour'), unit='h'))
    s.index = dt_index
    s = s.sort_index()

    # Nb de jours par période pour ramener à une "moyenne par jour"
    days_per = {'D': 1, 'W': 7, 'ME': 30.44, 'YE': 365.25}
    labels = {'D': 'daily', 'W': 'weekly', 'ME': 'monthly', 'YE': 'annual'}
    colors = {'D': 'lightgray', 'W': 'steelblue', 'ME': 'darkorange', 'YE': 'crimson'}

    fig, ax = plt.subplots(figsize=figsize)

    for r in resamples:
        s_agg = s.resample(r).sum()
        if normalize:
            s_agg = s_agg / days_per[r]      # -> événements/jour moyens
        ax.plot(s_agg.index, s_agg.values,
                lw=1.2, label=labels.get(r, r), color=colors.get(r))

    ax.set_xlabel('Date')
    ax.set_ylabel("events/ day" if normalize else "events")
    ax.set_title("Time density (different resolutions)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
