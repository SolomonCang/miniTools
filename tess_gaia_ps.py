#!/usr/bin/env python
"""
tess_gaia_ps.py — Query the Gaia → TIC cross-match catalog to check if the star has been observed by TESS.
If observations exist, download all SAP light curves and plot the Lomb-Scargle periodogram for periods in the range of 50–2000 seconds.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
import lightkurve as lk
from pathlib import Path
import re
from astropy.table import QTable


# -------------------------------------------------------------------
# 1. Gaia → TIC
# -------------------------------------------------------------------
def gaia_to_tic(gaia_id: str) -> int:
    print(f"[+] Querying TIC catalog for Gaia {gaia_id} …")
    tbl = Catalogs.query_object(objectname=f"Gaia DR3 {gaia_id}",
                                catalog="TIC")
    if len(tbl) == 0:
        raise ValueError(
            "Could not find the cross-match identifier for this Gaia source in the TIC catalog."
        )
    tic = int(tbl[0]["ID"])
    print(f"[+] Matched TIC {tic}")
    return tic


# -------------------------------------------------------------------
# 2. Search for TESS Light Curves
# -------------------------------------------------------------------
def search_tess_lc(tic_id: int) -> lk.search.SearchResult:
    """
    Search first for 20 s, then 120 s, then 30 min cadence light curves.
    If none are found, return an empty SearchResult.
    """
    for exptime in (20, 120, 1800):
        srch = lk.search_lightcurve(f"TIC {tic_id}",
                                    mission="TESS",
                                    exptime=exptime)
        if len(srch) > 0:
            print(f"[+] Found {len(srch)} LC files at {exptime} s cadence")
            return srch
    return lk.search.SearchResult([])  # Empty result


# -------------------------------------------------------------------
# 3. Download + Stitch — Auto-fix corrupted files, force use of SAP_FLUX
# -------------------------------------------------------------------
def _estimate_size_mb(srch):
    for col in ("filesize", "size", "file_size"):
        if col in srch.table.colnames:
            return srch.table[col].sum() / 1024**2
    return None


def _clean_corrupt_file(err_msg: str):
    """Extract the path of a corrupted FITS file from the LightkurveError message and delete it."""
    m = re.search(r"(/[^ ]+\.fits)", err_msg)
    if m:
        p = Path(m.group(1))
        if p.exists():
            print(f"[!] Removing corrupt file: {p}")
            p.unlink(missing_ok=True)


def _to_sap_lightcurve(obj) -> lk.LightCurve:
    """
    Convert any downloaded object into a *SAP flux* LightCurve.

    1. If obj is already a LightCurve:
       - If it contains a 'sap_flux' column → copy it into a new LightCurve and replace the flux column with this column
       - Otherwise, use obj directly
    2. If obj is a TargetPixelFile → call to_lightcurve(flux_column='sap_flux')
    """
    if isinstance(obj, lk.LightCurve):
        if "sap_flux" in obj.colnames:  # Contains multiple flux columns
            tbl = QTable(obj)  # Copy as an astropy Table
            tbl["flux"] = tbl[
                "sap_flux"]  # Replace default flux column with SAP
            if "flux_err" in tbl.colnames and "sap_flux_err" in tbl.colnames:
                tbl["flux_err"] = tbl["sap_flux_err"]
            lc = obj.__class__(tbl)  # Retain LightCurve subclass
        else:
            lc = obj
    else:  # TargetPixelFile or similar
        lc = obj.to_lightcurve(flux_column="sap_flux")

    # Clean & normalize
    lc = lc.remove_nans().normalize(unit="ppm").remove_outliers(sigma=10)
    return lc


def download_and_stitch(srch: lk.search.SearchResult) -> lk.LightCurve:
    est = _estimate_size_mb(srch)
    print(f"[+] Downloading {len(srch)} files"
          f"{f' (~{est:.1f} MB)' if est else ''} …")

    lcs = []
    for i, prod in enumerate(srch, 1):
        for attempt in (1, 2):  # Retry up to two times
            try:
                obj = prod.download(quality_bitmask="default")
                lc = _to_sap_lightcurve(obj)
                break
            except lk.utils.LightkurveError as e:
                print(f"[!] Read failed for file {i}/{len(srch)} "
                      f"(attempt {attempt})")
                _clean_corrupt_file(str(e))
                if attempt == 2:
                    raise
        lcs.append(lc)

    stitched = lk.LightCurveCollection(lcs).stitch()
    return stitched


# -------------------------------------------------------------------
# 4. Plot 50–2000 s Period Power Spectrum
# -------------------------------------------------------------------
def plot_periodogram(lc: lk.LightCurve, gaia_id: str):
    min_p, max_p = 50, 2000  # seconds
    min_f = 1 / max_p * 86400  # cycles per day
    max_f = 1 / min_p * 86400
    pg = lc.to_periodogram(method="lombscargle",
                           minimum_frequency=min_f,
                           maximum_frequency=max_f,
                           oversample_factor=5)

    fig, ax = plt.subplots(figsize=(8, 4))
    pg.plot(ax=ax, lw=0.7)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Power")
    ax.set_title(
        f"Gaia {gaia_id} – TESS SAP LS Periodogram\n(Periods 50–2000 s)")
    ax.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 5. Command-Line Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python tess_gaia_ps.py <Gaia_ID>")

    gaia_id = sys.argv[1]

    try:
        tic_id = gaia_to_tic(gaia_id)
    except ValueError as e:
        sys.exit(str(e))

    srch = search_tess_lc(tic_id)
    if len(srch) == 0:
        sys.exit("This target has not been observed by TESS.")

    lc = download_and_stitch(srch)
    plot_periodogram(lc, gaia_id)
