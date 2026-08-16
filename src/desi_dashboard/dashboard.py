# /// script
# dependencies = [
#     "marimo>=0.23.14",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt
    from pathlib import Path

    from waters_desi import WatersRaw

    get_selected_pixels, set_selected_pixels = mo.state((0,))
    get_selected_mz_range, set_selected_mz_range = mo.state(None)
    get_mz_view, set_mz_view = mo.state(None)

    def coordinate_grid(imaging, values):
        """Place a per-pixel vector on the measured x/y coordinate grid."""
        x_unique = np.unique(imaging.x)
        y_unique = np.unique(imaging.y)
        grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        x_index = np.searchsorted(x_unique, imaging.x)
        y_index = np.searchsorted(y_unique, imaging.y)
        grid[y_index, x_index] = np.asarray(values, dtype=float)
        dx = float(np.median(np.diff(x_unique))) if x_unique.size > 1 else 1.0
        dy = float(np.median(np.diff(y_unique))) if y_unique.size > 1 else 1.0
        extent = (
            float(x_unique.min() - dx / 2),
            float(x_unique.max() + dx / 2),
            float(y_unique.min() - dy / 2),
            float(y_unique.max() + dy / 2),
        )
        return grid, extent, x_unique, y_unique

    def selection_range(selection):
        """Extract a sorted ``(low, high)`` x-interval from a ``mo.ui.matplotlib`` selection."""
        if not selection:
            return None
        if isinstance(selection, (list, tuple)):
            selection = selection[0] if selection else None
            if selection is None:
                return None
        low = high = None
        if getattr(selection, "x_min", None) is not None:
            low, high = float(selection.x_min), float(selection.x_max)
        elif getattr(selection, "vertices", None) is not None:
            xs = [float(point[0]) for point in selection.vertices]
            if xs:
                low, high = min(xs), max(xs)
        if low is None or high is None:
            return None
        return (low, high) if low <= high else (high, low)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 25_000_000

    dashboard_style = mo.Html("""
    <style>
      :root {--content-width-medium: min(1600px, calc(100vw - 2rem));}
      .desi-card {border: 1px solid var(--slate-4); border-radius: 12px; padding: 0.9rem;}
      .desi-muted {color: var(--slate-11); font-size: 0.9rem;}
    </style>
    """)
    return (
        Path,
        WatersRaw,
        coordinate_grid,
        dashboard_style,
        get_mz_view,
        get_selected_mz_range,
        get_selected_pixels,
        mo,
        np,
        plt,
        selection_range,
        set_mz_view,
        set_selected_mz_range,
        set_selected_pixels,
    )


@app.cell(hide_code=True)
def _(Path, mo):
    raw_browser = mo.ui.file_browser(
        initial_path=Path.cwd(),
        selection_mode="directory",
        multiple=False,
        label="Waters .raw dataset",
    )

    raw_browser
    return (raw_browser,)


@app.cell(hide_code=True)
def _(Path, WatersRaw, mo, np, raw_browser):
    mo.stop(
        not raw_browser.value,
        mo.md("Select a Waters `.raw` directory."),
    )

    selected_raw_path = Path(raw_browser.value[0].path)
    mo.stop(
        selected_raw_path.suffix.lower() != ".raw",
        mo.md("Select a Waters `.raw` directory."),
    )

    raw_dataset = WatersRaw(selected_raw_path)
    function_number = sorted(raw_dataset.functions)[0]
    ms_function = raw_dataset.function(function_number)
    imaging = raw_dataset.read_imaging()

    x_values = np.unique(imaging.x)
    y_values = np.unique(imaging.y)
    mean_feature_intensity = imaging.intensity.mean(axis=0)
    default_feature_index = int(np.argmax(mean_feature_intensity))
    return (
        default_feature_index,
        imaging,
        ms_function,
        raw_dataset,
        selected_raw_path,
        x_values,
        y_values,
    )


@app.cell(hide_code=True)
def _(
    dashboard_style,
    imaging,
    mo,
    ms_function,
    raw_dataset,
    selected_raw_path,
    x_values,
    y_values,
):
    _instrument = raw_dataset.header.get("Instrument", "Unknown instrument")
    _acquired = " ".join(
        part
        for part in (
            raw_dataset.header.get("Acquired Date", ""),
            raw_dataset.header.get("Acquired Time", ""),
        )
        if part
    )
    _header_stats = mo.hstack(
        [
            mo.stat(value=f"{imaging.intensity.shape[0]:,}", label="pixels"),
            mo.stat(value=f"{imaging.mz.size:,}", label="m/z features"),
            mo.stat(value=f"{x_values.size} × {y_values.size}", label="map grid"),
            mo.stat(value=ms_function.polarity or "—", label="polarity"),
        ],
        justify="space-around",
    )

    mo.vstack(
        [
            dashboard_style,
            mo.md("# DESI-MS imaging dashboard"),
            mo.md(
                f"**{selected_raw_path.name}**  \n"
                f"{_instrument}" + (f" · acquired {_acquired}" if _acquired else "")
            ),
            _header_stats,
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(controls, map_panel, metadata_panel, mo, spectrum_panel):
    imaging_tab = mo.hstack(
        [controls, map_panel, spectrum_panel],
        widths=[0.8, 1.45, 1.55],
        align="start",
        gap=1.0,
    )
    mo.ui.tabs(
        {
            "Imaging": imaging_tab,
            "Metadata": metadata_panel,
        },
        lazy=False,
    )
    return


@app.cell(hide_code=True)
def _(imaging, mo, set_selected_pixels):
    cmap_selector = mo.ui.dropdown(
        options=["inferno", "magma", "viridis", "plasma", "cividis", "turbo"],
        value="inferno",
        label="Colour map",
        full_width=True,
    )
    interpolation_selector = mo.ui.dropdown(
        options=["nearest", "bilinear", "bicubic"],
        value="nearest",
        label="Interpolation",
        full_width=True,
    )
    show_range_markers = mo.ui.checkbox(
        value=True,
        label="Highlight aggregated m/z",
    )

    pixel_slider = mo.ui.slider(
        start=0,
        stop=imaging.intensity.shape[0] - 1,
        value=0,
        step=1,
        show_value=True,
        include_input=True,
        label="Manual pixel",
        on_change=lambda value: set_selected_pixels((int(value),)),
        full_width=True,
    )
    show_pixel_markers = mo.ui.checkbox(
        value=True,
        label="Show selected-pixel markers",
    )
    show_native_scan = mo.ui.checkbox(
        value=True, label="Use native profile spectra"
    )

    controls = mo.vstack(
        [
            mo.md("### Display"),
            cmap_selector,
            interpolation_selector,
            show_range_markers,
            mo.md("### Pixel selection"),
            mo.md("Drag or lasso points on the ion image, or choose a pixel manually."),
            pixel_slider,
            show_pixel_markers,
            mo.md("### Spectrum source"),
            show_native_scan,
        ],
        gap=0.65,
    )
    return (
        cmap_selector,
        controls,
        interpolation_selector,
        pixel_slider,
        show_native_scan,
        show_pixel_markers,
        show_range_markers,
    )


@app.cell(hide_code=True)
def _(imaging, ms_function, np, show_native_scan):
    native_mode = bool(show_native_scan.value)
    native_bin = 0.01

    if native_mode:
        native_profiles = []
        for _scan_number in imaging.source_scan_numbers:
            _scan = ms_function.read_scan(int(_scan_number) - 1, nonzero=True)
            _scan.mz = _scan.mz.astype(np.float32)
            _scan.intensity = _scan.intensity.astype(np.float32)
            native_profiles.append(_scan)
        source_mz_low = min(float(scan.mz[0]) for scan in native_profiles)
        source_mz_high = max(float(scan.mz[-1]) for scan in native_profiles)
    else:
        native_profiles = None
        source_mz_low = float(np.nanmin(imaging.mz))
        source_mz_high = float(np.nanmax(imaging.mz))
    return (
        native_bin,
        native_mode,
        native_profiles,
        source_mz_high,
        source_mz_low,
    )


@app.cell(hide_code=True)
def _(get_selected_pixels, imaging, pixel_slider):
    _valid_selected = tuple(
        int(i)
        for i in get_selected_pixels()
        if 0 <= int(i) < imaging.intensity.shape[0]
    )
    selected_pixels = _valid_selected or (int(pixel_slider.value),)
    return (selected_pixels,)


@app.cell(hide_code=True)
def _(
    default_feature_index,
    full_mz_high,
    full_mz_low,
    get_selected_mz_range,
    imaging,
    native_mode,
    np,
):
    _stored_range = get_selected_mz_range()
    if _stored_range is None:
        _default_mz = float(imaging.mz[default_feature_index])
        _range_low, _range_high = _default_mz - 0.5, _default_mz + 0.5
    else:
        _range_low, _range_high = sorted(float(value) for value in _stored_range)
    selected_mz_low = min(max(_range_low, full_mz_low), full_mz_high)
    selected_mz_high = min(max(_range_high, selected_mz_low), full_mz_high)

    _in_range = np.flatnonzero(
        (imaging.mz >= selected_mz_low) & (imaging.mz <= selected_mz_high)
    )
    if _in_range.size:
        selected_feature_indices = tuple(
            int(i) for i in _in_range[np.argsort(imaging.mz[_in_range])]
        )
    else:
        selected_feature_indices = (
            int(
                np.argmin(
                    np.abs(imaging.mz - 0.5 * (selected_mz_low + selected_mz_high))
                )
            ),
        )
    selected_mz_values = imaging.mz[list(selected_feature_indices)]
    selected_feature_ids = tuple(
        int(imaging.feature_ids[i]) for i in selected_feature_indices
    )
    selected_mz_label = (
        f"{float(selected_mz_values[0]):.4f}"
        if not native_mode and len(selected_feature_indices) == 1
        else f"{selected_mz_low:.4f}\u2013{selected_mz_high:.4f}"
    )
    return (
        selected_feature_ids,
        selected_feature_indices,
        selected_mz_high,
        selected_mz_label,
        selected_mz_low,
        selected_mz_values,
    )


@app.cell(hide_code=True)
def _(
    full_mz_high,
    full_mz_low,
    imaging,
    native_bin,
    native_mode,
    native_profiles,
    np,
    selected_pixels,
):
    feature_mean_intensity = imaging.intensity[list(selected_pixels)].mean(axis=0)

    # Depends on the pixel selection only -- never on the aggregated m/z range --
    # so changing the range does not redo this work.
    if native_mode:
        if len(selected_pixels) == 1:
            _scan = native_profiles[selected_pixels[0]]
            spectrum_mz = np.asarray(_scan.mz, dtype=float)
            spectrum_intensity = np.asarray(_scan.intensity, dtype=float)
        else:
            # Profile axes differ per scan, so average onto a common grid. The grid
            # is uniform, so integer indexing + bincount beats np.histogram ~9x and
            # avoids concatenating every scan into one multi-GB temporary.
            _bin_count = int(round((full_mz_high - full_mz_low) / native_bin))
            _totals = np.zeros(_bin_count, dtype=float)
            for _pixel in selected_pixels:
                _scan = native_profiles[_pixel]
                # Index arithmetic in float64: doing it in float32 migrates
                # points across 0.01 Da bin boundaries (~3% of peak height) for
                # no speed gain, since bincount dominates.
                _index = (
                    (_scan.mz.astype(np.float64) - full_mz_low) / native_bin
                ).astype(np.intp)
                np.clip(_index, 0, _bin_count - 1, out=_index)
                _totals += np.bincount(
                    _index, weights=_scan.intensity, minlength=_bin_count
                )
            spectrum_mz = full_mz_low + native_bin * (np.arange(_bin_count) + 0.5)
            spectrum_intensity = _totals / len(selected_pixels)
    else:
        _order = np.argsort(imaging.mz)
        spectrum_mz = imaging.mz[_order]
        spectrum_intensity = feature_mean_intensity[_order]
    return feature_mean_intensity, spectrum_intensity, spectrum_mz


@app.cell(hide_code=True)
def _(
    imaging,
    native_mode,
    native_profiles,
    np,
    selected_feature_indices,
    selected_mz_high,
    selected_mz_low,
    spectrum_mz,
):
    if native_mode:
        # Exact integration of the window on each scan's own m/z axis.
        _window_low = np.float32(selected_mz_low)
        _window_high = np.float32(selected_mz_high)
        ion_values = np.array(
            [
                float(
                    scan.intensity[
                        np.searchsorted(scan.mz, _window_low, "left"):
                        np.searchsorted(scan.mz, _window_high, "right")
                    ].sum(dtype=np.float64)
                )
                for scan in native_profiles
            ]
        )
        intensity_label = "Summed native intensity"
    else:
        ion_values = imaging.intensity[:, list(selected_feature_indices)].sum(axis=1)
        intensity_label = (
            "Raw intensity"
            if len(selected_feature_indices) == 1
            else "Summed raw intensity"
        )

    aggregated_mask = (spectrum_mz >= selected_mz_low) & (spectrum_mz <= selected_mz_high)
    return aggregated_mask, intensity_label, ion_values


@app.cell(hide_code=True)
def _(get_mz_view, np, source_mz_high, source_mz_low):
    full_mz_low = float(np.floor(source_mz_low))
    full_mz_high = float(np.ceil(source_mz_high))
    mz_step = 0.1
    _stored_mz_view = get_mz_view()
    if _stored_mz_view is None:
        mz_view_low, mz_view_high = full_mz_low, full_mz_high
    else:
        _stored_low, _stored_high = sorted(float(value) for value in _stored_mz_view)
        mz_view_low = max(full_mz_low, min(full_mz_high - mz_step, _stored_low))
        mz_view_high = min(full_mz_high, max(mz_view_low + mz_step, _stored_high))
    return full_mz_high, full_mz_low, mz_step, mz_view_high, mz_view_low


@app.cell(hide_code=True)
def _(
    full_mz_high,
    full_mz_low,
    mo,
    mz_step,
    mz_view_high,
    mz_view_low,
    set_mz_view,
):
    mz_low_input = mo.ui.number(
        start=full_mz_low,
        stop=full_mz_high - mz_step,
        step=mz_step,
        value=mz_view_low,
        debounce=True,
        label="m/z from",
        on_change=lambda value: set_mz_view((float(value), mz_view_high)),
        full_width=True,
    )
    mz_high_input = mo.ui.number(
        start=full_mz_low + mz_step,
        stop=full_mz_high,
        step=mz_step,
        value=mz_view_high,
        debounce=True,
        label="m/z to",
        on_change=lambda value: set_mz_view((mz_view_low, float(value))),
        full_width=True,
    )
    reset_zoom_button = mo.ui.button(
        label="Reset zoom",
        on_click=lambda _: set_mz_view(None),
        full_width=True,
    )
    zoom_controls = mo.hstack(
        [mz_low_input, mz_high_input, reset_zoom_button],
        widths=[1, 1, 1],
        gap=0.5,
        align="end",
    )
    return (zoom_controls,)


@app.cell(hide_code=True)
def _(
    cmap_selector,
    coordinate_grid,
    imaging,
    intensity_label,
    interpolation_selector,
    ion_values,
    mo,
    native_mode,
    np,
    plt,
    selected_feature_ids,
    selected_feature_indices,
    selected_mz_high,
    selected_mz_label,
    selected_mz_low,
    selected_pixels,
    show_pixel_markers,
):
    ion_grid, ion_extent, _, _ = coordinate_grid(imaging, ion_values)
    _fig_map, _ax_map = plt.subplots(figsize=(7.2, 6.0))
    _map_artist = _ax_map.imshow(
        ion_grid,
        extent=ion_extent,
        origin="lower",
        cmap=cmap_selector.value,
        interpolation=interpolation_selector.value,
        aspect="equal",
    )
    if show_pixel_markers.value:
        _ax_map.scatter(
            imaging.x[list(selected_pixels)],
            imaging.y[list(selected_pixels)],
            s=76,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.4,
            label=f"{len(selected_pixels)} selected",
        )
    _ax_map.set(
        xlabel="x position",
        ylabel="y position",
        title=f"Ion image \u00b7 {selected_mz_label} m/z",
    )
    if show_pixel_markers.value:
        # Only artist with a label; skip the legend entirely when hidden.
        _ax_map.legend(loc="upper right", frameon=False, fontsize=8)
    _map_colorbar = _fig_map.colorbar(_map_artist, ax=_ax_map, fraction=0.046, pad=0.04)
    _map_colorbar.set_label(intensity_label)
    _fig_map.tight_layout()
    map_selection = mo.ui.matplotlib(_ax_map, debounce=True)
    plt.close(_fig_map)
    if native_mode:
        _feature_summary = (
            f"Native profile integrated over `{selected_mz_low:.4f}"
            f"\u2013{selected_mz_high:.4f}` m/z"
        )
    elif len(selected_feature_indices) == 1:
        _feature_summary = f"Feature **{selected_feature_ids[0]}**"
    else:
        _feature_summary = (
            f"**{len(selected_feature_indices)} features** in the selected range"
        )
    map_panel = mo.vstack(
        [
            mo.md(f"### Ion image \u00b7 `{selected_mz_label}` m/z"),
            map_selection,
            mo.md(
                f"{_feature_summary} \u00b7 intensity range "
                f"**{float(np.nanmin(ion_values)):,.0f}\u2013{float(np.nanmax(ion_values)):,.0f}**"
            ),
        ],
        gap=0.45,
    )
    return map_panel, map_selection


@app.cell(hide_code=True)
def _(imaging, map_selection, np, selected_pixels, set_selected_pixels):
    if map_selection.value:
        try:
            _selection_mask = map_selection.value.get_mask(imaging.x, imaging.y)
        except Exception:
            _selection_mask = None
        if _selection_mask is not None:
            _picked = tuple(int(i) for i in np.flatnonzero(_selection_mask))
            if _picked and _picked != selected_pixels:
                set_selected_pixels(_picked)
    return


@app.cell(hide_code=True)
def _(
    aggregated_mask,
    feature_mean_intensity,
    full_mz_high,
    full_mz_low,
    imaging,
    ion_values,
    mo,
    mz_view_high,
    mz_view_low,
    native_bin,
    native_mode,
    native_profiles,
    np,
    plt,
    selected_feature_indices,
    selected_mz_high,
    selected_mz_label,
    selected_mz_low,
    selected_mz_values,
    selected_pixels,
    show_range_markers,
    spectrum_intensity,
    spectrum_mz,
    zoom_controls,
):
    _fig_overview, _ax_overview = plt.subplots(figsize=(8.2, 1.35))
    _ax_overview.plot(spectrum_mz, spectrum_intensity, color="#52606d", linewidth=0.55)
    _ax_overview.axvspan(mz_view_low, mz_view_high, color="#2563eb", alpha=0.16)
    _ax_overview.set(
        xlim=(full_mz_low, full_mz_high),
        ylabel="",
        title="Full m/z overview \u00b7 drag to set the visible window",
    )
    _ax_overview.set_yticks([])
    _fig_overview.tight_layout()
    overview_selection = mo.ui.matplotlib(_ax_overview, debounce=True)
    plt.close(_fig_overview)

    _fig_spectrum, _ax_spectrum = plt.subplots(figsize=(8.2, 4.1))
    _ax_spectrum.plot(spectrum_mz, spectrum_intensity, color="#18212b", linewidth=0.8)
    if not native_mode and len(selected_feature_indices) == 1:
        _ax_spectrum.axvline(
            float(selected_mz_values[0]),
            color="crimson",
            linewidth=1.2,
            alpha=0.85,
            label=f"selected {selected_mz_label} m/z",
        )
    else:
        _ax_spectrum.axvspan(
            selected_mz_low,
            selected_mz_high,
            color="crimson",
            alpha=0.16,
            label=(
                f"aggregating {selected_mz_label} m/z"
                if native_mode
                else f"{len(selected_feature_indices)} selected features"
            ),
        )
    if show_range_markers.value:
        if native_mode:
            _ax_spectrum.fill_between(
                spectrum_mz,
                0.0,
                np.where(aggregated_mask, spectrum_intensity, 0.0),
                color="crimson",
                alpha=0.55,
                linewidth=0.0,
            )
        else:
            _ax_spectrum.scatter(
                selected_mz_values,
                feature_mean_intensity[list(selected_feature_indices)],
                color="crimson",
                s=24,
                zorder=3,
            )
    _visible = (spectrum_mz >= mz_view_low) & (spectrum_mz <= mz_view_high)
    _visible_max = (
        float(np.nanmax(spectrum_intensity[_visible]))
        if np.any(_visible)
        else float(np.nanmax(spectrum_intensity))
    )
    _pixel_count = len(selected_pixels)
    _plural = "s" if _pixel_count != 1 else ""
    if native_mode:
        _spectrum_title = (
            f"Native profile \u00b7 pixel {selected_pixels[0]}"
            if _pixel_count == 1
            else f"Mean native profile \u00b7 {_pixel_count} pixels \u00b7 {native_bin:g} Da bins"
        )
    else:
        _spectrum_title = (
            f"Mean feature spectrum \u00b7 {_pixel_count} selected pixel{_plural}"
        )
    _ax_spectrum.set(
        xlabel="m/z \u2014 drag to set the aggregated range",
        ylabel="Mean raw intensity",
        title=_spectrum_title,
        xlim=(mz_view_low, mz_view_high),
        ylim=(0, max(_visible_max * 1.08, 1.0)),
    )
    _ax_spectrum.legend(loc="upper right", frameon=False, fontsize=8)
    _fig_spectrum.tight_layout()
    spectrum_selection = mo.ui.matplotlib(_ax_spectrum, debounce=True)
    plt.close(_fig_spectrum)

    _selected_rows = [
        {
            "pixel": int(i),
            "source scan": int(imaging.source_scan_numbers[i]),
            "x": float(imaging.x[i]),
            "y": float(imaging.y[i]),
            f"intensity @ {selected_mz_label}": float(ion_values[i]),
        }
        for i in selected_pixels
    ]
    _spectrum_caption = (
        f"Pixel **{selected_pixels[0]}** at x/y "
        f"**{float(imaging.x[selected_pixels[0]]):g}, {float(imaging.y[selected_pixels[0]]):g}**"
        if _pixel_count == 1
        else f"Raw mean across **{_pixel_count} pixels**"
    )
    if native_mode:
        _selection_detail = (
            f" \u00b7 aggregating `{selected_mz_low:.4f}\u2013{selected_mz_high:.4f}` m/z"
        )
        if _pixel_count == 1:
            _scan = native_profiles[selected_pixels[0]]
            _selection_detail += (
                f" \u00b7 scan **{int(_scan.scan_number)}** \u00b7 RT "
                f"**{float(_scan.retention_time):.4f} min** \u00b7 TIC "
                f"**{float(_scan.tic):,.0f}** \u00b7 **{_scan.mz.size:,}** non-zero points"
            )
    else:
        _feature_count = len(selected_feature_indices)
        _selection_detail = (
            f" \u00b7 **{_feature_count} feature"
            f"{'s' if _feature_count != 1 else ''} selected**"
        )
    spectrum_panel = mo.vstack(
        [
            mo.md(
                "### Selected-pixel "
                + ("native profile" if native_mode else "feature spectrum")
            ),
            mo.md(_spectrum_caption + _selection_detail),
            zoom_controls,
            overview_selection,
            spectrum_selection,
            mo.ui.table(
                _selected_rows, selection=None, pagination=len(_selected_rows) > 12
            ),
        ],
        gap=0.45,
    )
    return overview_selection, spectrum_panel, spectrum_selection


@app.cell(hide_code=True)
def _(
    full_mz_high,
    full_mz_low,
    selected_mz_high,
    selected_mz_low,
    selection_range,
    set_selected_mz_range,
    spectrum_selection,
):
    _spectrum_pick = selection_range(spectrum_selection.value)
    if _spectrum_pick is not None:
        _pick_low, _pick_high = _spectrum_pick
        _pick_low = min(max(_pick_low, full_mz_low), full_mz_high)
        _pick_high = min(max(_pick_high, _pick_low), full_mz_high)
        if (
            abs(_pick_low - selected_mz_low) > 1e-9
            or abs(_pick_high - selected_mz_high) > 1e-9
        ):
            set_selected_mz_range((_pick_low, _pick_high))
    return


@app.cell(hide_code=True)
def _(
    full_mz_high,
    full_mz_low,
    mz_step,
    mz_view_high,
    mz_view_low,
    overview_selection,
    selection_range,
    set_mz_view,
):
    _view_pick = selection_range(overview_selection.value)
    if _view_pick is not None:
        _view_low, _view_high = _view_pick
        _view_low = min(max(_view_low, full_mz_low), full_mz_high - mz_step)
        _view_high = min(max(_view_high, _view_low + mz_step), full_mz_high)
        if (
            abs(_view_low - mz_view_low) > 1e-9
            or abs(_view_high - mz_view_high) > 1e-9
        ):
            set_mz_view((_view_low, _view_high))
    return


@app.cell(hide_code=True)
def _(mo, raw_dataset):
    acquisition_rows = [
        {"field": key, "value": value}
        for key, value in raw_dataset.header.items()
        if value and key not in {"Cal CoVar 1"}
    ]
    metadata_panel = mo.vstack(
        [
            mo.md("### Acquisition metadata"),
            mo.ui.table(
                acquisition_rows, selection=None, pagination=True, page_size=12
            ),
        ],
        gap=0.5,
    )
    return (metadata_panel,)


if __name__ == "__main__":
    app.run()
