# /// script
# dependencies = [
#     "marimo>=0.23.14",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt
    from pathlib import Path

    from waters_desi import WatersRaw

    get_selected_pixels, set_selected_pixels = mo.state((0,))
    get_selected_features, set_selected_features = mo.state(None)
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

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

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
        get_selected_features,
        get_selected_pixels,
        mo,
        np,
        plt,
        set_mz_view,
        set_selected_features,
        set_selected_pixels,
    )


@app.cell(hide_code=True)
def _(Path, mo):
    raw_candidates = sorted(Path.cwd().glob("*.raw"))
    if not raw_candidates:
        raise FileNotFoundError(f"No .raw directories found in {Path.cwd()}")

    raw_browser = mo.ui.file_browser(
        initial_path=Path.cwd(),
        selection_mode="directory",
        multiple=False,
        restrict_navigation=True,
        filter=lambda path: path.is_dir() and path.suffix.lower() == ".raw",
        label="Waters .raw dataset",
    )
    return raw_browser, raw_candidates


@app.cell(hide_code=True)
def _(Path, WatersRaw, np, raw_browser, raw_candidates):
    if raw_browser.value:
        selected_raw_path = Path(raw_browser.value[0].path)
    else:
        selected_raw_path = raw_candidates[0]

    if selected_raw_path.suffix.lower() != ".raw":
        raise ValueError("Select a Waters .raw directory")

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
    raw_browser,
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
            raw_browser,
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
def _(controls, map_panel, metadata_panel, mo, native_panel, spectrum_panel):
    imaging_tab = mo.hstack(
        [controls, map_panel, spectrum_panel],
        widths=[0.8, 1.45, 1.55],
        align="start",
        gap=1.0,
    )
    mo.ui.tabs(
        {
            "Imaging": imaging_tab,
            "Native scan": native_panel,
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
        label="Show selected-feature markers",
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
    show_native_scan = mo.ui.checkbox(
        value=False, label="Load native profile for selected pixel"
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
        show_range_markers,
    )


@app.cell(hide_code=True)
def _(
    default_feature_index,
    get_selected_features,
    get_selected_pixels,
    imaging,
    np,
    pixel_slider,
):
    _valid_selected = tuple(
        int(i)
        for i in get_selected_pixels()
        if 0 <= int(i) < imaging.intensity.shape[0]
    )
    selected_pixels = _valid_selected or (int(pixel_slider.value),)
    _state_features = get_selected_features()
    _valid_features = (
        tuple(
            sorted(
                {int(i) for i in _state_features if 0 <= int(i) < imaging.mz.size},
                key=lambda i: float(imaging.mz[i]),
            )
        )
        if _state_features is not None
        else ()
    )
    selected_feature_indices = _valid_features or (default_feature_index,)
    selected_mz_values = imaging.mz[list(selected_feature_indices)]
    selected_mz_low = float(np.min(selected_mz_values))
    selected_mz_high = float(np.max(selected_mz_values))
    selected_mz_label = (
        f"{selected_mz_low:.4f}"
        if len(selected_feature_indices) == 1
        else f"{selected_mz_low:.4f}–{selected_mz_high:.4f}"
    )
    ion_values = imaging.intensity[:, list(selected_feature_indices)].sum(axis=1)
    selected_feature_ids = tuple(
        int(imaging.feature_ids[i]) for i in selected_feature_indices
    )
    return (
        ion_values,
        selected_feature_ids,
        selected_feature_indices,
        selected_mz_high,
        selected_mz_label,
        selected_mz_low,
        selected_mz_values,
        selected_pixels,
    )


@app.cell(hide_code=True)
def _(get_mz_view, imaging, np, selected_mz_high, selected_mz_low):
    full_mz_low = float(np.floor(np.nanmin(imaging.mz)))
    full_mz_high = float(np.ceil(np.nanmax(imaging.mz)))
    mz_step = 0.1
    _stored_mz_view = get_mz_view()
    if _stored_mz_view is None:
        mz_view_low, mz_view_high = full_mz_low, full_mz_high
    else:
        _stored_low, _stored_high = sorted(float(value) for value in _stored_mz_view)
        mz_view_low = max(full_mz_low, min(full_mz_high - mz_step, _stored_low))
        mz_view_high = min(full_mz_high, max(mz_view_low + mz_step, _stored_high))

    _selection_span = max(selected_mz_high - selected_mz_low, 1.0)
    _selection_padding = max(1.0, 0.1 * _selection_span)
    selection_zoom_view = (
        max(full_mz_low, selected_mz_low - _selection_padding),
        min(full_mz_high, selected_mz_high + _selection_padding),
    )
    return (
        full_mz_high,
        full_mz_low,
        mz_step,
        mz_view_high,
        mz_view_low,
        selection_zoom_view,
    )


@app.cell(hide_code=True)
def _(
    full_mz_high,
    full_mz_low,
    mo,
    mz_step,
    mz_view_high,
    mz_view_low,
    selection_zoom_view,
    set_mz_view,
):
    mz_view_slider = mo.ui.range_slider(
        start=full_mz_low,
        stop=full_mz_high,
        step=mz_step,
        value=(mz_view_low, mz_view_high),
        debounce=True,
        show_value=True,
        label="Visible m/z range",
        on_change=lambda value: set_mz_view(tuple(float(v) for v in value)),
        full_width=True,
    )
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
    zoom_selection_button = mo.ui.button(
        label="Zoom to selected range",
        on_click=lambda _: set_mz_view(selection_zoom_view),
        full_width=True,
    )
    zoom_controls = mo.vstack(
        [
            mz_view_slider,
            mo.hstack([mz_low_input, mz_high_input], widths=[1, 1], gap=0.5),
            mo.hstack(
                [reset_zoom_button, zoom_selection_button], widths=[1, 1], gap=0.5
            ),
        ],
        gap=0.4,
    )
    return (zoom_controls,)


@app.cell(hide_code=True)
def _(
    cmap_selector,
    coordinate_grid,
    imaging,
    interpolation_selector,
    ion_values,
    mo,
    np,
    plt,
    selected_feature_ids,
    selected_feature_indices,
    selected_mz_label,
    selected_pixels,
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
        title=f"Ion image · {selected_mz_label} m/z",
    )
    _ax_map.legend(loc="upper right", frameon=False, fontsize=8)
    _map_colorbar = _fig_map.colorbar(_map_artist, ax=_ax_map, fraction=0.046, pad=0.04)
    _map_colorbar.set_label(
        "Raw intensity"
        if len(selected_feature_indices) == 1
        else "Summed raw intensity"
    )
    _fig_map.tight_layout()
    map_selection = mo.ui.matplotlib(_ax_map, debounce=True)
    plt.close(_fig_map)
    _feature_summary = (
        f"Feature **{selected_feature_ids[0]}**"
        if len(selected_feature_indices) == 1
        else f"**{len(selected_feature_indices)} features** in the selected range"
    )
    map_panel = mo.vstack(
        [
            mo.md(f"### Ion image · `{selected_mz_label}` m/z"),
            map_selection,
            mo.md(
                f"{_feature_summary} · raw intensity range "
                f"**{float(np.nanmin(ion_values)):,.0f}–{float(np.nanmax(ion_values)):,.0f}**"
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
    full_mz_high,
    full_mz_low,
    imaging,
    ion_values,
    mo,
    mz_view_high,
    mz_view_low,
    np,
    plt,
    selected_feature_indices,
    selected_mz_high,
    selected_mz_label,
    selected_mz_low,
    selected_mz_values,
    selected_pixels,
    show_range_markers,
    zoom_controls,
):
    selected_feature_spectrum = imaging.intensity[list(selected_pixels)].mean(axis=0)
    _mz_order = np.argsort(imaging.mz)
    _sorted_mz = imaging.mz[_mz_order]
    _sorted_intensity = selected_feature_spectrum[_mz_order]

    _fig_overview, _ax_overview = plt.subplots(figsize=(8.2, 1.35))
    _ax_overview.plot(_sorted_mz, _sorted_intensity, color="#52606d", linewidth=0.55)
    _ax_overview.axvspan(mz_view_low, mz_view_high, color="#2563eb", alpha=0.16)
    _ax_overview.set(
        xlim=(full_mz_low, full_mz_high),
        ylabel="",
        title="Full m/z overview · blue region is the visible window",
    )
    _ax_overview.set_yticks([])
    _fig_overview.tight_layout()
    plt.close(_fig_overview)

    _fig_spectrum, _ax_spectrum = plt.subplots(figsize=(8.2, 4.1))
    _ax_spectrum.plot(_sorted_mz, _sorted_intensity, color="#18212b", linewidth=0.8)
    if len(selected_feature_indices) == 1:
        _ax_spectrum.axvline(
            selected_mz_low,
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
            label=f"{len(selected_feature_indices)} selected features",
        )
    if show_range_markers.value:
        _ax_spectrum.scatter(
            selected_mz_values,
            selected_feature_spectrum[list(selected_feature_indices)],
            color="crimson",
            s=24,
            zorder=3,
        )
    _visible_features = (imaging.mz >= mz_view_low) & (imaging.mz <= mz_view_high)
    _visible_max = (
        float(np.nanmax(selected_feature_spectrum[_visible_features]))
        if np.any(_visible_features)
        else float(np.nanmax(selected_feature_spectrum))
    )
    _ax_spectrum.set(
        xlabel="m/z — drag to select one or more features",
        ylabel="Mean raw intensity",
        title=f"Mean feature spectrum · {len(selected_pixels)} selected pixel{'s' if len(selected_pixels) != 1 else ''}",
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
        if len(selected_pixels) == 1
        else f"Raw mean across **{len(selected_pixels)} pixels**"
    )
    spectrum_panel = mo.vstack(
        [
            mo.md("### Selected-pixel feature spectrum"),
            mo.md(
                _spectrum_caption
                + f" · **{len(selected_feature_indices)} feature{'s' if len(selected_feature_indices) != 1 else ''} selected**"
            ),
            zoom_controls,
            _fig_overview,
            spectrum_selection,
            mo.ui.table(
                _selected_rows, selection=None, pagination=len(_selected_rows) > 12
            ),
        ],
        gap=0.45,
    )
    return spectrum_panel, spectrum_selection


@app.cell(hide_code=True)
def _(
    imaging,
    np,
    selected_feature_indices,
    set_selected_features,
    spectrum_selection,
):
    if spectrum_selection.value:
        _spectrum_pick = spectrum_selection.value
        if hasattr(_spectrum_pick, "x_min"):
            _pick_low, _pick_high = _spectrum_pick.x_min, _spectrum_pick.x_max
        elif hasattr(_spectrum_pick, "vertices"):
            _pick_x = [point[0] for point in _spectrum_pick.vertices]
            _pick_low, _pick_high = min(_pick_x), max(_pick_x)
        else:
            _pick_low = _pick_high = None

        if _pick_low is not None and _pick_high is not None:
            _pick_low, _pick_high = sorted((float(_pick_low), float(_pick_high)))
            _feature_candidates = np.flatnonzero(
                (imaging.mz >= _pick_low) & (imaging.mz <= _pick_high)
            )
            if _feature_candidates.size:
                _new_features = tuple(
                    int(i)
                    for i in _feature_candidates[
                        np.argsort(imaging.mz[_feature_candidates])
                    ]
                )
            else:
                _pick_midpoint = 0.5 * (_pick_low + _pick_high)
                _new_features = (int(np.argmin(np.abs(imaging.mz - _pick_midpoint))),)
            if _new_features != selected_feature_indices:
                set_selected_features(_new_features)
    return


@app.cell(hide_code=True)
def _(
    imaging,
    mo,
    ms_function,
    plt,
    selected_feature_indices,
    selected_mz_high,
    selected_mz_low,
    selected_pixels,
    show_native_scan,
):
    if show_native_scan.value:
        _native_pixel = int(selected_pixels[0])
        _source_scan = int(imaging.source_scan_numbers[_native_pixel])
        native_scan = ms_function.read_scan(_source_scan - 1, nonzero=True)
        _fig_native, _ax_native = plt.subplots(figsize=(10.5, 4.2))
        _ax_native.plot(
            native_scan.mz, native_scan.intensity, color="#18212b", linewidth=0.45
        )
        if len(selected_feature_indices) == 1:
            _ax_native.axvline(
                selected_mz_low, color="crimson", linewidth=1.0, alpha=0.8
            )
        else:
            _ax_native.axvspan(
                selected_mz_low, selected_mz_high, color="crimson", alpha=0.16
            )
        _ax_native.set(
            xlabel="m/z",
            ylabel="Raw intensity",
            title=f"Native profile · source scan {_source_scan} · pixel {_native_pixel}",
            xlim=(float(native_scan.mz.min()), float(native_scan.mz.max())),
        )
        _fig_native.tight_layout()
        plt.close(_fig_native)
        native_panel = mo.vstack(
            [
                mo.md("### Native profile spectrum"),
                mo.md(
                    f"**{native_scan.mz.size:,} non-zero points** · retention time "
                    f"**{native_scan.retention_time:.4f} min** · TIC **{native_scan.tic:,.0f}**"
                ),
                _fig_native,
            ],
            gap=0.45,
        )
    else:
        native_scan = None
        native_panel = mo.callout(
            mo.md(
                "Enable **Load native profile** to lazily read the full MassLynx scan for the first selected pixel."
            ),
            kind="neutral",
        )
    return (native_panel,)


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
