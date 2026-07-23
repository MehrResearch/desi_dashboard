# DESI-MS imaging dashboard

An interactive [marimo](https://marimo.io) dashboard for native Waters
MassLynx DESI-MS imaging `.raw` directories. Browse a dataset, inspect raw ion
images, select spatial pixels, drag across m/z features, zoom into spectral
regions, and lazily inspect individual native profile scans.

No example or scientific data is bundled.

## Run it

Install [uv](https://docs.astral.sh/uv/), change to a directory containing your
Waters `.raw` data directories, then run:

```bash
uvx --from git+https://github.com/MehrResearch/desi_dashboard desi-dashboard
```

The dashboard opens in your browser and initially loads the first `.raw`
directory in the current working directory. Use the file browser at the top to
switch datasets.

Additional arguments are forwarded to `marimo run`, for example:

```bash
uvx --from git+https://github.com/MehrResearch/desi_dashboard \
  desi-dashboard --port 2718 --headless
```

To open the notebook in the marimo editor:

```bash
uvx --from git+https://github.com/MehrResearch/desi_dashboard \
  desi-dashboard-edit
```

## Features

- Reads native Waters `_FUNCnnn.IDX` and `_FUNCnnn.DAT` files lazily.
- Supports the 30-byte indexes and 8-byte profile encoding used by the
  accompanying Xevo G2-XS QTof data, plus classic 22-byte indexes and common
  6-byte spectra.
- Applies MassLynx T0 and QTof T1 m/z calibration.
- Loads processed `imaging/*.txt` feature tables without additional
  post-processing.
- Displays raw ion images for one feature or the summed intensity over a
  dragged m/z range.
- Supports lasso/box spatial selection and mean spectra over selected pixels.
- Provides marker visibility, m/z viewport, numeric zoom controls, full-range
  overview, and zoom-to-selection.
- Lazily loads a native profile spectrum only when requested.
- Exposes acquisition metadata from the Waters directory.

## Development

```bash
git clone https://github.com/MehrResearch/desi_dashboard.git
cd desi_dashboard

uv run desi-dashboard
uv run marimo edit src/desi_dashboard/dashboard.py
```

Run either command from a directory containing the `.raw` datasets you want to
inspect.

## Python reader

The native reader can also be used independently:

```python
from desi_dashboard.waters_desi import WatersRaw

raw = WatersRaw("sample.raw")
scan = raw.function(1).read_scan(0)
print(scan.mz, scan.intensity)
```

## License

MIT
