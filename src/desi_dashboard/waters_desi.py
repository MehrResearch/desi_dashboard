#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
# ]
# ///
"""Lazy reader for Waters MassLynx DESI-MS ``.raw`` directories.

This module reads the native ``_FUNCnnn.IDX``/``_FUNCnnn.DAT`` files without
MassLynx, a virtual environment, or conversion to mzML.  In particular, it
supports the newer 30-byte IDX records used by the data accompanying this
file; rainbow-api 1.3.0 assumes 22-byte records and cannot read them.

Library example::

    from waters_desi import WatersRaw

    raw = WatersRaw("sample.raw")
    scan = raw.function(1).read_scan(0)  # Python API uses zero-based scans
    print(scan.retention_time, scan.mz, scan.intensity)

    for scan in raw.function(1).iter_scans(nonzero=True):
        ...

Run directly for a command-line interface::

    uv run waters_desi.py info sample.raw
    uv run waters_desi.py scan sample.raw --scan 1 --output scan_1.npz
    uv run waters_desi.py extract sample.raw extracted/

The native format is proprietary and has several variants.  This reader
supports the 6- and 8-byte key/value spectrum encodings and both known 22- and
30-byte index layouts.  The supplied Xevo G2-XS QTof files use 8-byte spectra,
30-byte indexes, and T1 calibration; that combination is tested at startup by
normal file validation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FormatError",
    "ImagingData",
    "Scan",
    "WatersFunction",
    "WatersRaw",
    "open_raw",
]
__version__ = "0.1.0"


class FormatError(RuntimeError):
    """Raised when a MassLynx file is malformed or uses an unknown variant."""


@dataclass(slots=True)
class Scan:
    """One mass spectrum.

    ``index`` is zero-based, while ``scan_number`` is the conventional
    one-based number.  Retention time is in minutes, m/z is in daltons, and
    intensity is in the native detector units.
    """

    function_number: int
    index: int
    retention_time: float
    mz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    stored_tic: float

    @property
    def scan_number(self) -> int:
        return self.index + 1

    @property
    def tic(self) -> float:
        """TIC calculated from the returned points."""
        return float(self.intensity.sum(dtype=np.float64))

    @property
    def base_peak_index(self) -> int | None:
        return int(np.argmax(self.intensity)) if self.intensity.size else None

    @property
    def base_peak_mz(self) -> float | None:
        i = self.base_peak_index
        return float(self.mz[i]) if i is not None else None

    @property
    def base_peak_intensity(self) -> float:
        i = self.base_peak_index
        return float(self.intensity[i]) if i is not None else 0.0


@dataclass(slots=True)
class ImagingData:
    """Processed DESI feature table from the optional ``imaging/*.txt`` file."""

    path: Path
    feature_ids: NDArray[np.int64]
    mz: NDArray[np.float64]
    scan_numbers: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    intensity: NDArray[np.float64]
    region: NDArray[np.int64]
    source_scan_numbers: NDArray[np.int64]
    preamble: tuple[str, ...] = ()

    def nearest_feature(self, mz: float, tolerance: float | None = None) -> int:
        """Return the column index nearest to ``mz``, optionally within tolerance."""
        if not self.mz.size:
            raise LookupError("The imaging table has no m/z features")
        i = int(np.argmin(np.abs(self.mz - mz)))
        delta = abs(float(self.mz[i]) - mz)
        if tolerance is not None and delta > tolerance:
            raise LookupError(
                f"No imaging feature within {tolerance:g} Da of {mz:g}; "
                f"nearest is {self.mz[i]:g} ({delta:g} Da away)"
            )
        return i

    def ion_image(
        self, mz: float, tolerance: float | None = None
    ) -> NDArray[np.float64]:
        """Return the per-pixel intensity vector for the nearest feature."""
        return self.intensity[:, self.nearest_feature(mz, tolerance)]


@dataclass(frozen=True, slots=True)
class _IndexLayout:
    record_size: int
    offset_field: int
    offset_width: int
    bytes_per_pair: int


@dataclass(slots=True)
class _FunctionIndex:
    offsets: NDArray[np.uint64]
    pair_counts: NDArray[np.uint32]
    retention_times: NDArray[np.float32]
    stored_tic: NDArray[np.float32]
    auxiliary: NDArray[np.uint32]
    flags: NDArray[np.uint16]
    layout: _IndexLayout


_HEADER_RE = re.compile(r"^\$\$\s*([^:]+):\s?(.*)$")
_FUNC_DAT_RE = re.compile(r"^_FUNC(\d{3})\.DAT$", re.IGNORECASE)
_CAL_RE = re.compile(r"^Cal Function\s+(\d+)$", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("cp1252", "replace")


def _parse_header(
    path: Path,
) -> tuple[dict[str, str], dict[int, tuple[tuple[float, ...], str]]]:
    fields: dict[str, str] = {}
    calibrations: dict[int, tuple[tuple[float, ...], str]] = {}
    if not path.is_file():
        return fields, calibrations
    for line in _read_text(path).splitlines():
        match = _HEADER_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        fields[key.strip()] = value.strip()
        cal_match = _CAL_RE.match(key.strip())
        if not cal_match:
            continue
        parts = [part.strip() for part in value.split(",")]
        if len(parts) < 2:
            continue
        kind = parts[-1].upper()
        try:
            coefficients = tuple(float(part) for part in parts[:-1] if part)
        except ValueError:
            continue
        calibrations[int(cal_match.group(1))] = (coefficients, kind)
    return fields, calibrations


def _parse_polarities(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*Polarity\s+(.*?)\s*$", line, re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        if value.endswith("+"):
            values.append("+")
        elif value.endswith("-"):
            values.append("-")
        else:
            values.append(value)
    return values


def _candidate_layout(
    raw: bytes,
    dat_size: int,
    record_size: int,
    offset_field: int,
    offset_width: int,
) -> tuple[int, _IndexLayout] | None:
    if not raw or len(raw) % record_size:
        return None
    count = len(raw) // record_size
    if count == 0 or offset_field + offset_width > record_size:
        return None
    offset_dtype = "<u8" if offset_width == 8 else "<u4"
    offsets = np.ndarray(count, offset_dtype, raw, offset_field, (record_size,)).astype(
        np.uint64, copy=False
    )
    pairs = np.ndarray(count, "<u4", raw, 4, (record_size,)) & 0x3FFFFF
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        return None
    if np.any(pairs == 0) or offsets[-1] > dat_size:
        return None
    byte_lengths = np.diff(np.append(offsets, np.uint64(dat_size)))
    if np.any(byte_lengths == 0) or np.any(byte_lengths % pairs != 0):
        return None
    widths = byte_lengths // pairs
    unique, frequencies = np.unique(widths, return_counts=True)
    width = int(unique[np.argmax(frequencies)])
    matches = int(np.count_nonzero(widths == width))
    if width not in (2, 4, 6, 8) or matches != count:
        return None
    # More records matching exactly wins if an unusual file is ambiguous.
    return matches, _IndexLayout(record_size, offset_field, offset_width, width)


def _read_index(idx_path: Path, dat_path: Path) -> _FunctionIndex:
    raw = idx_path.read_bytes()
    dat_size = dat_path.stat().st_size
    # Classic MassLynx uses 22-byte records with an offset at byte 0.  Recent
    # high-resolution imaging data uses 30-byte records with a uint64 offset at
    # byte 22; the first uint32 is then reserved/zero.
    candidates = [
        candidate
        for candidate in (
            _candidate_layout(raw, dat_size, 30, 22, 8),
            _candidate_layout(raw, dat_size, 22, 0, 4),
        )
        if candidate is not None
    ]
    if not candidates:
        raise FormatError(
            f"Cannot identify the IDX layout in {idx_path}; expected validated "
            "22- or 30-byte records and a 2/4/6/8-byte DAT encoding"
        )
    layout = max(candidates, key=lambda item: item[0])[1]
    n = len(raw) // layout.record_size
    stride = (layout.record_size,)
    offset_dtype = "<u8" if layout.offset_width == 8 else "<u4"
    offsets = np.ndarray(n, offset_dtype, raw, layout.offset_field, stride).astype(
        np.uint64
    )
    return _FunctionIndex(
        offsets=offsets,
        pair_counts=(np.ndarray(n, "<u4", raw, 4, stride) & 0x3FFFFF).astype(np.uint32),
        stored_tic=np.ndarray(n, "<f4", raw, 8, stride).copy(),
        retention_times=np.ndarray(n, "<f4", raw, 12, stride).copy(),
        auxiliary=np.ndarray(n, "<u4", raw, 16, stride).copy(),
        flags=np.ndarray(n, "<u2", raw, 20, stride).copy(),
        layout=layout,
    )


def _decode_fraction(
    bits: NDArray[np.uint64], bit_count: NDArray[np.uint64]
) -> NDArray[np.float64]:
    masks = np.left_shift(np.uint64(1), bit_count) - np.uint64(1)
    return (bits & masks).astype(np.float64) / np.exp2(bit_count.astype(np.float64))


def _decode_8(
    raw: bytes, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    words = np.frombuffer(raw, dtype="<u8", count=count)

    key_bits = words >> np.uint64(28)
    integer_bits = key_bits >> np.uint64(31)
    if np.any(integer_bits > 31):
        raise FormatError("Invalid m/z bit allocation in 8-byte spectrum data")
    fractional_bits = np.uint64(31) - integer_bits
    integer_masks = np.left_shift(np.uint64(1), integer_bits) - np.uint64(1)
    mz = ((key_bits >> fractional_bits) & integer_masks).astype(np.float64)
    mz += _decode_fraction(key_bits, fractional_bits)

    value_bits = words & np.uint64(0x0FFFFFFF)
    encoded_integer_bits = value_bits >> np.uint64(22)
    shifts = np.maximum(encoded_integer_bits.astype(np.int16) - 21, 0).astype(np.uint64)
    integer_bits = np.minimum(encoded_integer_bits, np.uint64(21))
    fractional_bits = np.uint64(21) - integer_bits
    integer_masks = np.left_shift(np.uint64(1), integer_bits) - np.uint64(1)
    intensity = (((value_bits >> fractional_bits) & integer_masks) << shifts).astype(
        np.float64
    )
    intensity += _decode_fraction(value_bits, fractional_bits)
    return mz, intensity


def _decode_6(
    raw: bytes, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    packed = np.ndarray(count, dtype="<u4", buffer=raw, offset=2, strides=(6,))
    key_base = packed >> np.uint32(9)
    key_exponent = ((packed & np.uint32(0x1F0)) >> np.uint32(4)).astype(np.int16) - 23
    mz = key_base.astype(np.float64) * np.exp2(key_exponent)
    value_base = np.ndarray(count, dtype="<i2", buffer=raw, offset=0, strides=(6,))
    value_exponent = (packed & np.uint32(0xF)).astype(np.int16)
    intensity = value_base.astype(np.float64) * np.exp2(2 * value_exponent)
    return mz, intensity


def _calibrate(
    mz: NDArray[np.float64], coefficients: Sequence[float], kind: str
) -> NDArray[np.float64]:
    """Apply MassLynx polynomial calibration (T0 or QTof T1)."""
    if not coefficients:
        return mz
    kind = kind.upper()
    if kind == "T0":
        argument = mz
        squared = False
    elif kind == "T1":
        # QTof calibration is a polynomial in flight time (sqrt(mass)), with
        # the result transformed back to mass.  Applying the polynomial to
        # mass directly gives nonsensical high-mass values.
        argument = np.sqrt(mz)
        squared = True
    else:
        raise FormatError(
            f"Unsupported MassLynx calibration type {kind!r}; "
            "call read_scan(..., calibrated=False) to obtain encoded m/z"
        )
    result = np.polynomial.polynomial.polyval(argument, coefficients)
    return np.square(result) if squared else result


class WatersFunction:
    """Lazy access to one native Waters acquisition function."""

    def __init__(
        self,
        raw_path: Path,
        number: int,
        calibration: tuple[tuple[float, ...], str] | None,
        polarity: str | None,
    ) -> None:
        self.raw_path = raw_path
        self.number = number
        self.dat_path = raw_path / f"_FUNC{number:03d}.DAT"
        self.idx_path = raw_path / f"_FUNC{number:03d}.IDX"
        if not self.dat_path.is_file() or not self.idx_path.is_file():
            raise FileNotFoundError(
                f"Function {number} requires {self.dat_path.name} and {self.idx_path.name}"
            )
        self._index = _read_index(self.idx_path, self.dat_path)
        self.calibration = calibration
        self.polarity = polarity

    def __len__(self) -> int:
        return self._index.pair_counts.size

    @property
    def scan_count(self) -> int:
        return len(self)

    @property
    def retention_times(self) -> NDArray[np.float32]:
        """Read-only view of scan retention times in minutes."""
        view = self._index.retention_times.view()
        view.flags.writeable = False
        return view

    @property
    def stored_tic(self) -> NDArray[np.float32]:
        """Read-only view of TIC values stored in the IDX file."""
        view = self._index.stored_tic.view()
        view.flags.writeable = False
        return view

    @property
    def pair_counts(self) -> NDArray[np.uint32]:
        view = self._index.pair_counts.view()
        view.flags.writeable = False
        return view

    @property
    def native_data_offsets(self) -> NDArray[np.uint64]:
        """Read-only byte offsets of scans in the native DAT file."""
        view = self._index.offsets.view()
        view.flags.writeable = False
        return view

    @property
    def index_auxiliary(self) -> NDArray[np.uint32]:
        """Read-only vendor auxiliary field retained from every IDX record."""
        view = self._index.auxiliary.view()
        view.flags.writeable = False
        return view

    @property
    def index_flags(self) -> NDArray[np.uint16]:
        """Read-only vendor flags retained from every IDX record."""
        view = self._index.flags.view()
        view.flags.writeable = False
        return view

    @property
    def bytes_per_pair(self) -> int:
        return self._index.layout.bytes_per_pair

    @property
    def index_record_size(self) -> int:
        return self._index.layout.record_size

    def read_scan(
        self, index: int, *, calibrated: bool = True, nonzero: bool = False
    ) -> Scan:
        """Read one scan without loading the rest of the DAT file.

        Args:
            index: Zero-based scan index; negative Python indexes are accepted.
            calibrated: Apply the calibration from ``_HEADER.TXT`` when present.
            nonzero: Discard zero-intensity profile points to reduce memory.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Scan index {index} is outside 0..{len(self) - 1}")
        count = int(self._index.pair_counts[index])
        width = self.bytes_per_pair
        with self.dat_path.open("rb") as stream:
            stream.seek(int(self._index.offsets[index]))
            raw = stream.read(count * width)
        if len(raw) != count * width:
            raise FormatError(
                f"Short read in {self.dat_path}: expected {count * width} bytes, got {len(raw)}"
            )
        if width == 8:
            mz, intensity = _decode_8(raw, count)
        elif width == 6:
            mz, intensity = _decode_6(raw, count)
        else:
            raise FormatError(
                f"Detected {width}-byte data in {self.dat_path}; this reader only "
                "decodes key/value 6- and 8-byte MS spectra"
            )
        if calibrated and self.calibration is not None:
            mz = _calibrate(mz, *self.calibration)
        if nonzero:
            keep = intensity != 0
            mz, intensity = mz[keep], intensity[keep]
        return Scan(
            function_number=self.number,
            index=index,
            retention_time=float(self._index.retention_times[index]),
            mz=mz,
            intensity=intensity,
            stored_tic=float(self._index.stored_tic[index]),
        )

    def iter_scans(
        self, *, calibrated: bool = True, nonzero: bool = False
    ) -> Iterator[Scan]:
        for index in range(len(self)):
            yield self.read_scan(index, calibrated=calibrated, nonzero=nonzero)

    def ion_chromatogram(
        self, mz: float, tolerance: float = 0.01, *, calibrated: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate an extracted-ion chromatogram using a ±Da tolerance."""
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        abundance = np.empty(len(self), dtype=np.float64)
        for i, scan in enumerate(self.iter_scans(calibrated=calibrated)):
            abundance[i] = scan.intensity[np.abs(scan.mz - mz) <= tolerance].sum()
        return self.retention_times.astype(np.float64), abundance

    def summary(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "scan_count": len(self),
            "retention_time_min": float(self.retention_times[0]),
            "retention_time_max": float(self.retention_times[-1]),
            "pairs_min": int(self.pair_counts.min()),
            "pairs_max": int(self.pair_counts.max()),
            "bytes_per_pair": self.bytes_per_pair,
            "index_record_size": self.index_record_size,
            "polarity": self.polarity,
            "calibration": (
                {"coefficients": list(self.calibration[0]), "type": self.calibration[1]}
                if self.calibration
                else None
            ),
        }


class WatersRaw:
    """A Waters ``.raw`` directory with lazy native scan access."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        self.header_path = self.path / "_HEADER.TXT"
        self.header, calibrations = _parse_header(self.header_path)
        extern_path = self.path / "_extern.inf"
        self.external_text = _read_text(extern_path) if extern_path.is_file() else ""
        inlet_path = self.path / "_INLET.INF"
        self.inlet_text = _read_text(inlet_path) if inlet_path.is_file() else ""
        self.polarities = _parse_polarities(self.external_text)
        numbers = sorted(
            int(match.group(1))
            for item in self.path.iterdir()
            if item.is_file() and (match := _FUNC_DAT_RE.match(item.name))
        )
        if not numbers:
            raise FormatError(f"No _FUNCnnn.DAT files found in {self.path}")
        self.functions: dict[int, WatersFunction] = {}
        for position, number in enumerate(numbers):
            polarity = (
                self.polarities[position] if position < len(self.polarities) else None
            )
            self.functions[number] = WatersFunction(
                self.path, number, calibrations.get(number), polarity
            )

    def __repr__(self) -> str:
        scans = sum(len(function) for function in self.functions.values())
        return f"WatersRaw({str(self.path)!r}, functions={len(self.functions)}, scans={scans})"

    def function(self, number: int = 1) -> WatersFunction:
        try:
            return self.functions[number]
        except KeyError as error:
            raise KeyError(
                f"No function {number}; available: {sorted(self.functions)}"
            ) from error

    @property
    def imaging_paths(self) -> tuple[Path, ...]:
        directory = self.path / "imaging"
        return tuple(sorted(directory.glob("*.txt"))) if directory.is_dir() else ()

    def read_imaging(self, path: str | Path | None = None) -> ImagingData:
        """Read the optional processed DESI feature table.

        If ``path`` is omitted, the sole ``imaging/*.txt`` file is selected.
        """
        if path is None:
            paths = self.imaging_paths
            if not paths:
                raise FileNotFoundError(f"No imaging/*.txt table in {self.path}")
            if len(paths) > 1:
                raise ValueError("Multiple imaging tables found; pass path explicitly")
            table_path = paths[0]
        else:
            table_path = Path(path)
        lines = _read_text(table_path).splitlines()
        if len(lines) < 4:
            raise FormatError(
                f"Imaging table {table_path} has fewer than four header lines"
            )

        # Locate the adjacent feature-ID and m/z header rows.  Their first
        # three fields are blank; data rows begin with scan, x, and y.
        header_index = -1
        feature_ids: list[int] = []
        for i in range(len(lines) - 1):
            fields = lines[i].split("\t")
            next_fields = lines[i + 1].split("\t")
            if len(fields) < 4 or fields[:3] != ["", "", ""]:
                continue
            try:
                ids = [int(value) for value in fields[3:] if value != ""]
                masses = [float(value) for value in next_fields[3:] if value != ""]
            except ValueError:
                continue
            if ids and len(ids) == len(masses):
                header_index, feature_ids = i, ids
                break
        if header_index < 0:
            raise FormatError(f"Cannot locate feature headers in {table_path}")
        masses = [float(value) for value in lines[header_index + 1].split("\t")[3:]]
        n_features = len(feature_ids)

        scans: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        values: list[list[float]] = []
        regions: list[int] = []
        source_scans: list[int] = []
        for line_number, line in enumerate(lines[header_index + 2 :], header_index + 3):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != n_features + 5:
                raise FormatError(
                    f"{table_path}:{line_number}: expected {n_features + 5} fields, got {len(fields)}"
                )
            try:
                scans.append(int(fields[0]))
                xs.append(float(fields[1]))
                ys.append(float(fields[2]))
                values.append([float(value) for value in fields[3 : 3 + n_features]])
                regions.append(int(fields[-2]))
                source_scans.append(int(fields[-1]))
            except ValueError as error:
                raise FormatError(
                    f"Invalid number in {table_path}:{line_number}"
                ) from error
        return ImagingData(
            path=table_path,
            feature_ids=np.asarray(feature_ids, dtype=np.int64),
            mz=np.asarray(masses, dtype=np.float64),
            scan_numbers=np.asarray(scans, dtype=np.int64),
            x=np.asarray(xs, dtype=np.float64),
            y=np.asarray(ys, dtype=np.float64),
            intensity=np.asarray(values, dtype=np.float64),
            region=np.asarray(regions, dtype=np.int64),
            source_scan_numbers=np.asarray(source_scans, dtype=np.int64),
            preamble=tuple(lines[:header_index]),
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable acquisition and file metadata."""
        source_files = {
            str(item.relative_to(self.path)): item.stat().st_size
            for item in sorted(self.path.rglob("*"))
            if item.is_file()
        }
        return {
            "path": str(self.path.resolve()),
            "header": self.header,
            "functions": [
                self.functions[number].summary() for number in sorted(self.functions)
            ],
            "imaging_tables": [str(path) for path in self.imaging_paths],
            "source_file_sizes": source_files,
            "external_parameters_text": self.external_text,
            "inlet_parameters_text": self.inlet_text,
        }

    def extract(
        self,
        output: str | Path,
        *,
        calibrated: bool = True,
        nonzero: bool = False,
    ) -> Path:
        """Extract metadata, scans, and imaging data to a portable directory.

        Spectra are written as concatenated ``.npy`` arrays plus offsets, so
        scans may have different lengths and extraction does not hold all scans
        in RAM.  Scan *i* occupies ``mz[offsets[i]:offsets[i+1]]`` and the same
        intensity slice.
        """
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata()
        metadata["extraction"] = {"calibrated": calibrated, "nonzero": nonzero}
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        for number, function in self.functions.items():
            stem = f"function_{number:03d}"
            if nonzero:
                lengths = np.empty(len(function), dtype=np.uint64)
                for i, scan in enumerate(
                    function.iter_scans(calibrated=False, nonzero=True)
                ):
                    lengths[i] = scan.mz.size
            else:
                lengths = function.pair_counts.astype(np.uint64)
            offsets = np.empty(len(function) + 1, dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(lengths, out=offsets[1:])
            np.save(destination / f"{stem}_offsets.npy", offsets)
            np.save(
                destination / f"{stem}_retention_time.npy", function.retention_times
            )
            np.save(destination / f"{stem}_stored_tic.npy", function.stored_tic)
            np.save(destination / f"{stem}_pair_count.npy", function.pair_counts)
            np.save(
                destination / f"{stem}_index_auxiliary.npy", function.index_auxiliary
            )
            np.save(destination / f"{stem}_index_flags.npy", function.index_flags)
            mz_out = np.lib.format.open_memmap(
                destination / f"{stem}_mz.npy",
                mode="w+",
                dtype=np.float64,
                shape=(int(offsets[-1]),),
            )
            intensity_out = np.lib.format.open_memmap(
                destination / f"{stem}_intensity.npy",
                mode="w+",
                dtype=np.float64,
                shape=(int(offsets[-1]),),
            )
            for i, scan in enumerate(
                function.iter_scans(calibrated=calibrated, nonzero=nonzero)
            ):
                start, stop = int(offsets[i]), int(offsets[i + 1])
                mz_out[start:stop] = scan.mz
                intensity_out[start:stop] = scan.intensity
            mz_out.flush()
            intensity_out.flush()
            del mz_out, intensity_out

        if self.imaging_paths:
            imaging = self.read_imaging()
            np.savez_compressed(
                destination / "imaging.npz",
                feature_ids=imaging.feature_ids,
                mz=imaging.mz,
                scan_numbers=imaging.scan_numbers,
                x=imaging.x,
                y=imaging.y,
                intensity=imaging.intensity,
                region=imaging.region,
                source_scan_numbers=imaging.source_scan_numbers,
            )
        return destination


def open_raw(path: str | Path) -> WatersRaw:
    """Convenience alias returning :class:`WatersRaw`."""
    return WatersRaw(path)


def _cmd_info(args: argparse.Namespace) -> None:
    raw = WatersRaw(args.raw)
    metadata = raw.metadata()
    # The complete external parameter record is useful through the API and in
    # extracted metadata, but too noisy for a terminal summary.
    metadata["external_parameters_text"] = (
        f"<{len(raw.external_text.encode('utf-8'))} UTF-8 bytes; available via API/extract>"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _cmd_scan(args: argparse.Namespace) -> None:
    raw = WatersRaw(args.raw)
    scan = raw.function(args.function).read_scan(
        args.scan - 1, calibrated=not args.uncalibrated, nonzero=args.nonzero
    )
    if args.output:
        np.savez_compressed(
            args.output,
            mz=scan.mz,
            intensity=scan.intensity,
            scan_number=scan.scan_number,
            function_number=scan.function_number,
            retention_time=scan.retention_time,
            stored_tic=scan.stored_tic,
        )
        print(args.output)
    else:
        print(
            json.dumps(
                {
                    "function": scan.function_number,
                    "scan": scan.scan_number,
                    "retention_time": scan.retention_time,
                    "points": int(scan.mz.size),
                    "stored_tic": scan.stored_tic,
                    "calculated_tic": scan.tic,
                    "base_peak_mz": scan.base_peak_mz,
                    "base_peak_intensity": scan.base_peak_intensity,
                },
                indent=2,
            )
        )


def _cmd_extract(args: argparse.Namespace) -> None:
    result = WatersRaw(args.raw).extract(
        args.output, calibrated=not args.uncalibrated, nonzero=args.nonzero
    )
    print(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show acquisition and function metadata")
    info.add_argument("raw", type=Path)
    info.set_defaults(handler=_cmd_info)

    scan = commands.add_parser("scan", help="read one one-based scan")
    scan.add_argument("raw", type=Path)
    scan.add_argument("--function", type=int, default=1)
    scan.add_argument("--scan", type=int, required=True, help="one-based scan number")
    scan.add_argument("--output", type=Path, help="write a compressed NPZ spectrum")
    scan.add_argument(
        "--nonzero", action="store_true", help="discard zero-intensity profile points"
    )
    scan.add_argument(
        "--uncalibrated",
        action="store_true",
        help="return encoded rather than calibrated m/z",
    )
    scan.set_defaults(handler=_cmd_scan)

    extract = commands.add_parser("extract", help="extract all spectra and metadata")
    extract.add_argument("raw", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument(
        "--nonzero", action="store_true", help="discard zero-intensity profile points"
    )
    extract.add_argument(
        "--uncalibrated", action="store_true", help="do not apply m/z calibration"
    )
    extract.set_defaults(handler=_cmd_extract)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
