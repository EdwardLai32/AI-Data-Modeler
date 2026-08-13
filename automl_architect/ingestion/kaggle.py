"""Kaggle connector: download a dataset (or competition) and read one file.

The ``kaggle`` package is optional and authenticates eagerly on import of its
top-level module, which would turn a missing credential into an import-time
crash. This module therefore imports the API submodule directly and calls
``authenticate()`` itself, so an absent credential surfaces as a
:class:`~automl_architect.core.errors.ConfigurationError` naming the environment
variables to set.

Downloads are cached under ``<workspace>/cache/kaggle/<slug>`` so a re-run or a
replan does not re-fetch hundreds of megabytes.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path

from ..config import get_settings
from ..core.errors import ConfigurationError, IngestionError
from ..core.schemas import SourceKind
from .base import Connector, LoadOutcome, lazy_import, register, resolve_secret
from .files import detect_format, read_any

logger = logging.getLogger(__name__)

#: File extensions the connector will consider reading, best format first.
PREFERRED_EXTENSIONS: tuple[str, ...] = (
    ".parquet",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".xlsx",
    ".xls",
    ".csv.gz",
    ".csv.zip",
)

_SLUG = re.compile(r"^[\w.-]+/[\w.-]+$")


@register(SourceKind.KAGGLE)
class KaggleConnector(Connector):
    """Downloads a Kaggle dataset and reads the most plausible data file.

    ``DataSource.uri`` holds the slug (``"owner/dataset"``) or a full
    ``https://www.kaggle.com/datasets/owner/dataset`` URL. Useful options:

    ``file``
        Name (or suffix) of the file to read. Without it the largest supported
        data file wins, preferring Parquet and CSV.
    ``competition``
        Set true to pull from a competition instead of a dataset.
    ``dest``
        Download directory. Defaults to a cache under the workspace.
    ``force``
        Re-download even when the cache is populated.
    """

    def _slug(self) -> str:
        """Resolve the dataset or competition slug.

        Returns:
            ``"owner/dataset"``, or a bare competition name.

        Raises:
            ConfigurationError: Nothing slug-shaped could be found.
        """
        raw = (
            self.source.uri
            or self.opt_str("dataset", "slug", "competition_name", default="")
            or ""
        ).strip()
        if not raw:
            raise ConfigurationError(
                "A kaggle source needs a dataset slug in DataSource.uri, e.g. "
                "'blastchar/telco-customer-churn'."
            )
        if raw.startswith("http"):
            path = raw.split("kaggle.com/", 1)[-1].strip("/")
            for prefix in ("datasets/", "competitions/", "c/"):
                if path.startswith(prefix):
                    path = path[len(prefix) :]
            segments = [s for s in path.split("/") if s]
            if len(segments) >= 2:
                raw = "/".join(segments[:2])
            else:
                raw = segments[0] if segments else ""
        if not raw:
            raise ConfigurationError(
                f"Could not extract a Kaggle slug from {self.source.uri!r}."
            )
        if not self._is_competition() and not _SLUG.match(raw):
            raise ConfigurationError(
                f"Kaggle dataset slug {raw!r} should look like 'owner/dataset-name'."
            )
        return raw

    def _is_competition(self) -> bool:
        """Whether this source refers to a competition rather than a dataset."""
        if self.opt_bool("competition", "is_competition", default=False):
            return True
        uri = self.source.uri or ""
        return "/competitions/" in uri or "kaggle.com/c/" in uri

    def _apply_credentials(self) -> None:
        """Copy credentials from ``secret_env`` into the vars Kaggle expects.

        The Kaggle client only reads ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or
        ``~/.kaggle/kaggle.json``, so a source that names its own variables needs
        them bridged across. Values are moved between environment variables and
        never logged.
        """
        username = resolve_secret(
            self.source.secret_env, "username", "user", allow_single=False
        )
        key = resolve_secret(self.source.secret_env, "key", "token", "secret")
        if username and not os.environ.get("KAGGLE_USERNAME"):
            os.environ["KAGGLE_USERNAME"] = username
        if key and not os.environ.get("KAGGLE_KEY"):
            os.environ["KAGGLE_KEY"] = key
        if key:
            self.resolved_secrets.append(key)

        has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        if not has_env and not (config_dir / "kaggle.json").exists():
            raise ConfigurationError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY, or "
                f"place kaggle.json in {config_dir}."
            )

    def _destination(self, slug: str) -> Path:
        """Directory to download into, created if needed."""
        explicit = self.opt_str("dest", "download_dir", "path")
        if explicit:
            path = Path(explicit)
        else:
            path = get_settings().workspace / "cache" / "kaggle" / slug.replace("/", "__")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download(self, slug: str, dest: Path) -> None:
        """Fetch the dataset or competition archive into ``dest``.

        Raises:
            IngestionError: The Kaggle API call failed.
        """
        module = lazy_import(
            "kaggle.api.kaggle_api_extended", "Kaggle dataset ingestion", "kaggle"
        )
        api = module.KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            raise ConfigurationError(
                f"Kaggle authentication failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            if self._is_competition():
                api.competition_download_files(slug, path=str(dest), quiet=True)
            else:
                api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=True)
        except Exception as exc:
            raise IngestionError(
                f"Kaggle download of {slug!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._unpack(dest)

    def _unpack(self, dest: Path) -> None:
        """Extract any archives the API left behind (competitions ship zips)."""
        for archive in sorted(dest.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(dest)
                self.note(f"Extracted {archive.name}.")
            except zipfile.BadZipFile:
                self.note(f"Could not extract {archive.name}; it is not a valid zip.")

    def _candidates(self, dest: Path) -> list[Path]:
        """Readable data files under ``dest``, largest first within a format."""
        files = [p for p in dest.rglob("*") if p.is_file()]
        scored: list[tuple[int, int, Path]] = []
        for path in files:
            name = path.name.lower()
            rank = next(
                (
                    index
                    for index, ext in enumerate(PREFERRED_EXTENSIONS)
                    if name.endswith(ext)
                ),
                None,
            )
            if rank is None:
                continue
            scored.append((rank, -path.stat().st_size, path))
        scored.sort()
        return [item[2] for item in scored]

    def _pick(self, dest: Path) -> Path:
        """Choose which downloaded file to read.

        Raises:
            IngestionError: No readable data file was found.
        """
        candidates = self._candidates(dest)
        wanted = self.opt_str("file", "filename", "file_name")
        if wanted:
            exact = [p for p in candidates if p.name == wanted or str(p).endswith(wanted)]
            if exact:
                return exact[0]
            loose = [p for p in candidates if wanted.lower() in p.name.lower()]
            if loose:
                self.note(f"File {wanted!r} not found exactly; used {loose[0].name}.")
                return loose[0]
            self.note(
                f"Requested file {wanted!r} is not in the download; falling back to "
                "the largest data file."
            )
        if not candidates:
            present = sorted(p.name for p in dest.rglob("*") if p.is_file())[:10]
            raise IngestionError(
                f"No readable data file in the Kaggle download. Files present: "
                f"{', '.join(present) or 'none'}."
            )
        if len(candidates) > 1:
            self.note(
                f"Download holds {len(candidates)} data files; read "
                f"{candidates[0].name}. Others: "
                f"{', '.join(p.name for p in candidates[1:6])}."
            )
        return candidates[0]

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Download (or reuse a cached copy of) the dataset and read one file.

        Args:
            max_rows: Row cap passed to the underlying file reader.

        Returns:
            The loaded frame plus notes about the file chosen.
        """
        slug = self._slug()
        dest = self._destination(slug)
        cached = self._candidates(dest)
        if cached and not self.opt_bool("force", "refresh", default=False):
            self.note(f"Reusing the cached Kaggle download in {dest}.")
        else:
            self._apply_credentials()
            self._download(slug, dest)

        path = self._pick(dest)
        fmt = detect_format(path.name, self.opt_str("format"))
        outcome = read_any(
            path,
            fmt=fmt,
            options=self.options,
            max_rows=max_rows,
            notes=self.notes + [f"Read {path.name} from Kaggle dataset {slug}."],
        )
        outcome.detail = f"kaggle:{slug}/{path.name}"
        return outcome


__all__ = ["KaggleConnector", "PREFERRED_EXTENSIONS"]
