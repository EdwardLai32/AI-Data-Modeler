"""Object-storage connectors: S3, Azure Blob, and Google Cloud Storage.

Each kind tries three routes in order and reports which one worked:

1. **fsspec** (``s3fs`` / ``adlfs`` / ``gcsfs``) — preferred, because pandas can
   then read the URI directly and Parquet reads stay lazy instead of pulling the
   whole object through Python.
2. **Vendor SDK** (``boto3`` / ``azure-storage-blob`` / ``google-cloud-storage``)
   — downloads the object into memory and reuses the local file readers.
3. **Public HTTPS** — for anonymous objects, needing no cloud dependency at all.
   Tried first when the source is marked ``public``/``anon``, last otherwise.

None of those packages is installed by default, so every import is lazy and a
missing one produces a :class:`MissingDependencyError` naming the exact install
target rather than an ``ImportError`` mid-run.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote, unquote, urlsplit

from ..core.errors import ConfigurationError, IngestionError, MissingDependencyError
from ..core.schemas import SourceKind
from .base import (
    Connector,
    LoadOutcome,
    lazy_import,
    module_available,
    redact_uri,
    register,
    scrub_secrets,
)
from .files import detect_format, read_any

logger = logging.getLogger(__name__)


@dataclass
class _ObjectRef:
    """A parsed object-storage location."""

    container: str
    key: str
    account: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Credential-free description for messages."""
        prefix = f"{self.account}/" if self.account else ""
        return f"{prefix}{self.container}/{self.key}"


class _CloudConnector(Connector):
    """Shared route selection for the three object stores."""

    #: fsspec protocol used when handing the URI to pandas.
    protocol: str = ""
    #: fsspec driver package and its pip target.
    fsspec_driver: tuple[str, str] = ("", "")
    #: Vendor SDK module and its pip target.
    sdk: tuple[str, str] = ("", "")
    feature: str = "cloud ingestion"

    # -- subclass hooks ---------------------------------------------------

    def _parse(self) -> _ObjectRef:
        """Parse ``DataSource.uri`` into an object reference."""
        raise NotImplementedError

    def _storage_options(self, ref: _ObjectRef) -> dict[str, Any]:
        """fsspec ``storage_options`` for this source, credentials included."""
        raise NotImplementedError

    def _download(self, ref: _ObjectRef) -> bytes:
        """Fetch the object with the vendor SDK."""
        raise NotImplementedError

    def _public_url(self, ref: _ObjectRef) -> str | None:
        """HTTPS URL for anonymous access, or ``None`` if not constructible."""
        return None

    def _fsspec_uri(self, ref: _ObjectRef) -> str:
        """URI to hand to fsspec (defaults to what the user supplied)."""
        return self.source.uri

    # -- shared load ------------------------------------------------------

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Read the object through the first route that works.

        Args:
            max_rows: Row cap.

        Returns:
            The loaded frame plus a note naming the route used.

        Raises:
            MissingDependencyError: Every route is unavailable for want of a
                package.
            IngestionError: Routes were available but all failed.
        """
        ref = self._parse()
        fmt = detect_format(ref.key or self.source.uri, self.opt_str("format"))
        anonymous = self.opt_bool("public", "anon", "anonymous", default=False)
        order = (
            ("public", "fsspec", "sdk")
            if anonymous
            else ("fsspec", "sdk", "public")
        )

        routes: dict[str, Callable[[], LoadOutcome]] = {
            "fsspec": lambda: self._load_fsspec(ref, fmt, max_rows),
            "sdk": lambda: self._load_sdk(ref, fmt, max_rows),
            "public": lambda: self._load_public(ref, fmt, max_rows, anonymous),
        }

        missing: dict[str, str] = {}
        failures: dict[str, str] = {}
        for name in order:
            try:
                return routes[name]()
            except MissingDependencyError as exc:
                missing[name] = str(exc)
            except (ConfigurationError, IngestionError) as exc:
                failures[name] = self._scrub(str(exc))
            except Exception as exc:  # pragma: no cover - SDK-specific errors
                failures[name] = f"{type(exc).__name__}: {self._scrub(str(exc))}"

        # When neither authenticated route was even importable, the actionable
        # problem is the missing package — not whatever the anonymous HTTPS
        # fallback happened to return.
        if {"fsspec", "sdk"} <= missing.keys():
            driver, extra = self.fsspec_driver
            feature = f"{self.feature} for {redact_uri(self.source.uri)}"
            if "public" in failures:
                feature += (
                    f" (the anonymous HTTPS fallback also failed: "
                    f"{failures['public'][:160]})"
                )
            raise MissingDependencyError(
                driver or self.sdk[0], feature, extra or self.sdk[1]
            )
        raise IngestionError(
            f"Could not read {redact_uri(self.source.uri)}. Attempts: "
            + " | ".join(
                f"{name}: {text}" for name, text in {**failures, **missing}.items()
            )
        )

    # -- routes -----------------------------------------------------------

    def _load_fsspec(
        self, ref: _ObjectRef, fmt: str, max_rows: int | None
    ) -> LoadOutcome:
        """Read via fsspec, letting pandas address the URI directly."""
        driver, extra = self.fsspec_driver
        if not module_available("fsspec"):
            raise MissingDependencyError("fsspec", self.feature, "fsspec")
        if driver and not module_available(driver):
            raise MissingDependencyError(driver, self.feature, extra or driver)
        outcome = read_any(
            self._fsspec_uri(ref),
            fmt=fmt,
            options=self.options,
            max_rows=max_rows,
            storage_options=self._storage_options(ref),
            notes=self.notes + [f"Read {ref.label} via fsspec/{driver}."],
        )
        outcome.detail = f"{self.protocol} via fsspec"
        return outcome

    def _load_sdk(self, ref: _ObjectRef, fmt: str, max_rows: int | None) -> LoadOutcome:
        """Read by downloading the object with the vendor SDK."""
        module, extra = self.sdk
        if module and not module_available(module):
            raise MissingDependencyError(module, self.feature, extra or module)
        data = self._download(ref)
        outcome = read_any(
            io.BytesIO(data),
            fmt=fmt,
            options=self.options,
            max_rows=max_rows,
            notes=self.notes
            + [f"Downloaded {ref.label} ({len(data):,} bytes) via {module}."],
        )
        outcome.detail = f"{self.protocol} via {module}"
        return outcome

    def _load_public(
        self, ref: _ObjectRef, fmt: str, max_rows: int | None, anonymous: bool
    ) -> LoadOutcome:
        """Read over plain HTTPS, which only works for public objects."""
        url = self._public_url(ref)
        if not url:
            raise ConfigurationError(
                f"No public HTTPS URL can be derived for {ref.label}."
            )
        note = f"Read {ref.label} over public HTTPS."
        if not anonymous:
            note += (
                " No cloud SDK was usable, so anonymous access was attempted as a "
                "fallback."
            )
        outcome = read_any(
            url,
            fmt=fmt,
            options=self.options,
            max_rows=max_rows,
            notes=self.notes + [note],
        )
        outcome.detail = f"{self.protocol} via public https"
        return outcome

    # -- helpers ----------------------------------------------------------

    def _require_uri(self) -> str:
        """Return the source URI, or raise when it is empty."""
        uri = (self.source.uri or "").strip()
        if not uri:
            raise ConfigurationError(
                f"A {self.source.kind.value} source needs an object URI in DataSource.uri."
            )
        return uri

    def _scrub(self, message: str) -> str:
        """Remove resolved credential values from ``message``."""
        return scrub_secrets(message, self.resolved_secrets)


# ---------------------------------------------------------------------------
# Amazon S3
# ---------------------------------------------------------------------------


@register(SourceKind.S3)
class S3Connector(_CloudConnector):
    """Reads an object from Amazon S3 (or an S3-compatible endpoint)."""

    protocol = "s3"
    fsspec_driver = ("s3fs", "s3fs")
    sdk = ("boto3", "boto3")
    feature = "S3 ingestion"

    def _parse(self) -> _ObjectRef:
        uri = self._require_uri()
        parts = urlsplit(uri)
        if parts.scheme.lower() in {"s3", "s3a", "s3n"}:
            bucket, key = parts.netloc, parts.path.lstrip("/")
        elif parts.scheme.lower() in {"http", "https"}:
            bucket, key = _bucket_from_https(parts.netloc, parts.path)
        else:
            bucket = self.opt_str("bucket", default="") or ""
            key = uri.lstrip("/")
        if not bucket or not key:
            raise ConfigurationError(
                f"Could not read a bucket and key from {redact_uri(uri)}; expected "
                "'s3://bucket/path/to/object'."
            )
        return _ObjectRef(container=bucket, key=unquote(key))

    def _fsspec_uri(self, ref: _ObjectRef) -> str:
        # Always address s3fs by protocol. Handing it the user's literal URI
        # would route an https:// or bare bucket/key form to the http or *local*
        # filesystem instead — the latter can silently read a same-named file
        # from disk — and s3a:///s3n:// are not registered fsspec protocols.
        return f"s3://{ref.container}/{ref.key}"

    def _storage_options(self, ref: _ObjectRef) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.opt_bool("public", "anon", "anonymous", default=False):
            options["anon"] = True
        else:
            key = self.secret("access_key_id", "AWS_ACCESS_KEY_ID", allow_single=False)
            secret = self.secret(
                "secret_access_key", "AWS_SECRET_ACCESS_KEY", allow_single=False
            )
            token = self.secret("session_token", "AWS_SESSION_TOKEN", allow_single=False)
            if key:
                options["key"] = key
            if secret:
                options["secret"] = secret
            if token:
                options["token"] = token
        client_kwargs: dict[str, Any] = {}
        region = self.opt_str("region", "region_name")
        endpoint = self.opt_str("endpoint_url", "endpoint")
        if region:
            client_kwargs["region_name"] = region
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        if client_kwargs:
            options["client_kwargs"] = client_kwargs
        if self.opt_str("profile"):
            options["profile"] = self.opt_str("profile")
        return options

    def _download(self, ref: _ObjectRef) -> bytes:
        boto3 = lazy_import("boto3", self.feature, "boto3")
        client_kwargs: dict[str, Any] = {}
        region = self.opt_str("region", "region_name")
        endpoint = self.opt_str("endpoint_url", "endpoint")
        if region:
            client_kwargs["region_name"] = region
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint

        if self.opt_bool("public", "anon", "anonymous", default=False):
            botocore = lazy_import("botocore.client", self.feature, "botocore")
            unsigned = lazy_import("botocore", self.feature, "botocore")
            client_kwargs["config"] = botocore.Config(
                signature_version=unsigned.UNSIGNED
            )
        else:
            key = self.secret("access_key_id", "AWS_ACCESS_KEY_ID", allow_single=False)
            secret = self.secret(
                "secret_access_key", "AWS_SECRET_ACCESS_KEY", allow_single=False
            )
            token = self.secret("session_token", "AWS_SESSION_TOKEN", allow_single=False)
            if key and secret:
                client_kwargs.update(
                    aws_access_key_id=key,
                    aws_secret_access_key=secret,
                    aws_session_token=token,
                )
        profile = self.opt_str("profile")
        session = (
            boto3.session.Session(profile_name=profile)
            if profile
            else boto3.session.Session()
        )
        client = session.client("s3", **client_kwargs)
        response = client.get_object(Bucket=ref.container, Key=ref.key)
        return response["Body"].read()

    def _public_url(self, ref: _ObjectRef) -> str | None:
        endpoint = self.opt_str("endpoint_url", "endpoint")
        path = quote(ref.key)
        if endpoint:
            return f"{endpoint.rstrip('/')}/{ref.container}/{path}"
        region = self.opt_str("region", "region_name")
        host = (
            f"{ref.container}.s3.amazonaws.com"
            if not region or region == "us-east-1"
            else f"{ref.container}.s3.{region}.amazonaws.com"
        )
        return f"https://{host}/{path}"


def _bucket_from_https(netloc: str, path: str) -> tuple[str, str]:
    """Split an S3 HTTPS URL into ``(bucket, key)`` for either URL style."""
    host = netloc.lower()
    virtual = re.match(r"^([^.]+)\.s3[.-]", host)
    if virtual:
        return virtual.group(1), path.lstrip("/")
    segments = path.lstrip("/").split("/", 1)
    return segments[0], segments[1] if len(segments) > 1 else ""


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------


@register(SourceKind.AZURE_BLOB)
class AzureBlobConnector(_CloudConnector):
    """Reads a blob from Azure Blob Storage / ADLS Gen2.

    Accepts ``az://``, ``abfs(s)://container@account.dfs.core.windows.net/path``,
    ``wasbs://``, and plain ``https://account.blob.core.windows.net/container/path``.
    """

    protocol = "abfs"
    fsspec_driver = ("adlfs", "adlfs")
    sdk = ("azure.storage.blob", "azure-storage-blob")
    feature = "Azure Blob ingestion"

    def _parse(self) -> _ObjectRef:
        uri = self._require_uri()
        parts = urlsplit(uri)
        scheme = parts.scheme.lower()
        account = self.opt_str("account", "account_name", default="") or ""
        netloc = parts.netloc
        container = ""

        if scheme in {"abfs", "abfss", "wasb", "wasbs", "az", "adl"}:
            if "@" in netloc:
                container, host = netloc.split("@", 1)
                account = account or host.split(".", 1)[0]
            else:
                container = netloc
            key = parts.path.lstrip("/")
        elif scheme in {"http", "https"}:
            account = account or netloc.split(".", 1)[0]
            segments = parts.path.lstrip("/").split("/", 1)
            container = segments[0]
            key = segments[1] if len(segments) > 1 else ""
        else:
            container = self.opt_str("container", default="") or ""
            key = uri.lstrip("/")

        container = container or (self.opt_str("container") or "")
        if not container or not key:
            raise ConfigurationError(
                f"Could not read a container and blob path from {redact_uri(uri)}; "
                "expected 'az://container/path' or "
                "'abfss://container@account.dfs.core.windows.net/path'."
            )
        return _ObjectRef(container=container, key=unquote(key), account=account)

    def _credentials(self) -> dict[str, str]:
        """Resolve whichever Azure credential form is available."""
        out: dict[str, str] = {}
        connection = self.secret(
            "connection_string", "AZURE_STORAGE_CONNECTION_STRING", allow_single=False
        )
        if connection and "AccountName" in connection:
            out["connection_string"] = connection
            return out
        account_key = self.secret(
            "account_key", "AZURE_STORAGE_KEY", "access_key", allow_single=False
        )
        sas = self.secret(
            "sas", "signature", "AZURE_STORAGE_SAS_TOKEN", allow_single=False
        )
        if account_key:
            out["account_key"] = account_key
        if sas:
            out["sas_token"] = sas.lstrip("?")
        return out

    def _storage_options(self, ref: _ObjectRef) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if ref.account:
            options["account_name"] = ref.account
        credentials = self._credentials()
        options.update(credentials)
        if not credentials and self.opt_bool(
            "public", "anon", "anonymous", default=False
        ):
            options["anon"] = True
        return options

    def _fsspec_uri(self, ref: _ObjectRef) -> str:
        # adlfs understands abfs://container/path with account_name supplied
        # separately; normalising avoids depending on the user's scheme choice.
        return f"abfs://{ref.container}/{ref.key}"

    def _download(self, ref: _ObjectRef) -> bytes:
        blob_module = lazy_import("azure.storage.blob", self.feature, "azure-storage-blob")
        credentials = self._credentials()
        if "connection_string" in credentials:
            service = blob_module.BlobServiceClient.from_connection_string(
                credentials["connection_string"]
            )
        else:
            if not ref.account:
                raise ConfigurationError(
                    "Azure Blob access needs an 'account' option or an account name in "
                    "the URI."
                )
            credential: Any = credentials.get("account_key") or credentials.get("sas_token")
            if credential is None and not self.opt_bool(
                "public", "anon", "anonymous", default=False
            ):
                identity = lazy_import(
                    "azure.identity", self.feature, "azure-identity"
                )
                credential = identity.DefaultAzureCredential()
            service = blob_module.BlobServiceClient(
                account_url=f"https://{ref.account}.blob.core.windows.net",
                credential=credential,
            )
        client = service.get_blob_client(container=ref.container, blob=ref.key)
        return client.download_blob().readall()

    def _public_url(self, ref: _ObjectRef) -> str | None:
        if not ref.account:
            return None
        sas = self._credentials().get("sas_token")
        url = (
            f"https://{ref.account}.blob.core.windows.net/"
            f"{ref.container}/{quote(ref.key)}"
        )
        return f"{url}?{sas}" if sas else url


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------


@register(SourceKind.GCS)
class GcsConnector(_CloudConnector):
    """Reads an object from Google Cloud Storage."""

    protocol = "gs"
    fsspec_driver = ("gcsfs", "gcsfs")
    sdk = ("google.cloud.storage", "google-cloud-storage")
    feature = "GCS ingestion"

    def _parse(self) -> _ObjectRef:
        uri = self._require_uri()
        parts = urlsplit(uri)
        if parts.scheme.lower() in {"gs", "gcs"}:
            bucket, key = parts.netloc, parts.path.lstrip("/")
        elif parts.scheme.lower() in {"http", "https"}:
            segments = parts.path.lstrip("/").split("/", 1)
            bucket = segments[0]
            key = segments[1] if len(segments) > 1 else ""
        else:
            bucket = self.opt_str("bucket", default="") or ""
            key = uri.lstrip("/")
        if not bucket or not key:
            raise ConfigurationError(
                f"Could not read a bucket and object from {redact_uri(uri)}; expected "
                "'gs://bucket/path/to/object'."
            )
        return _ObjectRef(container=bucket, key=unquote(key))

    def _fsspec_uri(self, ref: _ObjectRef) -> str:
        # See S3Connector._fsspec_uri: address gcsfs by protocol so an https://
        # or bare bucket/key form cannot be routed to the wrong filesystem.
        return f"gs://{ref.container}/{ref.key}"

    def _storage_options(self, ref: _ObjectRef) -> dict[str, Any]:
        if self.opt_bool("public", "anon", "anonymous", default=False):
            return {"token": "anon"}
        options: dict[str, Any] = {}
        credentials = self.secret(
            "credentials", "service_account", "GOOGLE_APPLICATION_CREDENTIALS", "key_file"
        )
        if credentials:
            options["token"] = credentials
        if self.opt_str("project"):
            options["project"] = self.opt_str("project")
        return options

    def _download(self, ref: _ObjectRef) -> bytes:
        storage = lazy_import("google.cloud.storage", self.feature, "google-cloud-storage")
        if self.opt_bool("public", "anon", "anonymous", default=False):
            client = storage.Client.create_anonymous_client()
        else:
            key_file = self.secret(
                "credentials", "service_account", "GOOGLE_APPLICATION_CREDENTIALS"
            )
            project = self.opt_str("project")
            if key_file:
                client = storage.Client.from_service_account_json(
                    key_file, project=project
                )
            else:
                client = storage.Client(project=project) if project else storage.Client()
        blob = client.bucket(ref.container).blob(ref.key)
        return blob.download_as_bytes()

    def _public_url(self, ref: _ObjectRef) -> str | None:
        return f"https://storage.googleapis.com/{ref.container}/{quote(ref.key)}"


__all__ = ["AzureBlobConnector", "GcsConnector", "S3Connector"]
