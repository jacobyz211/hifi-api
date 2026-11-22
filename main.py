#!/usr/bin/env python3
import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Union

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import logging

logger = logging.getLogger(__name__)

load_dotenv()

API_VERSION = "2.10"

# Shared HTTP client is created in app lifespan for connection reuse
_http_client: Optional[httpx.AsyncClient] = None
_http_client_proxy_url: Optional[str] = None
_http_client_lock = asyncio.Lock()

# One lock per credential to avoid global contention during token refreshes
_refresh_locks: Dict[str, asyncio.Lock] = {}

# Loaded credential set from token.json; each entry will be enriched with access cache
_creds: List[dict] = []

# Optional credential reserved for catalog/metadata traffic. It is deliberately
# kept out of the playback pool and is not request-serialized.
_catalog_cred: Optional[dict] = None

# Built later from _creds; named here so the token-file reloader can test it.
_playback_pool = None

# Global semaphore to limit concurrent album track fetches across all requests
_album_tracks_sem = asyncio.Semaphore(20)

# List of proxies loaded from file at startup
_proxies: List[str] = []

# Cache of the last proxy confirmed to be working
_last_known_good_proxy: Optional[str] = None


def _build_http_client(proxy_url: Optional[str] = None) -> httpx.AsyncClient:
    # Pack common settings into a dictionary to keep things DRY
    client_kwargs = {
        "http2": True,
        "headers": _tidal_headers(),
        "timeout": httpx.Timeout(connect=3.0, read=12.0, write=8.0, pool=12.0),
        "limits": httpx.Limits(
            max_keepalive_connections=500,
            max_connections=1000,
            keepalive_expiry=30.0,
        ),
    }

    try:
        # Modern httpx
        return httpx.AsyncClient(proxy=proxy_url, **client_kwargs)
    except TypeError:
        # Legacy httpx
        # If proxy_url is None, proxies=None is perfectly valid.
        # If it's a string, older httpx versions require it to be a dictionary mapping.
        legacy_proxies = {"all://": proxy_url} if proxy_url else None
        return httpx.AsyncClient(proxies=legacy_proxies, **client_kwargs)


def _build_proxy_test_client(proxy_url: str) -> httpx.AsyncClient:
    try:
        # Modern httpx
        return httpx.AsyncClient(proxy=proxy_url, timeout=5.0)
    except TypeError:
        # Legacy httpx
        return httpx.AsyncClient(proxies={"all://": proxy_url}, timeout=5.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _http_client_proxy_url
    if DEV_MODE:
        logger.warning("DEV_MODE is enabled — upstream responses will be logged at DEBUG level")
    if _http_client is None:
        proxy_url = None
        if USE_PROXIES:
            proxy_url = await get_working_proxy()
            if not proxy_url and not FALLBACK_TO_DIRECT_CONNECTION:
                logger.error("Could not find a working proxy and FALLBACK_TO_DIRECT_CONNECTION is False. Shutting down.")
                raise RuntimeError("No working proxies available")
            elif not proxy_url and FALLBACK_TO_DIRECT_CONNECTION:
                logger.warning("Could not find a working proxy, falling back to direct connection. HOST IP MAY BE EXPOSED!")
        _http_client = _build_http_client(proxy_url)
        _http_client_proxy_url = proxy_url
    try:
        start_playback_watchdog()
        yield
    finally:
        await stop_playback_watchdog()
        if _http_client:
            await _http_client.aclose()
            _http_client = None
            _http_client_proxy_url = None

app = FastAPI(
    title="HiFi-RestAPI",
    version=API_VERSION,
    description="Tidal Music Proxy",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Config (defaults act as fallback if token file missing)
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
REFRESH_TOKEN: Optional[str] = os.getenv("REFRESH_TOKEN")
USER_ID = os.getenv("USER_ID")
TOKEN_FILE = os.getenv("TOKEN_FILE", "token.json")
COUNTRY_CODE = os.getenv("COUNTRY_CODE", "US")

# Catalog traffic can use either a static bearer token or a refreshable account.
# A token.json entry with `"role": "catalog"` takes precedence over these.
CATALOG_TOKEN = os.getenv("CATALOG_TOKEN") or os.getenv("CATALOG_ACCESS_TOKEN")
CATALOG_CLIENT_ID = os.getenv("CATALOG_CLIENT_ID") or CLIENT_ID
CATALOG_CLIENT_SECRET = os.getenv("CATALOG_CLIENT_SECRET") or CLIENT_SECRET
CATALOG_REFRESH_TOKEN = os.getenv("CATALOG_REFRESH_TOKEN")
CATALOG_USER_ID = os.getenv("CATALOG_USER_ID")

USE_PROXIES = os.getenv("USE_PROXIES", "False").lower() in ("true", "1", "yes")
ROTATE_PROXIES_ON_REFRESH = os.getenv("ROTATE_PROXIES_ON_REFRESH", "False").lower() in ("true", "1", "yes")
PROXIES_FILE = os.getenv("PROXIES_FILE", "proxies.txt")
FALLBACK_TO_DIRECT_CONNECTION = os.getenv("FALLBACK_TO_DIRECT_CONNECTION", "False").lower() in ("true", "1", "yes")
# Maximum number of proxy candidates to test per get_working_proxy() call
MAX_PROXY_CANDIDATES = 10
# Maximum number of concurrent proxy tests inside get_working_proxy()
_PROXY_TEST_CONCURRENCY = 5
_max_retries_raw = os.getenv("MAX_RETRIES", "2")
USER_AGENT = os.getenv("USER_AGENT", "okhttp/5.3.2")


def _tidal_headers(extra: dict | None = None) -> dict:
    h = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Encoding": "gzip",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Platform": "android",
        "X-Tidal-Platform": "android",
    }
    if extra:
        h.update(extra)
    return h


_TIDAL_DEFAULT_HEADERS = _tidal_headers()

DEV_MODE = os.getenv("DEV_MODE", "False").lower() in ("true", "1", "yes")

_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BASE_DELAY = 1.0
_RATE_LIMIT_MAX_DELAY = 10.0

# Playback queue hardening. A slot that is never returned wedges the whole
# instance: once every account is held, every playback request is answered
# with 202 and nothing ever completes, so clients report "no playable
# stream". These limits let the instance heal itself instead of waiting for
# a restart.
try:
    PLAYBACK_LEASE_TIMEOUT = float(os.getenv("PLAYBACK_LEASE_TIMEOUT", "90"))
except ValueError:
    PLAYBACK_LEASE_TIMEOUT = 90.0
try:
    PLAYBACK_JOB_TIMEOUT = float(os.getenv("PLAYBACK_JOB_TIMEOUT", "180"))
except ValueError:
    PLAYBACK_JOB_TIMEOUT = 180.0
try:
    PLAYBACK_MAX_PENDING_JOBS = int(os.getenv("PLAYBACK_MAX_PENDING_JOBS", "200"))
except ValueError:
    PLAYBACK_MAX_PENDING_JOBS = 200
# How long a request waits for a free account before it is answered with 202.
# Answering immediately made a busy instance look broken to clients that give
# up on a queued reply; a short wait turns most of those into a real stream.
# Kept under the 8 s a client waits for a manifest, so a held request can
# still answer with the stream it was holding for.
try:
    PLAYBACK_QUEUE_HOLD = float(os.getenv("PLAYBACK_QUEUE_HOLD", "5"))
except ValueError:
    PLAYBACK_QUEUE_HOLD = 5.0
_PLAYBACK_WATCHDOG_INTERVAL = 5.0

def _log_response(method: str, url: str, resp: httpx.Response):
    if not DEV_MODE:
        return
    logger.info(
        "[DEV] %s %s → %s\n  headers: %s\n  body: %s",
        method,
        url,
        resp.status_code,
        dict(resp.headers),
        resp.text[:2000],
    )

try:
    MAX_RETRIES = int(_max_retries_raw)
except ValueError:
    MAX_RETRIES = 2
if MAX_RETRIES < 1:
    MAX_RETRIES = 1
def load_proxies():
    """Load proxies from file into the global _proxies list."""
    global _proxies
    if not os.path.exists(PROXIES_FILE):
        logger.warning(f"Proxies file {PROXIES_FILE} not found.")
        _proxies = []
        return
    with open(PROXIES_FILE, "r") as f:
        _proxies = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(_proxies)} proxies.")


async def test_proxy(proxy_url: str) -> bool:
    try:
        async with _build_proxy_test_client(proxy_url) as client:
            resp = await client.get("http://example.com")
            return resp.status_code == 200
    except Exception:
        return False


async def get_working_proxy(avoid_proxy: Optional[str] = None) -> Optional[str]:
    global _last_known_good_proxy

    if not _proxies:
        return None

    # Try the cached proxy first (unless it is the one we want to avoid)
    if _last_known_good_proxy and _last_known_good_proxy != avoid_proxy:
        if await test_proxy(_last_known_good_proxy):
            return _last_known_good_proxy

    shuffled_proxies = _proxies[:]
    random.shuffle(shuffled_proxies)

    if avoid_proxy:
        candidate_proxies = [p for p in shuffled_proxies if p != avoid_proxy]
        if not candidate_proxies:
            candidate_proxies = shuffled_proxies
    else:
        candidate_proxies = shuffled_proxies

    # Exclude the already-tested cached proxy and cap the candidate list
    if _last_known_good_proxy:
        candidate_proxies = [p for p in candidate_proxies if p != _last_known_good_proxy]
    candidate_proxies = candidate_proxies[:MAX_PROXY_CANDIDATES]

    # Test candidates concurrently, returning the first one that succeeds
    sem = asyncio.Semaphore(_PROXY_TEST_CONCURRENCY)
    found_event = asyncio.Event()
    selected_proxy: List[Optional[str]] = [None]

    async def probe(proxy: str) -> None:
        if found_event.is_set():
            return
        async with sem:
            if found_event.is_set():
                return
            if await test_proxy(proxy):
                if not found_event.is_set():
                    selected_proxy[0] = proxy
                    found_event.set()

    await asyncio.gather(*[probe(p) for p in candidate_proxies], return_exceptions=True)

    if selected_proxy[0]:
        _last_known_good_proxy = selected_proxy[0]
    return selected_proxy[0]

async def _delayed_close(client: httpx.AsyncClient):
    await asyncio.sleep(15)
    await client.aclose()

async def update_global_client(force_new_proxy: bool = False):
    global _http_client, _http_client_proxy_url
    async with _http_client_lock:
        proxy_to_avoid = None
        if force_new_proxy and _http_client_proxy_url:
            proxy_to_avoid = _http_client_proxy_url

        proxy_url = None
        if USE_PROXIES:
            proxy_url = await get_working_proxy(avoid_proxy=proxy_to_avoid)
            if not proxy_url:
                if FALLBACK_TO_DIRECT_CONNECTION:
                    logger.warning("Could not find a working proxy, falling back to direct connection. HOST IP MAY BE EXPOSED!")
                else:
                    logger.error("Could not find a working proxy and FALLBACK_TO_DIRECT_CONNECTION is False.")
                    raise HTTPException(status_code=503, detail="Service Unavailable")

        # Only create a new client if the proxy is actually different
        if _http_client and _http_client_proxy_url == proxy_url:
            return

        new_client = _build_http_client(proxy_url)
        old_client = _http_client
        _http_client = new_client
        _http_client_proxy_url = proxy_url

        if old_client is not None:
            asyncio.create_task(_delayed_close(old_client))


if USE_PROXIES:
    load_proxies()

_ENV_CLIENT_ID = CLIENT_ID
_ENV_CLIENT_SECRET = CLIENT_SECRET
_ENV_REFRESH_TOKEN = REFRESH_TOKEN
_ENV_USER_ID = USER_ID
_TOKEN_FILE_MTIME = 0.0


def _credential_from_entry(entry: dict) -> dict:
    return {
        "client_id": entry.get("client_ID") or _ENV_CLIENT_ID,
        "client_secret": entry.get("client_secret") or _ENV_CLIENT_SECRET,
        "refresh_token": entry.get("refresh_token") or _ENV_REFRESH_TOKEN,
        "user_id": entry.get("userID") or _ENV_USER_ID,
        # Access tokens in file have unknown expiry; force refresh on first use
        "access_token": None,
        "expires_at": 0,
        # When the last refresh of this account was refused. The root
        # endpoint uses it to report the instance as unusable.
        "_failed_at": 0,
        # When this account last answered with a PREVIEW clip. Free-tier
        # accounts stay in token.json but are skipped for playback once seen.
        "_preview_at": 0,
    }


def _carry_live_state(old: List[dict], new: List[dict]) -> None:
    """Keep a still-valid access token when the same account stays in the pool."""
    by_refresh = {c.get("refresh_token"): c for c in old}
    for cred in new:
        prev = by_refresh.get(cred.get("refresh_token"))
        if prev is None:
            continue
        cred["access_token"] = prev.get("access_token")
        cred["expires_at"] = prev.get("expires_at", 0) or 0
        cred["_failed_at"] = prev.get("_failed_at", 0) or 0
        cred["_preview_at"] = prev.get("_preview_at", 0) or 0


def reload_credentials(force: bool = False) -> bool:
    """Re-read token.json when it changes, so adding an account needs no restart.

    A restart used to be the only way to load a new account, and it also threw
    away every in-memory access token (forcing N refreshes at once). This keeps
    live tokens for accounts that are still present and swaps the pool only
    when the set of accounts actually changed.
    """
    global _creds, _catalog_cred, _playback_pool, _TOKEN_FILE_MTIME
    global CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN

    try:
        mtime = os.path.getmtime(TOKEN_FILE)
    except OSError:
        if not force:
            return False
        mtime = 0.0
    if not force and mtime and mtime == _TOKEN_FILE_MTIME:
        return False
    _TOKEN_FILE_MTIME = mtime

    new_creds: List[dict] = []
    new_catalog: Optional[dict] = None
    if mtime:
        try:
            with open(TOKEN_FILE, "r") as tok:
                token_data = json.load(tok)
        except Exception as exc:
            logger.warning("Could not reload %s: %s", TOKEN_FILE, exc)
            return False
        if isinstance(token_data, dict):
            token_data = [token_data]
        for entry in token_data:
            if not isinstance(entry, dict):
                continue
            cred = _credential_from_entry(entry)
            if not cred["refresh_token"]:
                continue
            is_catalog = entry.get("role") == "catalog" or entry.get("catalog") is True
            if is_catalog:
                if new_catalog is None:
                    new_catalog = cred
                else:
                    logger.warning("Ignoring additional catalog credential in %s", TOKEN_FILE)
            elif not any(c["refresh_token"] == cred["refresh_token"] for c in new_creds):
                new_creds.append(cred)

    # Add env var credential if available and unique (simple check)
    if _ENV_REFRESH_TOKEN and not any(c["refresh_token"] == _ENV_REFRESH_TOKEN for c in new_creds):
        new_creds.append({
            "client_id": _ENV_CLIENT_ID,
            "client_secret": _ENV_CLIENT_SECRET,
            "refresh_token": _ENV_REFRESH_TOKEN,
            "user_id": _ENV_USER_ID,
            "access_token": None,
            "expires_at": 0,
            "_failed_at": 0,
        })

    if new_catalog is None and CATALOG_REFRESH_TOKEN:
        new_catalog = {
            "client_id": CATALOG_CLIENT_ID,
            "client_secret": CATALOG_CLIENT_SECRET,
            "refresh_token": CATALOG_REFRESH_TOKEN,
            "user_id": CATALOG_USER_ID,
            "access_token": None,
            "expires_at": 0,
        }

    if new_catalog is not None:
        # A credential explicitly reserved for catalog traffic must never also
        # become a playback slot, even if it was duplicated via environment vars.
        new_creds = [
            cred for cred in new_creds
            if cred["refresh_token"] != new_catalog["refresh_token"]
        ]
        if _catalog_cred is not None:
            _carry_live_state([_catalog_cred], [new_catalog])

    _carry_live_state(_creds, new_creds)
    changed = (
        [c["refresh_token"] for c in new_creds]
        != [c["refresh_token"] for c in _creds]
    )
    _creds[:] = new_creds
    _catalog_cred = new_catalog
    if changed and _playback_pool is not None:
        _playback_pool = PlaybackCredentialPool(_creds)
    if _creds:
        CLIENT_ID = _creds[0]["client_id"]
        CLIENT_SECRET = _creds[0]["client_secret"]
        REFRESH_TOKEN = _creds[0]["refresh_token"]
    logger.info("Loaded %d playback credential(s) from %s", len(_creds), TOKEN_FILE)
    return True


reload_credentials(force=True)


FAILED_COOLDOWN_SECONDS = 300
# A PREVIEW answer means the account has no active subscription
# (previewReason=FULL_REQUIRES_SUBSCRIPTION). Keep it out of playback for six
# hours; a plan can be restored, so this is a cooldown rather than a deletion.
PREVIEW_COOLDOWN_SECONDS = 6 * 60 * 60


def _is_failed(cred: dict) -> bool:
    """True while a refused credential is in its cooldown window."""
    at = cred.get("_failed_at") or 0
    return bool(at) and (time.time() - at) < FAILED_COOLDOWN_SECONDS


def _is_preview_broken(cred: dict) -> bool:
    """True while a free-tier account is in its preview cooldown window."""
    at = cred.get("_preview_at") or 0
    return bool(at) and (time.time() - at) < PREVIEW_COOLDOWN_SECONDS


def _is_preview_response(result: Any) -> bool:
    """True for a 200 that only contains a 30-second PREVIEW asset.

    /trackManifests wraps Tidal's payload under data.data.attributes; /track
    returns data.assetPresentation directly.
    """
    if not isinstance(result, dict):
        return False
    data = result.get("data")
    if not isinstance(data, dict):
        return False
    inner = data.get("data")
    if isinstance(inner, dict):
        attrs = inner.get("attributes") or {}
        presentation = attrs.get("trackPresentation")
        if presentation:
            return str(presentation).upper() != "FULL"
    presentation = data.get("assetPresentation")
    if presentation:
        return str(presentation).upper() != "FULL"
    return False


def _pick_credential() -> dict:
    reload_credentials()
    if not _creds:
        raise HTTPException(status_code=500, detail="No Tidal credentials available; populate token.json")
    # Prefer accounts that are neither refused nor preview-only. Fall back to
    # merely-not-refused, then to everything, so a pool of free-tier accounts
    # still answers instead of dead-ending the request.
    healthy = [c for c in _creds if not _is_failed(c) and not _is_preview_broken(c)]
    if not healthy:
        healthy = [c for c in _creds if not _is_failed(c)]
    return random.choice(healthy or _creds)


def _lock_for_cred(cred: dict) -> asyncio.Lock:
    key = f"{cred['client_id']}:{cred['refresh_token']}"
    lock = _refresh_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _refresh_locks[key] = lock
    return lock


def _task_brief(task: Optional[asyncio.Task]) -> Optional[str]:
    """Where a task is currently suspended, for the lease report."""
    if task is None or task.done():
        return None
    try:
        frames = task.get_stack()
    except Exception:
        return None
    if not frames:
        return None
    parts = [
        f"{frame.f_code.co_name} ({os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno})"
        for frame in frames[-6:]
    ]
    return " < ".join(parts)


class PlaybackCredentialPool:
    """Lease each playback credential to at most one request at a time.

    Every lease remembers the task that took it and when. A slot that is held
    past PLAYBACK_LEASE_TIMEOUT is reclaimed by the watchdog, because one
    wedged request used to hold a slot forever and, with every slot held, the
    instance answered 202 to every playback request and never recovered
    without a restart.
    """

    def __init__(self, credentials: List[dict]):
        self._available: asyncio.Queue[dict] = asyncio.Queue()
        self._size = len(credentials)
        self._members = {id(cred) for cred in credentials}
        self._leases: Dict[int, dict] = {}
        for cred in credentials:
            self._available.put_nowait(cred)

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._available.qsize()

    def _track(self, cred: dict, owner: Optional[str] = None) -> None:
        self._leases[id(cred)] = {
            "cred": cred,
            "task": asyncio.current_task(),
            "at": time.time(),
            "owner": owner,
        }

    def try_acquire(self, owner: Optional[str] = None) -> Optional[dict]:
        if self._size == 0:
            raise HTTPException(status_code=500, detail="No Tidal playback credentials available; populate token.json")
        try:
            cred = self._available.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._track(cred, owner)
        return cred

    def release(self, cred: dict) -> None:
        # Only the first release of a lease counts. A reclaimed or already
        # returned credential must not be handed out twice.
        if self._leases.pop(id(cred), None) is None:
            return
        self._available.put_nowait(cred)

    async def acquire(self, timeout: float, owner: Optional[str] = None) -> Optional[dict]:
        """Wait up to ``timeout`` for a free account instead of failing at once."""
        if self._size == 0:
            raise HTTPException(status_code=500, detail="No Tidal playback credentials available; populate token.json")
        try:
            cred = await asyncio.wait_for(self._available.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        self._track(cred, owner)
        return cred

    @asynccontextmanager
    async def lease(self, owner: Optional[str] = None) -> AsyncIterator[dict]:
        if self._size == 0:
            raise HTTPException(status_code=500, detail="No Tidal playback credentials available; populate token.json")

        cred = await self._available.get()
        self._track(cred, owner)
        try:
            yield cred
        finally:
            self.release(cred)

    def stale(self, timeout: float) -> List[dict]:
        now = time.time()
        return [
            {**info, "age": now - info["at"]}
            for info in list(self._leases.values())
            if now - info["at"] >= timeout
        ]

    def report(self) -> List[dict]:
        now = time.time()
        return [
            {
                "account": str(info["cred"].get("userID") or info["cred"].get("user_id") or "?"),
                "ageSeconds": round(now - info["at"], 1),
                "requestId": info.get("owner"),
                "awaiting": _task_brief(info["task"]),
            }
            for info in list(self._leases.values())
        ]


_playback_pool = PlaybackCredentialPool(_creds)


@dataclass
class PlaybackJob:
    request_id: str
    operation: Optional[Callable[[dict], Awaitable[Any]]]
    created_at: float
    state: str = "pending"
    result: Any = None
    error: Optional[str] = None
    error_status: int = 500
    finished_at: Optional[float] = None
    task: Optional[asyncio.Task] = None


_playback_jobs: Dict[str, PlaybackJob] = {}
# Identical requests collapse onto one job. Clients retry the same playback
# URL while it is pending, and every retry used to create another job: the
# queue only grew while the pool was busy.
_playback_jobs_by_key: Dict[tuple, str] = {}
_PLAYBACK_JOB_TTL_SECONDS = 300
_playback_watchdog_task: Optional[asyncio.Task] = None


def _active_job_count() -> int:
    return sum(1 for job in _playback_jobs.values() if job.state in ("pending", "processing"))


def _prune_playback_jobs() -> None:
    cutoff = time.time() - _PLAYBACK_JOB_TTL_SECONDS
    expired = [
        request_id
        for request_id, job in _playback_jobs.items()
        if job.finished_at is not None and job.finished_at < cutoff
    ]
    for request_id in expired:
        _playback_jobs.pop(request_id, None)
    for key, request_id in list(_playback_jobs_by_key.items()):
        if request_id not in _playback_jobs:
            _playback_jobs_by_key.pop(key, None)


def _fail_job(job: PlaybackJob, detail: str, status: int = 503) -> None:
    """Give a queued job a terminal answer instead of leaving it pending."""
    if job.state not in ("pending", "processing"):
        return
    job.state = "failed"
    job.error = detail
    job.error_status = status
    job.finished_at = time.time()
    if job.task is not None and not job.task.done():
        job.task.cancel()


def _expire_stalled_jobs() -> None:
    """Fail jobs that waited longer than PLAYBACK_JOB_TIMEOUT."""
    now = time.time()
    for job in list(_playback_jobs.values()):
        if job.state in ("pending", "processing") and now - job.created_at >= PLAYBACK_JOB_TIMEOUT:
            logger.warning(
                "Playback request %s waited %.0fs for a free account; failing it",
                job.request_id,
                now - job.created_at,
            )
            _fail_job(job, "Playback queue waited too long for a free account")


async def _playback_watchdog() -> None:
    """Reclaim playback slots that a request never gave back."""
    while True:
        await asyncio.sleep(_PLAYBACK_WATCHDOG_INTERVAL)
        try:
            for info in _playback_pool.stale(PLAYBACK_LEASE_TIMEOUT):
                cred = info["cred"]
                logger.error(
                    "Reclaiming playback slot for account %s after %.0fs (request %s, task %s, awaiting %s)",
                    cred.get("userID") or cred.get("user_id") or "?",
                    info["age"],
                    info.get("owner"),
                    getattr(info.get("task"), "get_name", lambda: None)(),
                    _task_brief(info.get("task")),
                )
                job = _playback_jobs.get(info.get("owner") or "")
                if job is not None:
                    _fail_job(job, "Playback account slot timed out")
                _playback_pool.release(cred)
            _expire_stalled_jobs()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Playback watchdog failed")


def start_playback_watchdog() -> None:
    global _playback_watchdog_task
    if _playback_watchdog_task is None or _playback_watchdog_task.done():
        _playback_watchdog_task = asyncio.create_task(_playback_watchdog())


async def stop_playback_watchdog() -> None:
    global _playback_watchdog_task
    if _playback_watchdog_task is None:
        return
    _playback_watchdog_task.cancel()
    try:
        await _playback_watchdog_task
    except (asyncio.CancelledError, Exception):
        pass
    _playback_watchdog_task = None


def _playback_job_position(job: PlaybackJob) -> int:
    if job.state != "pending":
        return 0
    pending = [queued for queued in _playback_jobs.values() if queued.state == "pending"]
    try:
        return pending.index(job) + 1
    except ValueError:
        return 0


def _playback_job_payload(job: PlaybackJob) -> dict:
    status_url = f"/playback/requests/{job.request_id}"
    return {
        "status": job.state,
        "requestId": job.request_id,
        "queuePosition": _playback_job_position(job),
        "statusUrl": status_url,
        "cancelUrl": status_url,
        "playbackAccounts": _playback_pool.size,
        "activePlaybackRequests": _playback_pool.size - _playback_pool.available,
    }


def _pending_playback_response(job: PlaybackJob) -> JSONResponse:
    position = _playback_job_position(job)
    return JSONResponse(
        status_code=202,
        content=_playback_job_payload(job),
        headers={
            "Location": f"/playback/requests/{job.request_id}",
            "Retry-After": "1",
            "X-Playback-Queue-Position": str(position),
            "X-Playback-Request-Id": job.request_id,
        },
    )


async def _run_queued_playback_job(job: PlaybackJob) -> None:
    try:
        job.state = "processing"
        if job.operation is None:
            raise RuntimeError("Playback operation is missing")
        # Same preview guard as the immediate path: a free-tier draw is marked
        # and retried with the next leased account instead of being returned.
        last_preview = None
        attempts = max(1, min(_playback_pool.size, 6))
        for _ in range(attempts):
            async with _playback_pool.lease(owner=job.request_id) as cred:
                result = await job.operation(cred)
                if not _is_preview_response(result):
                    cred["_preview_at"] = 0
                    job.result = result
                    job.state = "completed"
                    return
                cred["_preview_at"] = time.time()
                last_preview = result
        job.result = last_preview
        job.state = "completed"
    except asyncio.CancelledError:
        job.state = "cancelled"
        raise
    except HTTPException as exc:
        job.state = "failed"
        job.error = str(exc.detail)
        job.error_status = exc.status_code
    except Exception:
        logger.exception("Queued playback request %s failed", job.request_id)
        job.state = "failed"
        job.error = "Playback request failed"
        job.error_status = 500
    finally:
        job.finished_at = time.time()
        job.operation = None


async def dispatch_playback_request(operation: Callable[[dict], Awaitable[Any]], key: Optional[tuple] = None):
    """Run immediately when possible; otherwise return a pollable 202 job.

    A preview answer means the account behind the request has no subscription.
    Mark it preview-broken and immediately retry with the next leased account
    so a client is never handed a 30-second clip just because the pool drew a
    free-tier account.

    ``key`` identifies the request. While an identical request is still
    pending, its job is returned instead of queueing another one, so a client
    retrying a 202 cannot bury the instance in duplicate jobs.
    """
    if key is not None:
        existing_id = _playback_jobs_by_key.get(key)
        existing = _playback_jobs.get(existing_id) if existing_id else None
        if existing is not None and existing.state in ("pending", "processing"):
            return _pending_playback_response(existing)

    cred = _playback_pool.try_acquire()
    if cred is None and PLAYBACK_QUEUE_HOLD > 0:
        cred = await _playback_pool.acquire(PLAYBACK_QUEUE_HOLD)
    if cred is not None:
        last_preview = None
        attempts = max(1, min(_playback_pool.size, 6))
        for _ in range(attempts):
            try:
                result = await operation(cred)
            finally:
                _playback_pool.release(cred)
            if not _is_preview_response(result):
                cred["_preview_at"] = 0
                return result
            cred["_preview_at"] = time.time()
            last_preview = result
            cred = _playback_pool.try_acquire()
            if cred is None:
                break
        if last_preview is not None:
            return last_preview
        # Every credential was busy during the retries - queue a job.

    _prune_playback_jobs()
    if _active_job_count() >= PLAYBACK_MAX_PENDING_JOBS:
        return JSONResponse(
            status_code=503,
            content={"detail": "Playback queue is full; retry shortly"},
            headers={"Retry-After": "2"},
        )
    job = PlaybackJob(
        request_id=uuid.uuid4().hex,
        operation=operation,
        created_at=time.time(),
    )
    _playback_jobs[job.request_id] = job
    if key is not None:
        _playback_jobs_by_key[key] = job.request_id
    job.task = asyncio.create_task(_run_queued_playback_job(job))
    return _pending_playback_response(job)


async def get_http_client() -> httpx.AsyncClient:
    global _http_client, _http_client_proxy_url
    if _http_client is None:
        async with _http_client_lock:
            if _http_client is None:
                proxy_url = None
                if USE_PROXIES:
                    proxy_url = await get_working_proxy()
                    if not proxy_url and not FALLBACK_TO_DIRECT_CONNECTION:
                        raise HTTPException(status_code=503, detail="Service Unavailable")
                    elif not proxy_url and FALLBACK_TO_DIRECT_CONNECTION:
                        logger.warning("Could not find a working proxy, falling back to direct connection. HOST IP MAY BE EXPOSED!")
                _http_client = _build_http_client(proxy_url)
                _http_client_proxy_url = proxy_url
    return _http_client


async def refresh_tidal_token(cred: Optional[dict] = None):
    """Refresh a token for the provided credential set."""
    cred = cred or _pick_credential()

    async with _lock_for_cred(cred):
        if cred["access_token"] and time.time() < cred["expires_at"]:
            return cred["access_token"]

        if USE_PROXIES and ROTATE_PROXIES_ON_REFRESH:
            await update_global_client(force_new_proxy=True)

        max_retries = MAX_RETRIES if USE_PROXIES else 1
        for attempt in range(max_retries):
            try:
                client = await get_http_client()
                res = await client.post(
                    "https://auth.tidal.com/v1/oauth2/token",
                    data={
                        "client_id": cred["client_id"],
                        "refresh_token": cred["refresh_token"],
                        "grant_type": "refresh_token",
                        "scope": "r_usr+w_usr+w_sub",
                    },
                    auth=(cred["client_id"], cred["client_secret"]),
                )
                _log_response("POST", "https://auth.tidal.com/v1/oauth2/token", res)

                if res.status_code in [400, 401]:
                    try:
                        error_data = res.json()
                        if error_data.get("error") in ["invalid_client", "invalid_grant"]:
                            logger.error(f"Tidal Auth Error: {error_data}")
                            cred["_failed_at"] = time.time()
                            raise HTTPException(status_code=401, detail=f"Tidal Auth Error: {error_data.get('error_description')}")
                    except ValueError:
                        pass

                res.raise_for_status()
                data = res.json()
                new_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)

                cred["access_token"] = new_token
                cred["expires_at"] = time.time() + expires_in - 60
                cred["_failed_at"] = 0

                return new_token
            except httpx.RequestError as e:
                # A transport error is the network, not a refused account.
                # Marking the credential dead here made one blip look like a
                # revoked login and demoted the whole instance.
                if USE_PROXIES and attempt < max_retries - 1:
                    logger.warning(f"Proxy failed during token refresh: {e}. Healing proxy...")
                    await update_global_client(force_new_proxy=True)
                    continue
                raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")
            except httpx.HTTPStatusError as e:
                # Only a refused login condemns the account. 429/5xx are
                # transient and must not take the credential out of rotation.
                if e.response.status_code in (401, 403):
                    cred["_failed_at"] = time.time()
                if USE_PROXIES and e.response.status_code in [403, 429] and attempt < max_retries - 1:
                    logger.warning(f"Proxy blocked during token refresh ({e.response.status_code}). Healing proxy...")
                    await update_global_client(force_new_proxy=True)
                    continue
                raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")


async def get_tidal_token(force_refresh: bool = False):
    return await get_tidal_token_for_cred(force_refresh=force_refresh)


async def get_tidal_token_for_cred(force_refresh: bool = False, cred: Optional[dict] = None):
    """Retrieve an access token for a specific credential; pick random if not provided."""
    cred = cred or _pick_credential()

    if not force_refresh and cred["access_token"] and time.time() < cred["expires_at"]:
        return cred["access_token"], cred

    token = await refresh_tidal_token(cred)
    return token, cred


async def get_catalog_token_for_cred(force_refresh: bool = False, cred: Optional[dict] = None):
    """Return the unrestricted catalog credential, falling back for compatibility."""
    reload_credentials()
    if cred is not None:
        return await get_tidal_token_for_cred(force_refresh=force_refresh, cred=cred)
    if _catalog_cred is not None:
        return await get_tidal_token_for_cred(force_refresh=force_refresh, cred=_catalog_cred)
    if CATALOG_TOKEN:
        return CATALOG_TOKEN, None
    return await get_tidal_token_for_cred(force_refresh=force_refresh)


async def _get_request_token(*, catalog: bool, force_refresh: bool = False, cred: Optional[dict] = None):
    if catalog:
        return await get_catalog_token_for_cred(force_refresh=force_refresh, cred=cred)
    return await get_tidal_token_for_cred(force_refresh=force_refresh, cred=cred)


async def make_request(
    url: str,
    token: Optional[str] = None,
    params: Optional[dict] = None,
    cred: Optional[dict] = None,
    *,
    catalog: bool = True,
):
    if token is None:
        token, cred = await _get_request_token(catalog=catalog, cred=cred)
    client = await get_http_client()
    headers = {"authorization": f"Bearer {token}"}

    try:
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            resp = await client.get(url, headers=headers, params=params)
            _log_response("GET", url, resp)

            if resp.status_code == 401:
                token, cred = await _get_request_token(catalog=catalog, force_refresh=True, cred=cred)
                headers = {"authorization": f"Bearer {token}"}
                resp = await client.get(url, headers=headers, params=params)
                _log_response("GET (retry after 401)", url, resp)

            if resp.status_code == 429 and attempt < _RATE_LIMIT_MAX_RETRIES:
                delay = min(_RATE_LIMIT_BASE_DELAY * (2 ** attempt), _RATE_LIMIT_MAX_DELAY)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(delay, max(float(retry_after), 0))
                    except ValueError:
                        pass
                delay = min(delay, _RATE_LIMIT_MAX_DELAY)
                logger.warning("Upstream 429 for %s, retrying in %.1fs (attempt %d/%d)", url, delay, attempt + 1, _RATE_LIMIT_MAX_RETRIES)
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 404:
                fresh_token, fresh_cred = await _get_request_token(catalog=catalog, force_refresh=True, cred=cred)
                if fresh_token != token:
                    headers = {"authorization": f"Bearer {fresh_token}"}
                    resp = await client.get(url, headers=headers, params=params)
                    _log_response("GET (retry after 404 token refresh)", url, resp)
                    token, cred = fresh_token, fresh_cred

            break

        resp.raise_for_status()
        return {"version": API_VERSION, "data": resp.json()}
    except httpx.HTTPStatusError as e:
        logger.error(
            "Upstream API error %s %s %s",
            e.response.status_code,
            url,
            e.response.text[:1000],
            exc_info=e,
        )
        raise HTTPException(status_code=e.response.status_code, detail="Upstream API error")
    except httpx.RequestError as e:
        if isinstance(e, httpx.TimeoutException):
            raise HTTPException(status_code=429, detail="Upstream timeout")
        raise HTTPException(status_code=503, detail="Connection error to Tidal")


async def authed_get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    token: Optional[str] = None,
    cred: Optional[dict] = None,
    catalog: bool = True,
):
    """Perform an authenticated GET, retrying once on 401. Returns payload with updated token/cred."""

    if token is None:
        token, cred = await _get_request_token(catalog=catalog, cred=cred)

    client = await get_http_client()
    headers = {"authorization": f"Bearer {token}"}

    try:
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            resp = await client.get(url, headers=headers, params=params)
            _log_response("GET", url, resp)

            if resp.status_code == 401:
                token, cred = await _get_request_token(catalog=catalog, force_refresh=True, cred=cred)
                headers["authorization"] = f"Bearer {token}"
                resp = await client.get(url, headers=headers, params=params)
                _log_response("GET (retry after 401)", url, resp)

            if resp.status_code == 429 and attempt < _RATE_LIMIT_MAX_RETRIES:
                delay = min(_RATE_LIMIT_BASE_DELAY * (2 ** attempt), _RATE_LIMIT_MAX_DELAY)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(delay, max(float(retry_after), 0))
                    except ValueError:
                        pass
                delay = min(delay, _RATE_LIMIT_MAX_DELAY)
                logger.warning("Upstream 429 for %s, retrying in %.1fs (attempt %d/%d)", url, delay, attempt + 1, _RATE_LIMIT_MAX_RETRIES)
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 404:
                fresh_token, fresh_cred = await _get_request_token(catalog=catalog, force_refresh=True, cred=cred)
                if fresh_token != token:
                    headers["authorization"] = f"Bearer {fresh_token}"
                    resp = await client.get(url, headers=headers, params=params)
                    _log_response("GET (retry after 404 token refresh)", url, resp)
                    token, cred = fresh_token, fresh_cred

            break

        resp.raise_for_status()
        return resp.json(), token, cred
    except httpx.HTTPStatusError as e:
        logger.error(
            "Upstream API error %s %s %s",
            e.response.status_code,
            url,
            e.response.text[:1000],
            exc_info=e,
        )
        raise HTTPException(status_code=e.response.status_code, detail="Upstream API error")
    except httpx.RequestError as e:
        if isinstance(e, httpx.TimeoutException):
            raise HTTPException(status_code=429, detail="Upstream timeout")
        raise HTTPException(status_code=503, detail="Connection error to Tidal")


async def make_playback_request_for_cred(cred: dict, url: str, params: Optional[dict] = None):
    token, cred = await get_tidal_token_for_cred(cred=cred)
    return await make_request(url, token=token, params=params, cred=cred, catalog=False)


async def make_playback_request(url: str, params: Optional[dict] = None):
    """Run one complete playback operation in an exclusive account slot."""
    async with _playback_pool.lease() as cred:
        return await make_playback_request_for_cred(cred, url, params=params)


@app.get("/playback/state")
async def get_playback_state():
    """What the playback queue and every account slot are doing right now."""
    _prune_playback_jobs()
    counts: Dict[str, int] = {}
    for job in _playback_jobs.values():
        counts[job.state] = counts.get(job.state, 0) + 1
    return {
        "version": API_VERSION,
        "pool": {
            "accounts": _playback_pool.size,
            "available": _playback_pool.available,
            "leases": _playback_pool.report(),
        },
        "jobs": counts,
        "oldestJobAgeSeconds": round(
            max((time.time() - job.created_at for job in _playback_jobs.values()), default=0.0),
            1,
        ),
        "limits": {
            "leaseTimeoutSeconds": PLAYBACK_LEASE_TIMEOUT,
            "jobTimeoutSeconds": PLAYBACK_JOB_TIMEOUT,
            "maxPendingJobs": PLAYBACK_MAX_PENDING_JOBS,
            "queueHoldSeconds": PLAYBACK_QUEUE_HOLD,
        },
    }


@app.get("/playback/requests/{request_id}")
async def get_playback_request(request_id: str):
    """Poll a playback request that was accepted with HTTP 202."""
    _prune_playback_jobs()
    job = _playback_jobs.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Playback request not found or expired")

    if job.state in ("pending", "processing"):
        return _pending_playback_response(job)
    if job.state == "completed":
        return job.result
    if job.state == "cancelled":
        return JSONResponse(
            status_code=410,
            content={**_playback_job_payload(job), "detail": "Playback request was cancelled"},
        )
    return JSONResponse(
        status_code=job.error_status,
        content={**_playback_job_payload(job), "detail": job.error or "Playback request failed"},
    )


@app.delete("/playback/requests/{request_id}")
async def cancel_playback_request(request_id: str):
    """Cancel a pending or processing playback request and release its slot."""
    _prune_playback_jobs()
    job = _playback_jobs.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Playback request not found or expired")
    if job.state in ("pending", "processing") and job.task is not None:
        job.task.cancel()
        job.state = "cancelled"
        job.finished_at = time.time()
    return _playback_job_payload(job)


@app.get("/")
async def index():
    # The tidy addon's health probe hits this to decide whether this
    # instance is worth trying. An instance whose Tidal credentials are dead
    # answers every playback request with a 401, so answering "healthy" here
    # made the addon queue it FIRST on every track - it is the fastest to
    # reach, being the closest - and every first play of a lossless track
    # paid for a doomed probe before the working instance was reached.
    # A failure is remembered per credential, so a transient hiccup on one
    # of several accounts does not condemn the instance.
    if _creds:
        broken = [c for c in _creds if _is_failed(c)]
        if len(broken) == len(_creds):
            raise HTTPException(
                status_code=503,
                detail="Tidal credentials for every account have been refused",
            )
    return {"version": API_VERSION, "Repo": "https://github.com/binimum/hifi-api"}

@app.get("/info/")
async def get_info(id: int):
    url = f"https://api.tidal.com/v1/tracks/{id}/"
    return await make_request(url, params={"countryCode": COUNTRY_CODE})

@app.get("/track/")
async def get_track(id: int, quality: str = "HI_RES_LOSSLESS", immersiveaudio: bool = False):
    track_url = f"https://api.tidal.com/v1/tracks/{id}/playbackinfo"
    params = {
        "audioquality": quality,
        "playbackmode": "STREAM",
        "assetpresentation": "FULL",
        "immersiveaudio": immersiveaudio
    }
    return await dispatch_playback_request(
        lambda cred: make_playback_request_for_cred(cred, track_url, params=params),
        key=("track", id, quality, immersiveaudio),
    )


@app.get("/trackManifests/")
async def get_track_manifests(
    id: str,
    request: Request,
    formats: List[str] = Query(default=["HEAACV1", "AACLC", "FLAC", "FLAC_HIRES", "EAC3_JOC"]),
    adaptive: str = Query(default="true"),
    manifestType: str = Query(default="MPEG_DASH"),
    uriScheme: str = Query(default="HTTPS"),
    usage: str = Query(default="PLAYBACK")
):
    url = f"https://openapi.tidal.com/v2/trackManifests/{id}"
    params = [
        ("adaptive", adaptive),
        ("manifestType", manifestType),
        ("uriScheme", uriScheme),
        ("usage", usage),
    ]
    for f in formats:
        params.append(("formats", f))

    async def fetch_manifest(cred: dict):
        res = await make_playback_request_for_cred(cred, url, params=params)
        try:
            drm_data = res["data"]["data"]["attributes"]["drmData"]
            if drm_data:
                proxy_url = str(request.base_url).rstrip("/") + "/widevine"
                drm_data["licenseUrl"] = proxy_url
                drm_data["certificateUrl"] = proxy_url
        except (KeyError, TypeError):
            pass
        return res

    return await dispatch_playback_request(
        fetch_manifest,
        key=("manifest", id, tuple(formats), adaptive, manifestType, uriScheme, usage),
    )

# Not really necessary but I'm including it anyway
@app.api_route("/widevine", methods=["GET", "POST"])
async def widevine_proxy(request: Request):
    client = await get_http_client()
    body = await request.body()
    url = "https://api.tidal.com/v2/widevine"

    async def fetch_license(cred: dict):
        token, cred = await get_tidal_token_for_cred(cred=cred)
        headers = {
            "authorization": f"Bearer {token}",
            "Content-Type": request.headers.get("Content-Type", "application/octet-stream")
        }

        try:
            resp = await client.request(request.method, url, headers=headers, content=body)
            _log_response(request.method, url, resp)

            if resp.status_code == 401:
                token, cred = await get_tidal_token_for_cred(force_refresh=True, cred=cred)
                headers["authorization"] = f"Bearer {token}"
                resp = await client.request(request.method, url, headers=headers, content=body)
                _log_response(f"{request.method} (retry)", url, resp)

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={"Content-Type": resp.headers.get("Content-Type", "application/json")}
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail="Error communicating with widevine server") from e

    return await dispatch_playback_request(fetch_license)


@app.get("/recommendations/")
async def get_recommendations(id: int):
    recommendations_url = f"https://api.tidal.com/v1/tracks/{id}/recommendations"
    params = {"limit": "20", "countryCode": COUNTRY_CODE}
    return await make_request(recommendations_url, params=params)


@app.api_route("/search/", methods=["GET"])
async def search(
    s: Union[str, None] = Query(default=None),
    a: Union[str, None] = Query(default=None),
    al: Union[str, None] = Query(default=None),
    v: Union[str, None] = Query(default=None),
    p: Union[str, None] = Query(default=None),
    i: Union[str, None] = Query(default=None, description="ISRC query"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=500),
):
    """Search endpoint supporting track/artist/album/video/playlist queries via distinct params."""
    isrc_query = i.strip() if isinstance(i, str) else None
    if isrc_query:
        return await make_request(
            "https://api.tidal.com/v1/tracks",
            params={
                "isrc": isrc_query,
                "limit": limit,
                "offset": offset,
                "countryCode": COUNTRY_CODE,
            },
        )

    queries = (
        (s, "https://api.tidal.com/v1/search/tracks", {
            "query": s,
            "limit": limit,
            "offset": offset,
            "countryCode": COUNTRY_CODE,
        }),
        (a, "https://api.tidal.com/v1/search/top-hits", {
            "query": a,
            "limit": limit,
            "offset": offset,
            "types": "ARTISTS,TRACKS",
            "countryCode": COUNTRY_CODE,
        }),
        (al, "https://api.tidal.com/v1/search/top-hits", {
            "query": al,
            "limit": limit,
            "offset": offset,
            "types": "ALBUMS",
            "countryCode": COUNTRY_CODE,
        }),
        (v, "https://api.tidal.com/v1/search/top-hits", {
            "query": v,
            "limit": limit,
            "offset": offset,
            "types": "VIDEOS",
            "countryCode": COUNTRY_CODE,
        }),
        (p, "https://api.tidal.com/v1/search/top-hits", {
            "query": p,
            "limit": limit,
            "offset": offset,
            "types": "PLAYLISTS",
            "countryCode": COUNTRY_CODE,
        }),
    )

    for value, url, params in queries:
        if value:
            return await make_request(url, params=params)

    raise HTTPException(status_code=400, detail="Provide one of s, a, al, v, p, or i")

@app.get("/album/")
async def get_album(
    id: int = Query(..., description="Album ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    token, cred = await get_catalog_token_for_cred()

    album_url = f"https://api.tidal.com/v1/albums/{id}"
    items_url = f"https://api.tidal.com/v1/albums/{id}/items"

    async def fetch(url: str, params: Optional[dict] = None):
        payload, _, _ = await authed_get_json(
            url,
            params=params,
            token=token,
            cred=cred,
        )
        return payload

    tasks = [fetch(album_url, {"countryCode": COUNTRY_CODE})]

    max_chunk = 100
    current_offset = offset
    remaining_limit = limit

    while remaining_limit > 0:
        chunk_size = min(remaining_limit, max_chunk)
        tasks.append(
            fetch(items_url, {"countryCode": COUNTRY_CODE, "limit": chunk_size, "offset": current_offset})
        )
        current_offset += chunk_size
        remaining_limit -= chunk_size

    results = await asyncio.gather(*tasks)

    album_data = results[0]
    items_pages = results[1:]

    all_items = []
    for page in items_pages:
        page_items = page.get("items", page) if isinstance(page, dict) else page
        if isinstance(page_items, list):
            all_items.extend(page_items)

    album_data["items"] = all_items

    return {
        "version": API_VERSION,
        "data": album_data,
    }


@app.get("/mix/")
async def get_mix(
    id: str = Query(..., description="Mix ID")
):
    """Fetch items from a Tidal mix by its ID."""
    token, cred = await get_catalog_token_for_cred()
    url = "https://api.tidal.com/v1/pages/mix"
    params = {
        "mixId": id,
        "countryCode": COUNTRY_CODE,
        "deviceType": "BROWSER",
    }

    data, _, _ = await authed_get_json(
        url,
        params=params,
        token=token,
        cred=cred,
    )

    header = {}
    items = []

    rows = data.get("rows", [])
    for row in rows:
        modules = row.get("modules", [])
        for module in modules:
            if module.get("type") == "MIX_HEADER":
                header = module.get("mix", {})
            elif module.get("type") == "TRACK_LIST":
                paged_list = module.get("pagedList", {})
                items = paged_list.get("items", [])

    return {
        "version": API_VERSION,
        "mix": header,
        "items": [item.get("item", item) if isinstance(item, dict) else item for item in items],
    }


@app.get("/playlist/")
async def get_playlist(
    id: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Fetch playlist metadata plus items concurrently, using shared client and single token."""

    token, cred = await get_catalog_token_for_cred()

    playlist_url = f"https://api.tidal.com/v1/playlists/{id}"
    items_url = f"https://api.tidal.com/v1/playlists/{id}/items"

    async def fetch(url: str, params: Optional[dict] = None):
        payload, _, _ = await authed_get_json(
            url,
            params=params,
            token=token,
            cred=cred,
        )
        return payload

    playlist_data, items_data = await asyncio.gather(
        fetch(playlist_url, {"countryCode": COUNTRY_CODE}),
        fetch(items_url, {"countryCode": COUNTRY_CODE, "limit": limit, "offset": offset}),
    )

    return {
        "version": API_VERSION,
        "playlist": playlist_data,
        "items": items_data.get("items", items_data) if isinstance(items_data, dict) else items_data,
    }


def _extract_uuid_from_tidal_url(href: str) -> Optional[str]:
    """Extract and reconstruct a hyphenated UUID from a Tidal resource URL."""
    parts = href.split("/") if href else []
    return "-".join(parts[4:9]) if len(parts) >= 9 else None


@app.get("/artist/similar/")
async def get_similar_artists(
    id: int = Query(..., description="Artist ID"),
    cursor: Union[int, str, None] = None
):
    """Fetch artists similar to another by its ID using V2 API."""
    url = f"https://openapi.tidal.com/v2/artists/{id}/relationships/similarArtists"
    params = {
        "page[cursor]": cursor,
        "countryCode": COUNTRY_CODE,
        "include": "similarArtists,similarArtists.profileArt"
    }

    payload, _, _ = await authed_get_json(url, params=params)
    included = payload.get("included", [])
    artists_map = {i["id"]: i for i in included if i["type"] == "artists"}
    artworks_map = {i["id"]: i for i in included if i["type"] == "artworks"}

    def resolve_artist(entry):
        aid = entry["id"]
        inc = artists_map.get(aid, {})
        attr = inc.get("attributes", {})

        pic_id = None
        if art_data := inc.get("relationships", {}).get("profileArt", {}).get("data"):
            if artwork := artworks_map.get(art_data[0].get("id")):
                if files := artwork.get("attributes", {}).get("files"):
                    pic_id = _extract_uuid_from_tidal_url(files[0].get("href"))

        return {
            **attr,
            "id": int(aid) if str(aid).isdigit() else aid,
            "picture": pic_id or attr.get("selectedAlbumCoverFallback"),
            "url": f"http://www.tidal.com/artist/{aid}",
            "relationType": "SIMILAR_ARTIST"
        }

    return {
        "version": API_VERSION,
        "artists": [resolve_artist(e) for e in payload.get("data", [])]
    }


@app.get("/album/similar/")
async def get_similar_albums(
    id: int = Query(..., description="Album ID"),
    cursor: Union[int, str, None] = None
):
    """Fetch albums similar to another by its ID using V2 API."""
    url = f"https://openapi.tidal.com/v2/albums/{id}/relationships/similarAlbums"
    params = {
        "page[cursor]": cursor,
        "countryCode": COUNTRY_CODE,
        "include": "similarAlbums,similarAlbums.coverArt,similarAlbums.artists"
    }

    payload, _, _ = await authed_get_json(url, params=params)
    included = payload.get("included", [])
    albums_map = {i["id"]: i for i in included if i["type"] == "albums"}
    artworks_map = {i["id"]: i for i in included if i["type"] == "artworks"}
    artists_map = {i["id"]: i for i in included if i["type"] == "artists"}

    def resolve_album(entry):
        aid = entry["id"]
        inc = albums_map.get(aid, {})
        attr = inc.get("attributes", {})

        cover_id = None
        if art_data := inc.get("relationships", {}).get("coverArt", {}).get("data"):
            if artwork := artworks_map.get(art_data[0].get("id")):
                if files := artwork.get("attributes", {}).get("files"):
                    cover_id = _extract_uuid_from_tidal_url(files[0].get("href"))

        artist_list = []
        if art_data := inc.get("relationships", {}).get("artists", {}).get("data"):
             for a_entry in art_data:
                 if a_obj := artists_map.get(a_entry["id"]):
                     a_id = a_obj["id"]
                     artist_list.append({
                         "id": int(a_id) if str(a_id).isdigit() else a_id,
                         "name": a_obj["attributes"]["name"]
                     })

        return {
            **attr,
            "id": int(aid) if str(aid).isdigit() else aid,
            "cover": cover_id,
            "artists": artist_list,
            "url": f"http://www.tidal.com/album/{aid}"
        }

    return {
        "version": API_VERSION,
        "albums": [resolve_album(e) for e in payload.get("data", [])]
    }


@app.get("/artist/")
async def get_artist(
    id: Optional[int] = Query(default=None),
    f: Optional[int] = Query(default=None),
    skip_tracks: bool = Query(default=False),
):
    """Artist detail or album+track aggregation.

    - id: basic artist metadata + cover URLs
    - f: fetch artist albums page and aggregate tracks across albums (capped concurrency)
    - skip_tracks: if true, returns only albums without aggregating tracks (when using 'f')
    """

    if id is None and f is None:
        raise HTTPException(status_code=400, detail="Provide id or f query param")

    token, cred = await get_catalog_token_for_cred()

    if id is not None:
        artist_url = f"https://api.tidal.com/v1/artists/{id}"
        artist_data, token, cred = await authed_get_json(
            artist_url,
            params={"countryCode": COUNTRY_CODE},
            token=token,
            cred=cred,
        )

        picture = artist_data.get("picture")
        fallback = artist_data.get("selectedAlbumCoverFallback")

        if not picture and fallback:
            artist_data["picture"] = fallback
            picture = fallback

        cover = None
        if picture:
            slug = picture.replace("-", "/")
            cover = {
                "id": artist_data.get("id"),
                "name": artist_data.get("name"),
                "750": f"https://resources.tidal.com/images/{slug}/750x750.jpg",
            }

        return {"version": API_VERSION, "artist": artist_data, "cover": cover}

    # Fetch albums and singles/EPs directly in parallel
    albums_url = f"https://api.tidal.com/v1/artists/{f}/albums"
    common_params = {"countryCode": COUNTRY_CODE, "limit": 100}

    tasks = [
        authed_get_json(albums_url, params=common_params, token=token, cred=cred),
        authed_get_json(albums_url, params={**common_params, "filter": "EPSANDSINGLES"}, token=token, cred=cred),
    ]

    if skip_tracks:
        tasks.append(
            authed_get_json(
                f"https://api.tidal.com/v1/artists/{f}/toptracks",
                params={"countryCode": COUNTRY_CODE, "limit": 15},
                token=token,
                cred=cred
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    unique_releases = []
    seen_ids = set()

    # Process albums (first 2 results)
    for res in results[:2]:
        if isinstance(res, tuple) and len(res) > 0:
            data = res[0]
            items = data.get("items", []) if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("id") and item["id"] not in seen_ids:
                        unique_releases.append(item)
                        seen_ids.add(item["id"])
        elif isinstance(res, Exception):
            logger.warning("Error fetching artist releases: %s", res)

    album_ids: List[int] = [item["id"] for item in unique_releases]
    page_data = {"items": unique_releases}

    if skip_tracks:
        top_tracks = []
        if len(results) > 2:
            res = results[2]
            if isinstance(res, tuple) and len(res) > 0:
                data = res[0]
                top_tracks = data.get("items", []) if isinstance(data, dict) else data
            elif isinstance(res, Exception):
                logger.warning("Error fetching top tracks: %s", res)

        return {"version": API_VERSION, "albums": page_data, "tracks": top_tracks}

    if not album_ids:
        return {"version": API_VERSION, "albums": page_data, "tracks": []}

    async def fetch_album_tracks(album_id: int):
        async with _album_tracks_sem:
            album_data, _, _ = await authed_get_json(
                "https://api.tidal.com/v1/pages/album",
                params={
                    "albumId": album_id,
                    "countryCode": COUNTRY_CODE,
                    "deviceType": "BROWSER",
                },
                token=token,
                cred=cred,
            )

            rows = album_data.get("rows", [])
            if len(rows) < 2:
                return []
            modules = rows[1].get("modules", [])
            if not modules:
                return []
            paged_list = modules[0].get("pagedList", {})
            items = paged_list.get("items", [])
            tracks = [track.get("item", track) if isinstance(track, dict) else track for track in items]
            return tracks

    results = await asyncio.gather(
        *(fetch_album_tracks(album_id) for album_id in album_ids),
        return_exceptions=True,
    )

    tracks: List[dict] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        tracks.extend(res)

    return {"version": API_VERSION, "albums": page_data, "tracks": tracks}


@app.get("/cover/")
async def get_cover(
    id: Optional[int] = Query(default=None),
    q: Optional[str] = Query(default=None),
):
    """Fetch album cover data for a track id or search query."""

    if id is None and q is None:
        raise HTTPException(status_code=400, detail="Provide id or q query param")

    token, cred = await get_catalog_token_for_cred()

    def build_cover_entry(cover_slug: str, name: Optional[str], track_id: Optional[int]):
        slug = cover_slug.replace("-", "/")
        return {
            "id": track_id,
            "name": name,
            "1280": f"https://resources.tidal.com/images/{slug}/1280x1280.jpg",
            "640": f"https://resources.tidal.com/images/{slug}/640x640.jpg",
            "80": f"https://resources.tidal.com/images/{slug}/80x80.jpg",
        }

    if id is not None:
        track_data, token, cred = await authed_get_json(
            f"https://api.tidal.com/v1/tracks/{id}/",
            params={"countryCode": COUNTRY_CODE},
            token=token,
            cred=cred,
        )

        album = track_data.get("album") or {}
        cover_slug = album.get("cover")
        if not cover_slug:
            raise HTTPException(status_code=404, detail="Cover not found")

        entry = build_cover_entry(
            cover_slug,
            album.get("title") or track_data.get("title"),
            album.get("id") or id,
        )
        return {"version": API_VERSION, "covers": [entry]}

    search_data, token, cred = await authed_get_json(
        "https://api.tidal.com/v1/search/tracks",
        params={"countryCode": COUNTRY_CODE, "query": q, "limit": 10},
        token=token,
        cred=cred,
    )

    items = search_data.get("items", [])[:10]
    if not items:
        raise HTTPException(status_code=404, detail="Cover not found")

    covers = []
    for track in items:
        album = track.get("album") or {}
        cover_slug = album.get("cover")
        if not cover_slug:
            continue
        covers.append(
            build_cover_entry(
                cover_slug,
                track.get("title"),
                track.get("id"),
            )
        )

    if not covers:
        raise HTTPException(status_code=404, detail="Cover not found")

    return {"version": API_VERSION, "covers": covers}


@app.get("/lyrics/")
async def get_lyrics(id: int):
    url = f"https://api.tidal.com/v1/tracks/{id}/lyrics"
    data, token, cred = await authed_get_json(
        url,
        params={"countryCode": COUNTRY_CODE, "locale": "en_US", "deviceType": "BROWSER"},
    )

    if not data:
        raise HTTPException(status_code=404, detail="Lyrics not found")

    return {"version": API_VERSION, "lyrics": data}


@app.get("/topvideos/")
async def get_top_videos(
    countryCode: str = Query(default="US"),
    locale: str = Query(default="en_US"),
    deviceType: str = Query(default="BROWSER"),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Fetch recommended videos from Tidal."""
    token, cred = await get_catalog_token_for_cred()
    url = "https://api.tidal.com/v1/pages/mymusic_recommended_videos"
    params = {
        "countryCode": countryCode,
        "locale": locale,
        "deviceType": deviceType,
    }

    data, token, cred = await authed_get_json(
        url,
        params=params,
        token=token,
        cred=cred,
    )

    rows = data.get("rows", [])
    all_videos = []
    for row in rows:
        modules = row.get("modules", [])
        for module in modules:
            module_type = module.get("type")
            if module_type in ("VIDEO_PLAYLIST", "VIDEO_ROW", "PAGED_LIST"):
                paged_list = module.get("pagedList", {})
                if paged_list:
                    items = paged_list.get("items", [])
                    for item in items:
                        video = item.get("item", item) if isinstance(item, dict) else item
                        all_videos.append(video)
            elif module_type == "VIDEO" or (module_type and "video" in module_type.lower()):
                item = module.get("item", module)
                if isinstance(item, dict):
                    all_videos.append(item)

    paginated = all_videos[offset:offset + limit]

    response = {
        "version": API_VERSION,
        "videos": paginated,
        "total": len(all_videos),
    }
    return response

@app.get("/video/")
async def get_video(
    id: int = Query(..., description="Video ID"),
    quality: str = Query(default="HIGH", description="Video quality (HIGH, MEDIUM, LOW)"),
    mode: str = Query(default="STREAM", description="Playback mode (STREAM, OFFLINE)"),
    presentation: str = Query(default="FULL", description="Asset presentation (FULL, PREVIEW)"),
):
    """Fetch video playback info from Tidal."""
    url = f"https://api.tidal.com/v1/videos/{id}/playbackinfo"
    params = {
        "videoquality": quality,
        "playbackmode": mode,
        "assetpresentation": presentation,
    }

    async def fetch_video(cred: dict):
        token, cred = await get_tidal_token_for_cred(cred=cred)
        data, token, cred = await authed_get_json(
            url,
            params=params,
            token=token,
            cred=cred,
            catalog=False,
        )
        return {"version": API_VERSION, "video": data}

    return await dispatch_playback_request(
        fetch_video,
        key=("video", id, quality, mode, presentation),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
