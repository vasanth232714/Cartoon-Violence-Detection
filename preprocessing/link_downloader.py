"""Download videos from supported public video sites and direct media URLs."""
from __future__ import annotations

import logging
import ipaddress
import os
import re
import socket
import subprocess
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

from config.settings import MAX_UPLOAD_MB, UPLOAD_DIR, ensure_ffmpeg
from preprocessing.video_loader import safe_filename

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_HOSTS = (
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "reddit.com", "redd.it",
    "x.com", "twitter.com", "tiktok.com", "instagram.com", "facebook.com", "fb.watch",
    "streamable.com", "bilibili.com", "b23.tv", "drive.google.com", "docs.google.com",
    "dropbox.com", "dropboxusercontent.com",
)
DIRECT_VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".ts", ".m3u8")


def _public_host(hostname: str, port: int) -> bool:
    """Reject loopback, private, link-local, multicast, and reserved destinations."""
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addresses = list({ipaddress.ip_address(row[4][0]) for row in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)})
        except (OSError, ValueError):
            return False
    return bool(addresses) and all(address.is_global for address in addresses)


def _validate_public_url(
    url: str,
    require_supported_host: bool = False,
    *,
    resolve_host: bool = True,
) -> bool:
    """Validate an HTTP(S) URL and, when requested, resolve it to a public host.

    Form validation should not depend on a live DNS lookup. Video-site URLs are
    still restricted to the allowlist here, while the stricter DNS check remains
    enabled for HTTP requests and redirect targets.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port != (443 if parsed.scheme == "https" else 80):
            return False
        host = parsed.hostname.lower().rstrip(".")
        if require_supported_host:
            supported_host = any(host == allowed or host.endswith("." + allowed) for allowed in SUPPORTED_VIDEO_HOSTS)
            direct_file = Path(urllib.parse.unquote(parsed.path)).suffix.lower() in DIRECT_VIDEO_EXTENSIONS
            if not supported_host and not direct_file:
                return False
        if resolve_host and not _public_host(parsed.hostname, port):
            return False
        return True
    except (ValueError, TypeError):
        return False


def _safe_get(session, url: str, *, max_redirects: int = 5, **kwargs):
    """Follow HTTP redirects manually and validate each destination before requesting it."""
    current = url
    request_kwargs = dict(kwargs)
    for _ in range(max_redirects + 1):
        if not _validate_public_url(current):
            raise LinkDownloadError("The URL or a redirect points to a non-public or unsupported destination.")
        response = session.get(current, allow_redirects=False, **request_kwargs)
        request_kwargs.pop("params", None)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise LinkDownloadError("The video host returned an invalid redirect.")
            current = urllib.parse.urljoin(current, location)
            continue
        return response
    raise LinkDownloadError("The video URL redirected too many times.")


class LinkDownloadError(Exception):
    """Raised when downloading a video link fails."""


def extract_google_drive_id(url: str) -> str | None:
    """Extract Google Drive file ID from various sharing URL formats."""
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]{20,})",
        r"drive\.google\.com/open\?(?:.*&)?id=([a-zA-Z0-9_-]{20,})",
        r"drive\.google\.com/uc\?(?:.*&)?id=([a-zA-Z0-9_-]{20,})",
        r"docs\.google\.com/file/d/([a-zA-Z0-9_-]{20,})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def sanitize_url(raw_url: str) -> str:
    """Cleans up pasted URLs, normalizes YouTube/Shorts/Dropbox/Google Drive and strips tracking clutter."""
    if not raw_url:
        return ""
    # Browsers and chat clients can add invisible Unicode marks around a URL.
    url = re.sub(r"[\u200b-\u200d\ufeff]", "", raw_url)
    url = url.strip().strip("'\"<>[]() \t\n\r")

    # Handle mobile youtube URL
    if "m.youtube.com" in url:
        url = url.replace("m.youtube.com", "www.youtube.com")

    # Normalize YouTube shorts URL to standard watch URL for better extractor compatibility
    shorts_match = re.search(r"youtube\.com/shorts/([a-zA-Z0-9_-]+)", url)
    if shorts_match:
        video_id = shorts_match.group(1)
        url = f"https://www.youtube.com/watch?v={video_id}"

    # Normalize youtu.be shortlinks
    youtu_match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
    if youtu_match:
        video_id = youtu_match.group(1)
        url = f"https://www.youtube.com/watch?v={video_id}"

    # If it's a YouTube watch URL with playlist parameters, strip them to avoid playlist extraction
    if "youtube.com/watch" in url:
        parsed = urllib.parse.urlparse(url)
        q_params = urllib.parse.parse_qs(parsed.query)
        if "v" in q_params and q_params["v"]:
            clean_query = urllib.parse.urlencode({"v": q_params["v"][0]})
            url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", clean_query, ""))

    # Normalize Dropbox share link for direct downloading
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
        elif "&dl=0" in url:
            url = url.replace("&dl=0", "&dl=1")
        elif "?dl=" not in url and "&dl=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}dl=1"

    return url


def is_valid_url(url: str) -> bool:
    """Accept supported public video hosts and direct video files; reject internal targets."""
    if not url or not isinstance(url, str):
        return False
    u = sanitize_url(url)
    try:
        parsed = urllib.parse.urlparse(u)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    supported_video_host = any(host == allowed or host.endswith("." + allowed) for allowed in SUPPORTED_VIDEO_HOSTS)
    # Do not make the upload form depend on DNS availability for known video
    # sites. Direct file URLs still require a public DNS/IP result here, and
    # every HTTP request and redirect is checked again before fetching.
    return _validate_public_url(
        u,
        require_supported_host=True,
        resolve_host=not supported_video_host,
    )


def _download_google_drive(
    file_id: str,
    target_path: Path,
    max_mb: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> bool:
    """Directly stream and download a shared Google Drive video file."""
    try:
        import requests

        session = requests.Session()
        url = "https://drive.google.com/uc"
        res = _safe_get(session, url, params={"id": file_id, "export": "download"},
                        stream=True, timeout=30)

        # Handle virus-scan warning token for large files (>100MB)
        for k, v in res.cookies.items():
            if k.startswith("download_warning"):
                res.close()
                res = _safe_get(session, url, params={"id": file_id, "confirm": v, "export": "download"},
                                stream=True, timeout=30)
                break

        # Check if Google returned an HTML page with confirmation token
        ct = res.headers.get("content-type", "").lower()
        if "text/html" in ct:
            content_sample = res.content[:8192].decode("utf-8", errors="ignore")
            token_match = re.search(r"confirm=([0-9A-Za-z_]+)", content_sample)
            if token_match:
                confirm_token = token_match.group(1)
                res.close()
                res = _safe_get(session, url, params={"id": file_id, "confirm": confirm_token, "export": "download"},
                                stream=True, timeout=30)

        if "text/html" in res.headers.get("content-type", "").lower():
            raise LinkDownloadError("Google Drive returned a confirmation page instead of a video file.")
        res.raise_for_status()

        total = int(res.headers.get("content-length", 0))
        if total > max_mb * 1024 * 1024:
            raise LinkDownloadError(f"Video file exceeds maximum allowed size of {max_mb} MB.")

        downloaded = 0
        with open(target_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > max_mb * 1024 * 1024:
                    raise LinkDownloadError(f"Video exceeds maximum allowed size of {max_mb} MB.")
                if progress_callback and total > 0:
                    frac = downloaded / total
                    progress_callback(min(max(frac, 0.0), 0.99), f"Downloading stream... {int(frac * 100)}%")

        return target_path.is_file() and target_path.stat().st_size > 1024
    except Exception as exc:
        logger.warning("Google Drive download error: %s", exc)
        if target_path.is_file():
            target_path.unlink(missing_ok=True)
        return False
    finally:
        if "res" in locals():
            res.close()
        if "session" in locals():
            session.close()


def _download_direct_http(
    url: str,
    target_path: Path,
    max_mb: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> bool:
    """Direct streaming HTTP download for raw media endpoints, cloud storages, or CDNs."""
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        with requests.Session() as session:
            with _safe_get(session, url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                if total > max_mb * 1024 * 1024:
                    raise LinkDownloadError(f"Video file exceeds maximum allowed size of {max_mb} MB.")

                downloaded = 0
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > max_mb * 1024 * 1024:
                            raise LinkDownloadError(f"Video exceeds maximum allowed size of {max_mb} MB.")
                        if progress_callback and total > 0:
                            frac = downloaded / total
                            progress_callback(min(max(frac, 0.0), 0.99), f"Downloading stream... {int(frac * 100)}%")

        return target_path.is_file() and target_path.stat().st_size > 1024
    except Exception as exc:
        logger.warning("Direct HTTP download failed: %s", exc)
        if target_path.is_file():
            target_path.unlink(missing_ok=True)
        return False


def _is_direct_video_endpoint(url: str) -> bool:
    """Only treat self-contained media files as direct streams; manifests use yt-dlp."""
    parsed = urllib.parse.urlparse(url)
    suffix = Path(urllib.parse.unquote(parsed.path)).suffix.lower()
    return suffix in DIRECT_VIDEO_EXTENSIONS and suffix != ".m3u8"


def _remux_to_mp4(input_path: Path, ffmpeg_exe: str | None) -> Path:
    """Remuxes or re-encodes any non-MP4 container (e.g. .webm, .mkv, .ts) to standard MP4."""
    if not ffmpeg_exe or not os.path.isfile(ffmpeg_exe):
        return input_path
    if input_path.suffix.lower() == ".mp4":
        return input_path

    mp4_target = input_path.with_suffix(".mp4")
    try:
        # Fast lossless stream copy first
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel", "error",
            "-i", str(input_path),
            "-c:v", "copy",
            "-c:a", "aac",
            str(mp4_target),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if res.returncode == 0 and mp4_target.is_file() and mp4_target.stat().st_size > 0:
            input_path.unlink(missing_ok=True)
            return mp4_target

        # If lossless copy failed (e.g. VP9/AV1 codec incompatible with standard MP4), re-encode with libx264
        cmd_reencode = [
            ffmpeg_exe,
            "-y",
            "-loglevel", "error",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            str(mp4_target),
        ]
        res2 = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if res2.returncode == 0 and mp4_target.is_file() and mp4_target.stat().st_size > 0:
            input_path.unlink(missing_ok=True)
            return mp4_target
    except Exception as exc:
        logger.warning("FFmpeg remuxing error: %s", exc)

    return input_path


def download_video_from_url(
    url: str,
    output_dir: Path | str = UPLOAD_DIR,
    max_mb: int = MAX_UPLOAD_MB,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, str]:
    """Download a video from YouTube, social platforms, cloud storage, or direct links."""
    clean_url = sanitize_url(url)
    if not is_valid_url(clean_url):
        raise LinkDownloadError("Please provide a valid web URL starting with http:// or https://")

    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    tmpl = str(out_path_dir / f"stream_{uid}_%(title).60s.%(ext)s")
    ffmpeg_exe = ensure_ffmpeg()

    # Special Case 1: Google Drive shared link
    gdrive_id = extract_google_drive_id(clean_url)
    if gdrive_id:
        if progress_callback:
            progress_callback(0.05, "Connecting to Google Drive media stream...")
        direct_target = out_path_dir / f"stream_{uid}_gdrive.mp4"
        if _download_google_drive(gdrive_id, direct_target, max_mb, progress_callback):
            target_mp4 = _remux_to_mp4(direct_target, ffmpeg_exe)
            clean_title = safe_filename(f"gdrive_video_{uid}.mp4")
            return target_mp4, clean_title

    # Special Case 2: Dropbox or direct video stream endpoint
    if "dropbox.com" in clean_url or _is_direct_video_endpoint(clean_url):
        if progress_callback:
            progress_callback(0.05, "Connecting to media stream endpoint...")
        direct_target = out_path_dir / f"stream_{uid}_direct.mp4"
        if _download_direct_http(clean_url, direct_target, max_mb, progress_callback):
            target_mp4 = _remux_to_mp4(direct_target, ffmpeg_exe)
            clean_title = safe_filename(f"direct_stream_{uid}.mp4")
            return target_mp4, clean_title
        raise LinkDownloadError("Could not safely download this direct media URL. Try uploading the video file instead.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise LinkDownloadError("Install the project dependencies to download video-site links (yt-dlp is missing).") from exc

    def _hook(d):
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            frac = (downloaded / total) if total > 0 else 0.0
            spd = d.get("_speed_str", "")
            eta = d.get("_eta_str", "")
            msg = f"Downloading stream... {d.get('_percent_str', '')} ({spd}, ETA: {eta})".strip()
            progress_callback(min(max(frac, 0.0), 0.99), msg)

    # Base configuration for yt-dlp
    base_ydl_opts = {
        "outtmpl": tmpl,
        "max_filesize": max_mb * 1024 * 1024,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": False,
        "geo_bypass": True,
        "progress_hooks": [_hook],
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if ffmpeg_exe and Path(ffmpeg_exe).is_file():
        base_ydl_opts["ffmpeg_location"] = str(Path(ffmpeg_exe).parent)
        base_ydl_opts["format"] = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best"
        base_ydl_opts["merge_output_format"] = "mp4"
    else:
        base_ydl_opts["format"] = "best[ext=mp4]/best"

    last_error: Exception | None = None
    info = None

    # Tier 1: Modern Multi-Client with full signature decryption enabled
    try:
        if progress_callback:
            progress_callback(0.05, "Contacting video host...")
        opts_tier1 = dict(base_ydl_opts)
        opts_tier1["extractor_args"] = {
            "youtube": {
                "player_client": ["ios", "web", "mweb"],
            }
        }
        with yt_dlp.YoutubeDL(opts_tier1) as ydl:
            info = ydl.extract_info(clean_url, download=True)
    except Exception as exc1:
        last_error = exc1
        logger.info("Tier 1 extraction failed: %s. Trying mobile fallback...", exc1)

    # Tier 2: Mobile client fallback (Android / Creator) with single-stream format
    if not info:
        try:
            if progress_callback:
                progress_callback(0.12, "Retrying with mobile client stream...")
            opts_tier2 = dict(base_ydl_opts)
            opts_tier2["format"] = "best[ext=mp4]/best"
            opts_tier2.pop("merge_output_format", None)
            opts_tier2["extractor_args"] = {
                "youtube": {
                    "player_client": ["android", "mweb"],
                }
            }
            with yt_dlp.YoutubeDL(opts_tier2) as ydl:
                info = ydl.extract_info(clean_url, download=True)
        except Exception as exc2:
            last_error = exc2
            logger.info("Tier 2 extraction failed: %s", exc2)

    # Tier 3: Generic extraction (for Vimeo, Reddit, Twitter, TikTok, Dailymotion, etc.)
    if not info:
        try:
            if progress_callback:
                progress_callback(0.18, "Retrying with generic extractor...")
            opts_tier3 = dict(base_ydl_opts)
            opts_tier3.pop("extractor_args", None)
            opts_tier3["format"] = "best"
            with yt_dlp.YoutubeDL(opts_tier3) as ydl:
                info = ydl.extract_info(clean_url, download=True)
        except Exception as exc3:
            last_error = exc3
            logger.info("Tier 3 extraction failed: %s", exc3)

    # Tier 4: Direct streaming HTTP fallback if all yt-dlp extractors failed
    if not info:
        if progress_callback:
            progress_callback(0.22, "Attempting direct HTTP media stream...")
        direct_target = out_path_dir / f"stream_{uid}_fallback.mp4"
        if _download_direct_http(clean_url, direct_target, max_mb, progress_callback):
            target_mp4 = _remux_to_mp4(direct_target, ffmpeg_exe)
            clean_title = safe_filename(f"stream_{uid}.mp4")
            return target_mp4, clean_title

    if not info:
        err_msg = str(last_error or "Could not extract video stream from this link.")
        if "max_filesize" in err_msg.lower():
            raise LinkDownloadError(f"Video exceeds maximum allowed file size of {max_mb} MB.")
        if "private" in err_msg.lower() or "sign in" in err_msg.lower():
            raise LinkDownloadError("This video is private, age-restricted, or requires login credentials.")
        raise LinkDownloadError(f"Download failed: {err_msg.split('ERROR:')[-1].strip()}")

    title = info.get("title", f"web_video_{uid}")

    candidates = [
        f for f in out_path_dir.glob(f"stream_{uid}_*")
        if not f.name.endswith(".part") and not f.name.endswith(".ytdl") and not f.name.endswith(".temp")
    ]

    if not candidates:
        raise LinkDownloadError("Video was downloaded but the output file could not be located on disk.")

    # Prioritize MP4 container
    mp4_files = [f for f in candidates if f.suffix.lower() == ".mp4"]
    target_path = mp4_files[0] if mp4_files else candidates[0]

    # Convert non-MP4 formats (WebM, MKV, etc.) to standard MP4
    if target_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
        if progress_callback:
            progress_callback(0.95, "Converting container to MP4 format...")
        target_path = _remux_to_mp4(target_path, ffmpeg_exe)

    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise LinkDownloadError("Downloaded video file is empty or corrupted.")

    if progress_callback:
        progress_callback(1.0, "Download completed.")

    clean_title = safe_filename(f"{title}{target_path.suffix}")
    return target_path, clean_title
