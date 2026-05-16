"""
Downloads all MIDI files from the Mutopia Project FTP index.
Structure: /ftp/Composer/WorkID/PieceName/PieceName-mids.zip
Extracts all zips into data/mutopia/<Composer>/<PieceName>/
"""

import os
import zipfile
import io
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.mutopiaproject.org/ftp/"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "mutopia")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "MutopiaDownloader/1.0 (educational use)"


def list_links(url):
    """Return all href links from an Apache-style directory listing."""
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("?") or href.startswith("/") and href != url:
            continue
        if href in ("../", "/"):
            continue
        links.append(urljoin(url, href))
    return links


def download_and_extract(zip_url, dest_dir):
    resp = SESSION.get(zip_url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)


def crawl():
    os.makedirs(OUT_DIR, exist_ok=True)
    downloaded = 0
    skipped = 0
    errors = 0

    print(f"Fetching composer list from {BASE_URL}")
    composer_urls = [u for u in list_links(BASE_URL) if u.endswith("/") and u != BASE_URL]
    print(f"Found {len(composer_urls)} composers")

    for comp_url in composer_urls:
        composer = comp_url.rstrip("/").split("/")[-1]
        try:
            work_urls = [u for u in list_links(comp_url) if u.endswith("/") and u != comp_url]
        except Exception as e:
            print(f"  [ERROR] listing {comp_url}: {e}")
            errors += 1
            continue

        for work_url in work_urls:
            try:
                piece_urls = [u for u in list_links(work_url) if u.endswith("/") and u != work_url]
            except Exception as e:
                print(f"  [ERROR] listing {work_url}: {e}")
                errors += 1
                continue

            for piece_url in piece_urls:
                piece = piece_url.rstrip("/").split("/")[-1]
                try:
                    file_urls = list_links(piece_url)
                except Exception as e:
                    print(f"  [ERROR] listing {piece_url}: {e}")
                    errors += 1
                    continue

                midi_zips = [u for u in file_urls if u.endswith("-mids.zip")]
                for zip_url in midi_zips:
                    dest = os.path.join(OUT_DIR, composer, piece)
                    marker = os.path.join(dest, ".done")
                    if os.path.exists(marker):
                        skipped += 1
                        continue
                    os.makedirs(dest, exist_ok=True)
                    try:
                        download_and_extract(zip_url, dest)
                        open(marker, "w").close()
                        downloaded += 1
                        print(f"  [OK] {composer}/{piece} ({downloaded} total)")
                    except Exception as e:
                        print(f"  [ERROR] {zip_url}: {e}")
                        errors += 1
                    time.sleep(0.3)  # be polite

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    crawl()
