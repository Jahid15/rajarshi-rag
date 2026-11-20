"""Crawl every chapter subpage of the book from Bengali Wikisource and save clean text."""

import argparse
import json
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

import config

session = requests.Session()
session.headers["User-Agent"] = config.USER_AGENT


def api_parse(title: str) -> str:
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2}
    for attempt in range(8):
        r = session.get(config.API_URL, params=params, timeout=60)
        if r.status_code == 429:
            wait = int(r.headers.get("retry-after", "15")) + 2
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"{title}: {data['error']['info']}")
        return data["parse"]["text"]
    raise RuntimeError(f"{title}: too many retries")


def chapter_links(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    prefix = config.BOOK_TITLE + "/"
    seen, titles = set(), []
    for a in soup.find_all("a", title=True):
        t = a["title"]
        if t.startswith(prefix) and t not in seen:
            seen.add(t)
            titles.append(t)
    return titles


def clean_chapter(html: str, chapter: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    pages_tag = soup.select_one("span.ws-pages")
    pages = pages_tag.get_text(strip=True) if pages_tag else ""

    for tag in soup.select("style, script, .ws-noexport, .pagenum, sup.reference, .mw-references-wrap, ol.references"):
        tag.decompose()
    for tag in soup.select(".wst-center"):
        if tag.get_text(strip=True) == chapter:
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")

    blocks = []
    for el in soup.select("p, div.poem"):
        if el.name == "p" and el.find_parent("div", class_="poem"):
            continue
        text = el.get_text("").strip()
        if text.strip("\"“” ") and not re.fullmatch(r"\S+ পরিচ্ছেদ", text):
            blocks.append(text)
    return "\n\n".join(blocks), pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cached HTML and download again")
    args = ap.parse_args()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching index page: {config.BOOK_TITLE}")
    titles = chapter_links(api_parse(config.BOOK_TITLE))
    print(f"Found {len(titles)} chapter subpages")
    if not titles:
        sys.exit("No chapter links found on the index page")

    chapters = []
    for i, title in enumerate(titles, start=1):
        cache = config.RAW_DIR / f"{i:02d}.html"
        if cache.exists() and not args.refresh:
            html = cache.read_text(encoding="utf-8")
        else:
            html = api_parse(title)
            cache.write_text(html, encoding="utf-8")
            time.sleep(1.5)
        name = title.split("/", 1)[1]
        text, pages = clean_chapter(html, name)
        chapters.append({
            "order": i,
            "chapter": name,
            "title": title,
            "url": config.page_url(title),
            "pages": pages,
            "text": text,
        })
        print(f"  {i:02d} {name}: {len(text)} chars, book pages {pages}")

    config.BOOK_JSON.write_text(json.dumps(chapters, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(c["text"]) for c in chapters)
    print(f"Saved {len(chapters)} chapters, {total} characters to {config.BOOK_JSON}")


if __name__ == "__main__":
    main()
