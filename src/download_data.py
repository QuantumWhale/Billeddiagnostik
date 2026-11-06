"""
Henter CRIC-datasættet:

Kilde: Rezende et al. (2021) "Cric searchable image database as a public platform for conventional pap smear cytologic data", Scientific Data.
10.6084/m9.figshare.c.4960286.v2

Scriptet her: 
1. Lister alle billeder i den offentlige Figshare-samling via Figshare v2 REST API'et
2. Downloader alle de tilhørende filer til data/raw/
3. Skriver en metadata.csv, der kobler artikel-ID -> titel -> lokale filnavn(e),
så vi kan spore hver eneste downloadede fil tilbage til den post, den kom fra.

"""


 
import argparse
import csv
import sys
import time
from pathlib import Path
 
import requests
from tqdm import tqdm
 
FIGSHARE_API = "https://api.figshare.com/v2"
COLLECTION_ID = 4960286  # CRIC Cervix Cell Classification samling
PAGE_SIZE = 50
REQUEST_PAUSE_SEC = 0.2  # undgå at spamme API'en
 
 
def list_collection_articles(collection_id: int):
    """Giver hver article (dictionary med minimum 'id' og 'title') i samlingen, paginated."""
    page = 1
    while True:
        resp = requests.get(
            f"{FIGSHARE_API}/collections/{collection_id}/articles",
            params={"page": page, "page_size": PAGE_SIZE},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for article in batch:
            yield article
        page += 1
        time.sleep(REQUEST_PAUSE_SEC)
 
 
def get_article_files(article_id: int):
    """Returnerer listen af filer, der kan downloades, for et givet artikel-id."""
    resp = requests.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("files", [])
 
 
def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return  # er allerede hentet, så den springer vi over
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
 
 
def main():
    parser = argparse.ArgumentParser(description="Download CRIC Cervix dataset from Figshare.")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N articles (for testing)")
    args = parser.parse_args()
 
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"
 
    print("Fetching article list from Figshare collection...")
    articles = list(list_collection_articles(COLLECTION_ID))
    if args.limit:
        articles = articles[: args.limit]
    print(f"Found {len(articles)} articles in the collection.")
 
    rows = []
    for article in tqdm(articles, desc="Downloading"):
        article_id = article["id"]
        title = article.get("title", "")
        try:
            files = get_article_files(article_id)
        except requests.RequestException as e:
            print(f"  [warn] failed to fetch files for article {article_id}: {e}", file=sys.stderr)
            continue
 
        for file_info in files:
            fname = file_info["name"]
            url = file_info["download_url"]
            dest = out_dir / fname
            try:
                download_file(url, dest)
            except requests.RequestException as e:
                print(f"  [warn] failed to download {fname} (article {article_id}): {e}", file=sys.stderr)
                continue
            rows.append({"article_id": article_id, "title": title, "filename": fname})
        time.sleep(REQUEST_PAUSE_SEC)
 
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["article_id", "title", "filename"])
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"Done. {len(rows)} files saved to {out_dir}/")
    print(f"Metadata index written to {metadata_path}")
 
 
if __name__ == "__main__":
    main()
    