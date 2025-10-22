# tools/crawl.py
import csv, sys, time, json, urllib.parse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import trafilatura
from pdfminer.high_level import extract_text as pdf_extract_text
from urllib import robotparser
from slugify import slugify

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR = Path("data/corpus"); CORPUS_DIR.mkdir(parents=True, exist_ok=True)
META_P = Path("data/meta.jsonl")
UA = "GroundKGFetcher/1.0 (Educational/Research; +https://github.com/groundkg)"
SKIP_ROBOTS_DOMAINS = {"wikipedia.org", "arxiv.org"}  # Trusted domains where we skip robots.txt
TIMEOUT = 20
MAX_BYTES = 25_000_000

def sha256_bytes(b: bytes)->str:
    import hashlib; h = hashlib.sha256(); h.update(b); return h.hexdigest()

def allowed_by_robots(url: str)->bool:
    try:
        parsed = urllib.parse.urlparse(url)
        # Skip robots.txt check for trusted educational/research domains
        if any(domain in parsed.netloc for domain in SKIP_ROBOTS_DOMAINS):
            return True
        base = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(base)
        rp.read()
        allowed = rp.can_fetch(UA, url)
        if not allowed:
            print(f"[SKIP robots] {url}", file=sys.stderr)
        return allowed
    except Exception:
        # If robots.txt doesn't exist or errors, allow by default
        return True

def session():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.headers["User-Agent"] = UA
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def filename_for(doc_id: str, ctype: str)->Path:
    ext = ".html"
    if "pdf" in (ctype or "").lower(): ext = ".pdf"
    return RAW_DIR / f"{slugify(doc_id)}{ext}"

def fetch_and_snapshot(doc_id: str, url: str):
    if not allowed_by_robots(url):
        print(f"[SKIP robots] {url}", file=sys.stderr); return None
    s = session()
    try:
        h = s.head(url, timeout=TIMEOUT, allow_redirects=True)
        ctype = (h.headers.get("Content-Type","").split(";")[0] or "").lower()
        clen = int(h.headers.get("Content-Length","0") or 0)
        if clen and clen > MAX_BYTES:
            print(f"[SKIP size] {url} ({clen} bytes)", file=sys.stderr); return None
    except Exception:
        ctype = ""
    r = s.get(url, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    data = r.content
    if len(data) > MAX_BYTES:
        print(f"[SKIP size] {url} ({len(data)} bytes)", file=sys.stderr); return None
    if not ctype:
        ctype = (r.headers.get("Content-Type","").split(";")[0] or "").lower()
    out = filename_for(doc_id, ctype)
    out.write_bytes(data)
    return out, ctype, data

def html_to_text(data: bytes, url: str):
    html = data.decode("utf-8", errors="ignore")
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string: title = soup.title.string.strip()
    except Exception: pass
    extracted = trafilatura.extract(html, include_tables=True, url=url) or ""
    text = extracted.strip() if extracted else (soup.get_text(" ", strip=True) if 'soup' in locals() else "")
    return title or url, text

def pdf_to_text(data: bytes)->str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        try:
            txt = pdf_extract_text(tmp.name) or ""
        except Exception:
            txt = ""
    return txt.strip()

def save_text(doc_id: str, text: str)->Path:
    from slugify import slugify
    p = CORPUS_DIR / f"{slugify(doc_id)}.txt"
    p.write_text(text, encoding="utf-8")
    return p

def main():
    seed = Path("data/seed.csv")
    if not seed.exists():
        print("Missing data/seed.csv", file=sys.stderr); sys.exit(2)
    META_P.unlink(missing_ok=True)
    with seed.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            doc_id = row["doc_id"].strip()
            url = row["url"].strip()
            license_ = (row.get("license") or "UNKNOWN").strip()
            lang = (row.get("lang") or "en").strip()
            try:
                res = fetch_and_snapshot(doc_id, url)
                if not res: continue
                raw_path, ctype, data = res
                sha = sha256_bytes(data)
                if "pdf" in (ctype or ""):
                    title = doc_id
                    text = pdf_to_text(data)
                else:
                    title, text = html_to_text(data, url)
                if not text or len(text) < 200:
                    print(f"[WARN short] {doc_id} {url}", file=sys.stderr)
                txt_path = save_text(doc_id, text)
                meta = {
                    "doc_id": doc_id, "title": title[:300], "url": url, "license": license_, "lang": lang,
                    "raw_path": str(raw_path), "text_path": str(txt_path), "sha256_raw": sha, "bytes_raw": len(data)
                }
                with META_P.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                print(f"[OK] {doc_id} -> {txt_path}")
                time.sleep(0.2)
            except Exception as e:
                print(f"[ERR] {doc_id} {url} :: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

