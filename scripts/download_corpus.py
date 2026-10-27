"""Descarga un corpus grande de libros en español desde Project Gutenberg.

Usa la API de Gutendex para listar libros y descarga su texto plano,
quitando las cabeceras/pies de licencia de Project Gutenberg y
separando cada libro con <|endoftext|>.

Uso:
    uv run python scripts/download_corpus.py
    uv run python scripts/download_corpus.py --target-chars 25_000_000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://gutendex.com/books?languages=es&page={page}"
USER_AGENT = "llms-from-scratch/0.1 (corpus builder)"
SEPARATOR = "\n<|endoftext|>\n"

START_RE = re.compile(
    r"^\s*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
END_RE = re.compile(
    r"^\s*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

SKIP_TITLE = (
    "diccionario",
    "gramática",
    "gramatica",
    "vocabulario",
    "readme",
    "índice",
    "indice",
)

SPANISH_WORDS = {
    "de", "la", "que", "el", "en", "y", "los", "se", "del", "las", "un", "por",
    "con", "una", "para", "es", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin",
    "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo",
    "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
    "eso", "ante", "ellos", "esto", "antes", "algunos", "qué", "unos", "yo",
    "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "fue", "era", "son", "han", "ser", "tiene", "muy",
}  # fmt: skip

ENGLISH_WORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "it", "for", "was",
    "as", "with", "be", "by", "on", "not", "he", "i", "this", "are", "or",
    "his", "from", "at", "which", "but", "have", "an", "had", "they", "you",
    "were", "their", "one", "all", "we", "can", "her", "has", "there", "been",
    "if", "more", "when", "will", "would", "who", "so", "no", "she", "other",
    "its", "may", "what", "up", "out", "them", "than", "into", "only", "some",
    "could", "time", "these", "two", "then", "do", "first", "any", "my", "now",
    "such", "like", "our", "over", "man", "me", "even", "most", "made", "after",
    "also", "did", "many", "before", "must", "through", "back", "years",
    "where", "much", "your", "way", "well", "down", "should", "because",
    "each", "just", "those", "people", "how", "too", "little", "state", "good",
    "very", "make", "world", "still", "own", "see", "men", "work", "long",
    "get", "here", "between", "both", "life", "being", "under", "never", "day",
    "same", "another", "know", "while", "last", "might", "us", "great", "old",
    "year", "off", "come", "since", "against", "go", "came", "right", "used",
    "take", "three",
}  # fmt: skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga un corpus de Gutenberg")
    parser.add_argument("--out", type=Path, default=Path("data/corpus_es.txt"))
    parser.add_argument(
        "--target-chars",
        type=int,
        default=25_000_000,
        help="Tamaño objetivo del corpus en caracteres",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=250,
        help="Máximo de libros candidatos a descargar",
    )
    parser.add_argument("--min-chars", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Muestra el progreso cada N libros descargados",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Muestra una línea por cada libro (OK/ERROR)",
    )
    return parser.parse_args()


def http_get(url: str, timeout: float = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_books(max_books: int, page_size: int) -> list[dict]:
    books: list[dict] = []
    page = 1

    while len(books) < max_books:
        url = API_URL.format(page=page)
        payload = None
        for attempt in range(3):
            try:
                payload = json.loads(http_get(url, timeout=45).decode("utf-8"))
                break
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                if attempt == 2:
                    raise SystemExit(
                        f"No se pudo consultar Gutendex: {exc}"
                    ) from exc
                time.sleep(2 * (attempt + 1))

        results = payload.get("results", [])
        if not results:
            break

        books.extend(results)
        print(f"  página {page}: {len(books)} candidatos", flush=True)

        if payload.get("next") is None:
            break
        page += 1

    return books[:max_books]


def pick_text_url(formats: dict[str, str]) -> str | None:
    candidates = [
        url
        for mime, url in formats.items()
        if mime.startswith("text/plain") and "readme" not in url.lower()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda url: "utf-8" not in url.lower())
    return candidates[0]


def decode(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def download_book(
    book: dict,
) -> tuple[dict, str | None, int, str | None]:
    """Devuelve (libro, texto, bytes, error)."""
    url = pick_text_url(book.get("formats", {}))
    if url is None:
        return book, None, 0, "sin formato de texto plano"

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = http_get(url)
            return book, decode(raw), len(raw), None
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    return book, None, 0, str(last_error) or "error desconocido"


def strip_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start = START_RE.search(text)
    if start is not None:
        text = text[start.end() :]

    end = END_RE.search(text)
    if end is not None:
        text = text[: end.start()]

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_spanish(text: str) -> bool:
    sample = text[:100_000].lower()
    words = re.findall(r"[a-záéíóúüñ]+", sample)

    spanish = sum(word in SPANISH_WORDS for word in words)
    english = sum(word in ENGLISH_WORDS for word in words)

    if spanish + english == 0:
        return False
    return spanish > english * 1.2


def main() -> None:
    args = parse_args()

    print(f"Buscando hasta {args.max_books} libros en español...")
    books = fetch_books(args.max_books, args.page_size)
    if not books:
        raise SystemExit("Gutendex no devolvió ningún libro.")

    total = len(books)
    print(f"Candidatos: {total}")
    print(f"Descargando con {args.workers} workers...\n")

    downloaded: list[tuple[dict, str]] = []
    failures = 0
    downloaded_bytes = 0
    report_every = max(args.progress_every, 25) if args.verbose else args.progress_every
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_book, book) for book in books]
        for index, future in enumerate(as_completed(futures), start=1):
            book: dict = {}
            text: str | None
            size: int
            error: str | None

            try:
                book, text, size, error = future.result()
            except Exception as exc:  # noqa: BLE001 - una descarga no debe romper todo
                text, size, error = None, 0, f"inesperado: {exc}"

            title = str(book.get("title", "?"))[:55]
            book_id = book.get("id", "?")

            if text is None:
                failures += 1
                print(
                    f"[{index:>4}/{total}] ERROR id={book_id} | {title} | {error}",
                    flush=True,
                )
            else:
                downloaded.append((book, text))
                downloaded_bytes += size
                if args.verbose:
                    print(
                        f"[{index:>4}/{total}] OK    id={book_id:>6} | "
                        f"{title:<55} | {size / 1_000_000:6.2f} MB",
                        flush=True,
                    )

            if index % report_every == 0 or index == total:
                elapsed = time.perf_counter() - started
                rate = downloaded_bytes / elapsed / 1_000_000 if elapsed else 0.0
                eta = (elapsed / index) * (total - index) if index else 0.0
                line = (
                    f"Progreso {index}/{total} ({100 * index / total:5.1f}%) | "
                    f"{downloaded_bytes / 1_000_000:7.1f} MB | "
                    f"{rate:5.2f} MB/s | fallos {failures} | "
                    f"ETA {eta:4.0f}s | último: {title}"
                )
                if sys.stdout.isatty() and not args.verbose:
                    print("\r" + line[:160].ljust(160), end="", flush=True)
                else:
                    print(line, flush=True)

    if sys.stdout.isatty() and not args.verbose:
        print()

    print(f"\nDescargados: {len(downloaded)} | fallos: {failures}")

    kept: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    skipped_short = skipped_title = skipped_lang = skipped_dup = 0

    for book, raw in sorted(downloaded, key=lambda item: item[0]["id"]):
        title = book.get("title", "").lower()
        if any(keyword in title for keyword in SKIP_TITLE):
            skipped_title += 1
            continue

        text = strip_gutenberg(raw)
        if len(text) < args.min_chars:
            skipped_short += 1
            continue

        if not looks_spanish(text):
            skipped_lang += 1
            continue

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            skipped_dup += 1
            continue
        seen.add(digest)

        kept.append(text)
        total_chars += len(text)

        if total_chars >= args.target_chars:
            break

    if not kept:
        raise SystemExit("No se pudo descargar ningún libro válido.")

    corpus = SEPARATOR.join(kept).strip() + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp_path.write_text(corpus, encoding="utf-8")
    tmp_path.replace(args.out)

    print()
    print(f"Libros incluidos: {len(kept)}")
    print(f"Descartados: cortos={skipped_short} título={skipped_title} "
          f"idioma={skipped_lang} duplicados={skipped_dup}")
    print(f"Caracteres: {len(corpus):,}")

    try:
        import tiktoken

        tokens = len(
            tiktoken.get_encoding("gpt2").encode(
                corpus, allowed_special={"<|endoftext|>"}
            )
        )
        print(f"Tokens GPT-2: {tokens:,}")
    except ImportError:
        print(f"Tokens GPT-2 aprox.: {len(corpus) // 4:,}")

    print(f"Escrito en: {args.out}")


if __name__ == "__main__":
    main()
