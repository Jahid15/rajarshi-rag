import json
import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

BANGLA_SEPARATORS = ["\n\n", "\n", "। ", "? ", "! ", "; ", ", ", " ", ""]
INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(INVISIBLE)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = text.replace("|", "।")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_chapters() -> list[dict]:
    return json.loads(config.BOOK_JSON.read_text(encoding="utf-8"))


def make_chunks(size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=BANGLA_SEPARATORS,
        keep_separator="end",
        length_function=len,
    )
    docs = []
    for ch in load_chapters():
        text = normalize(ch["text"])
        for j, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            docs.append(Document(
                page_content=piece,
                metadata={
                    "chunk_id": f"{ch['order']:02d}-{j:03d}",
                    "chapter_no": ch["order"],
                    "chapter": ch["chapter"],
                    "pages": ch["pages"],
                    "url": ch["url"],
                },
            ))
    return docs


if __name__ == "__main__":
    import statistics
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = make_chunks()
    lengths = [len(d.page_content) for d in chunks]
    print(f"{len(chunks)} chunks, mean {statistics.mean(lengths):.0f} chars, max {max(lengths)}")
    print(chunks[0].metadata)
    print(chunks[0].page_content)
