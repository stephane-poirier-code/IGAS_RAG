import base64
import math
import os
from pathlib import Path

from mistralai import Mistral, models

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "library"
QUESTION = "Quelles recommandations sont à destination des ARS ?"
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
EMBEDDING_BATCH_SIZE = 32
RETRIEVED_CHUNKS = 5


def chunk_pages(pages: list[models.OCRPageObject]) -> list[tuple[int, str]]:
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP

    for page in pages:
        words = page.markdown.split()
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + CHUNK_WORDS])
            if text:
                chunks.append((page.index + 1, text))
            if start + CHUNK_WORDS >= len(words):
                break

    return chunks


def embed_texts(client: Mistral, texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=texts[start : start + EMBEDDING_BATCH_SIZE],
        )
        vectors.extend(
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index or 0)
            if item.embedding is not None
        )
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def main() -> None:
    pdf_files = sorted(LIBRARY_PATH.glob("*.pdf"))
    if len(pdf_files) != 1:
        raise RuntimeError(
            f"Expected exactly one PDF in {LIBRARY_PATH}, found {len(pdf_files)}."
        )

    pdf_path = pdf_files[0]
    pdf_data = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    document_url = f"data:application/pdf;base64,{pdf_data}"

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    print(f"Extracting text from {pdf_path.name} with Mistral OCR...")
    ocr_result = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": document_url,
            "document_name": pdf_path.name,
        },
    )

    chunks = chunk_pages(ocr_result.pages)
    if not chunks:
        raise RuntimeError(f"No text was extracted from {pdf_path.name}.")

    print(f"Embedding {len(chunks)} text chunks...")
    chunk_vectors = embed_texts(client, [text for _, text in chunks])
    if len(chunk_vectors) != len(chunks):
        raise RuntimeError("The embeddings response did not include every text chunk.")

    query_vector = embed_texts(client, [QUESTION])[0]
    ranked_chunks = sorted(
        (
            (
                cosine_similarity(query_vector, vector),
                page_number,
                text,
            )
            for (page_number, text), vector in zip(chunks, chunk_vectors)
        ),
        reverse=True,
    )[:RETRIEVED_CHUNKS]

    context = "\n\n".join(
        f"[Page {page_number}]\n{text}"
        for _, page_number, text in ranked_chunks
    )
    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[
            {
                "role": "user",
                "content": (
                    "Réponds à la question en français en t'appuyant uniquement "
                    "sur les extraits fournis. Si l'information n'y figure pas, "
                    "dis-le clairement. Cite les numéros de page pertinents.\n\n"
                    f"Question : {QUESTION}\n\n"
                    f"Extraits du document :\n{context}"
                ),
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
