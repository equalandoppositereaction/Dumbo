import argparse
import os

import numpy as np
from itertools import islice
from datasets import load_dataset

from tiny_tokenizer import Tokenizer


def _batched(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


def stream_encode(dataset, tok: Tokenizer, outfile: str, bos: bool, eos: bool, limit: int | None = None, flush_every: int = 1024, log_every: int = 1000, threads: int = 32, batch_size: int = 256):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    buffer: list[np.ndarray] = []

    def encode_batch(texts: list[str]) -> list[np.ndarray]:
        # Use sentencepiece's built-in batch/num_threads when available; fall back quietly.
        try:
            encoded = tok.sp_model.encode(texts, out_type=int, num_threads=threads)
        except TypeError:
            encoded = [tok.sp_model.encode(t, out_type=int) for t in texts]
        out = []
        for ids in encoded:
            if bos:
                ids = [tok.bos_id] + ids
            if eos:
                ids = ids + [tok.eos_id]
            out.append(np.asarray(ids, dtype=np.int32))
        return out

    with open(outfile, "wb") as f:
        processed = 0
        for batch in _batched(dataset, batch_size):
            # Respect limit before scheduling work.
            if limit is not None and processed >= limit:
                break

            remaining = None if limit is None else max(0, limit - processed)
            batch = batch if remaining is None else batch[:remaining]

            texts = [row["text"] for row in batch]
            for ids in encode_batch(texts):
                buffer.append(ids)
                processed += 1

                if len(buffer) >= flush_every:
                    np.concatenate(buffer).tofile(f)
                    buffer.clear()

                if log_every and processed % log_every == 0:
                    print(f"processed {processed} rows")

        if buffer:
            np.concatenate(buffer).tofile(f)
            if log_every:
                print(f"processed {processed} rows (final flush)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="tinytokenized.bin", help="Output binary file (int32 tokens)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows for smoke tests")
    parser.add_argument("--flush-every", type=int, default=1024, help="Flush buffer every N rows")
    parser.add_argument("--log-every", type=int, default=1000, help="Print progress every N rows (0 to disable)")
    parser.add_argument("--threads", type=int, default=32, help="Thread workers for tokenization")
    parser.add_argument("--batch-size", type=int, default=256, help="Rows per batch before threading")
    parser.add_argument("--bos", action="store_true", help="Prepend BOS")
    parser.add_argument("--no-bos", dest="bos", action="store_false")
    parser.add_argument("--eos", action="store_true", help="Append EOS")
    parser.add_argument("--no-eos", dest="eos", action="store_false")
    parser.set_defaults(bos=True, eos=True)

    args = parser.parse_args()

    tok = Tokenizer()

    ds = load_dataset(
    "roneneldan/TinyStories",
    split="train"
)

    stream_encode(
        dataset=ds,
        tok=tok,
        outfile=args.out,
        bos=args.bos,
        eos=args.eos,
        limit=args.limit,
        flush_every=args.flush_every,
        log_every=args.log_every,
        threads=args.threads,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
