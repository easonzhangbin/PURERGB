"""Private row scheduling for certified original-pixel detector statistics.

This helper changes no cell arithmetic or result ordering. Uncertified arrays
retain their original serial implementation. Each worker returns only a local
row; no background task writes a detector grid or another request's cache.
"""

from concurrent.futures import ThreadPoolExecutor
import os

from .immutable_source_v44 import certificate_for


MIN_SOURCE_PIXELS = 1_000_000
MAX_WORKERS = 4


def parallel_rows(source, row_count, row_factory):
    """Return original-order rows, or None to keep the public serial route."""
    certificate = certificate_for(source)
    if certificate is None or source.shape[0] * source.shape[1] < MIN_SOURCE_PIXELS:
        return None
    if type(row_count) is not int or row_count < 1 or not callable(row_factory):
        raise ValueError("Parallel cell statistics require complete original rows")
    try:
        workers = min(MAX_WORKERS, max(1, int(os.environ.get("RAYON_NUM_THREADS", "4"))))
    except ValueError:
        return None
    # A subsequent request may replace certificate.checkpoint. Capture this
    # request's token closure before submitting any work, and never reread it.
    checkpoint = certificate.checkpoint

    def check():
        if checkpoint is not None:
            checkpoint()
        certificate.validate(source)

    def compute(y):
        check()
        result = row_factory(y)
        check()
        return result

    check()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meter-cell-row")
    try:
        futures = []
        for y in range(row_count):
            check()
            futures.append(pool.submit(compute, y))
        results = []
        for future in futures:
            check()
            results.append(future.result())
    finally:
        # Wait for in-flight rows before another generation can start. Pending
        # stale rows are cancelled, rather than left to publish late results.
        pool.shutdown(wait=True, cancel_futures=True)
    check()
    return results
