import sys
from pathlib import Path
from microstructure.data.catalog import sync

chunk_file = sys.argv[1]
syms = Path(chunk_file).read_text().split()
failed = []
for s in syms:
    try:
        sync(Path('data'), s, 'aggTrades', '2023-06', '2023-06')
    except Exception as e:
        failed.append((s, str(e)[:100]))
print(f'CHUNK DONE: {len(syms) - len(failed)} ok, {len(failed)} failed')
for s, e in failed:
    print('FAILED', s, e)
