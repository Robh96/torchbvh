"""Refresh only the 40 torchbvh k-NN/MLS rows in the public figure dataset.

python -m benchmarks.refresh_torchbvh ORIGINAL.csv MERGED.csv
The other 160 rows retain their exact original CSV text and measurements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torchbvh
from . import third_party


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh(original: Path, output: Path):
    if original.resolve() == output.resolve():
        raise ValueError('The immutable original and merged output must differ')
    original_bytes = original.read_bytes()
    lines = original_bytes.decode('utf-8').splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    old_rows = list(csv.DictReader(io.StringIO(''.join(lines))))
    if len(old_rows) != 200 or len(lines) != 201:
        raise ValueError('Expected the original 200-row single-line-record CSV')
    selected = {('knn','torchbvh'),('interpolation','torchbvh_mls')}
    def key(row):
        return (row['operation'],row['method'],int(row['batch']),
                int(row['dim']),int(row['points']))
    replacements = {}
    for case in third_party.cases():
        for runner, method in [(third_party.run_knn,'torchbvh'),
                               (third_party.run_interpolation,'torchbvh_mls')]:
            result = runner(case,warmup=1,repeats=3,methods=(method,))
            if len(result) != 1 or result[0]['status'] != 'ok':
                raise RuntimeError(f'Invalid refreshed measurement: {result}')
            replacements[key(result[0])] = result[0]
        print(case, 'complete', flush=True)
    merged = [lines[0]]
    retained = []
    replaced = []
    for line,row in zip(lines[1:],old_rows):
        if (row['operation'],row['method']) in selected:
            value = replacements.pop(key(row))
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer,fieldnames=header,lineterminator='\n')
            writer.writerow(value)
            merged.append(buffer.getvalue())
            replaced.append(key(row))
        else:
            merged.append(line)
            retained.append(line)
    assert len(replaced)==40 and len(retained)==160 and not replacements
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(''.join(merged).encode('utf-8'))
    merged_rows = list(csv.DictReader(io.StringIO(output.read_text())))
    assert len(merged_rows)==200
    for a,b in zip(old_rows,merged_rows):
        if (a['operation'],a['method']) not in selected:
            assert a==b, 'Retained row changed'
    assert original.read_bytes()==original_bytes, 'Original CSV changed'
    environment = third_party.environment()
    metadata = dict(measured_at=datetime.now(timezone.utc).isoformat(),
        retained_measurement_date='2026-09-23', refreshed_rows=40, unchanged_rows=160,
        unchanged_rows_text_sha256=hashlib.sha256(''.join(retained).encode()).hexdigest(),
        original_sha256=digest(original), merged_sha256=digest(output),
        binary_sha256=digest(torchbvh._C.__file__), environment=environment,
        protocol=dict(warmup=1,repeats=3,statistic='median',channels=64,k=4,
                      inputs='original seeds and prestaged shapes',setup='one-shot build included',
                      quality='outside timing',methods=['torchbvh','torchbvh_mls']))
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    print(json.dumps(refresh(args.original,args.output),indent=2))


if __name__=='__main__':
    main()
