"""Download an immutable, minimal-column snapshot from Chicago's public API."""
from pathlib import Path
import datetime as dt
import gzip
import hashlib
import io
import json
import urllib.parse
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://data.cityofchicago.org/resource/4ijn-s7e5.csv'
COLUMNS = ['inspection_id','license_','facility_type','risk','address','zip',
           'inspection_date','inspection_type','results','latitude','longitude']

def main():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    frames, sources = [], []
    for year in range(2019, 2026):
        params = {'$select': ','.join(COLUMNS),
                  '$where': f"inspection_date >= '{year}-01-01T00:00:00' AND inspection_date < '{year+1}-01-01T00:00:00'",
                  '$order': 'inspection_date,inspection_id', '$limit': '50000'}
        url = ENDPOINT + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        df = pd.read_csv(io.BytesIO(content), dtype=str)
        if len(df) >= 50000:
            raise ValueError('Year hit API limit; implement pagination before continuing.')
        frames.append(df)
        sources.append({'year': year, 'url': url, 'rows': len(df),
                        'response_sha256': hashlib.sha256(content).hexdigest()})
        print(f'{year}: {len(df):,} records', flush=True)
    combined = pd.concat(frames, ignore_index=True)
    content = combined.to_csv(index=False).encode()
    target = folder / 'inspections_2019_2025.csv.gz'
    with target.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
            gz.write(content)
    manifest = {'publisher': 'City of Chicago / Department of Public Health',
                'dataset': '4ijn-s7e5',
                'landing_page': 'https://data.cityofchicago.org/d/4ijn-s7e5',
                'downloaded_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'columns': COLUMNS, 'rows': len(combined), 'sources': sources,
                'csv_sha256': hashlib.sha256(content).hexdigest(),
                'snapshot_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'note': 'Current export of historical events; not an archived as-of publication snapshot. No violations text downloaded.'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {len(combined):,} rows to {target}', flush=True)

if __name__ == '__main__':
    main()
