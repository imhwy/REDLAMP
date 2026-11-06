"""Download / unpack every dataset used in the paper into ./dataset (same layout loaders/load.py expects).

    python prepare_data.py                 # all datasets
    python prepare_data.py smd smap        # only some of them

Layout:
    dataset/NASA/{train,test,labeled_anomalies.csv}                 SMAP + MSL (from dataset/NASA.zip)
    dataset/ServerMachineDataset/{train,test,test_label}/machine-*.txt
    dataset/AnomalyArchive/*.txt                                    UCR Anomaly Archive (250 series)
    dataset/IOPS/KPI-*.{train,test}.out                             AIOps (from TSB-UAD-Public.zip)
"""
import os
import re
import sys
import zipfile

import requests
from tqdm import tqdm

from loaders.load import MACHINES, SMD_URL, ANOMALY_ARCHIVE_URI

ROOT = './dataset'
DOWNLOADS = os.path.join(ROOT, '_downloads')
TSB_UAD_URI = 'https://www.thedatum.org/datasets/TSB-UAD-Public.zip'


def download(url, path):
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    tmp = path + '.part'
    with open(tmp, 'wb') as f, tqdm(total=total, unit='iB', unit_scale=True, desc=os.path.basename(path)) as t:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            t.update(len(chunk))
    os.replace(tmp, path)
    return path


def extract_matching(zip_path, pattern, out_dir):
    """Extract members whose path matches `pattern` (group 1 = output file name) into out_dir, flattened."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = re.search(pattern, name)
            if m:
                with z.open(name) as src, open(os.path.join(out_dir, m.group(1)), 'wb') as dst:
                    dst.write(src.read())
                n += 1
    return n


def prepare_nasa():
    out = os.path.join(ROOT, 'NASA')
    if not os.path.isfile(os.path.join(out, 'labeled_anomalies.csv')):
        with zipfile.ZipFile(os.path.join(ROOT, 'NASA.zip')) as z:
            z.extractall(ROOT)
    print('NASA (SMAP/MSL):', len(os.listdir(os.path.join(out, 'train'))), 'channels')


def prepare_smd():
    out = os.path.join(ROOT, 'ServerMachineDataset')
    for machine in tqdm(MACHINES, desc='SMD'):
        for group in ['train', 'test', 'test_label']:
            download(f'{SMD_URL}/{group}/{machine}.txt', os.path.join(out, group, f'{machine}.txt'))
    print('SMD:', len(os.listdir(os.path.join(out, 'train'))), 'machines')


def prepare_ucr():
    out = os.path.join(ROOT, 'AnomalyArchive')
    if not (os.path.isdir(out) and len(os.listdir(out)) >= 250):
        z = download(ANOMALY_ARCHIVE_URI, os.path.join(DOWNLOADS, 'UCR.zip'))
        extract_matching(z, r'UCR_Anomaly_FullData/([^/]+\.txt)$', out)
    print('UCR:', len(os.listdir(out)), 'series')


def prepare_iops():
    out = os.path.join(ROOT, 'IOPS')
    if not (os.path.isdir(out) and len(os.listdir(out)) >= 58):
        z = download(TSB_UAD_URI, os.path.join(DOWNLOADS, 'TSB-UAD-Public.zip'))
        extract_matching(z, r'^TSB-UAD-Public/IOPS/(KPI-[^/]+\.out)$', out)
    print('AIOps (IOPS):', len(os.listdir(out)) // 2, 'KPIs')


PREPARE = {'smap': prepare_nasa, 'msl': prepare_nasa, 'smd': prepare_smd,
           'anomaly_archive': prepare_ucr, 'ucr': prepare_ucr, 'iops': prepare_iops, 'aiops': prepare_iops}


def prepare(datasets=None):
    done = set()
    for name in datasets or ['smap', 'smd', 'anomaly_archive', 'iops']:
        fn = PREPARE[name.lower()]
        if fn not in done:
            fn()
            done.add(fn)


if __name__ == '__main__':
    prepare(sys.argv[1:] or None)
