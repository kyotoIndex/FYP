"""Prepare shared datasets, a pinned BERT snapshot and fresh cloud configs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
DATA_HASHES = {
    'train.tsv': '02eaea97ddfaa1ba0fa169c33a18bf9c8b280a0209ab8dab526dffb80248076f',
    'dev.tsv': '419184c5e817e3a450fccde842ea872cd745c0ff0cf146f8f7255b18a6671b02',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--data-only', action='store_true', help='Prepare and verify datasets only')
    args = parser.parse_args()
    shared_data = ROOT.parent/'data'
    data = ROOT/'data'
    (data/'AG_NEWS').mkdir(parents=True, exist_ok=True)
    for name, expected in DATA_HASHES.items():
        path = data/'AG_NEWS'/name
        raw = path.read_bytes() if path.exists() else (shared_data/'AG_NEWS'/name).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        candidates = [raw, lf, lf.replace(b'\n', b'\r\n')]
        canonical = next((b for b in candidates if hashlib.sha256(b).hexdigest() == expected), None)
        if canonical is None:
            raise RuntimeError(f'Dataset content differs from frozen experiment: {name}')
        if not path.exists() or raw != canonical:
            path.write_bytes(canonical)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Dataset SHA-256 mismatch: {path}: {actual}')
    print('Both AG_NEWS input hashes match the completed HetSA experiments.')
    if args.data_only:
        return
    model = (args.model_dir or ROOT/'models/bert-base-uncased').resolve()
    if args.download_model:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='bert-base-uncased', revision=REVISION,
            local_dir=str(model), local_dir_use_symlinks=False,
            allow_patterns=['config.json','pytorch_model.bin','vocab.txt',
                            'tokenizer.json','tokenizer_config.json','special_tokens_map.json'])
    if not (model/'config.json').is_file() or not (model/'vocab.txt').is_file() or not any(
            (model/name).is_file() for name in ('pytorch_model.bin','model.safetensors')):
        raise FileNotFoundError(f'BERT files missing in {model}; use --download-model or --model-dir')
    config = json.loads((ROOT/'configs/hetsa_lora_confirm_extension_v1.json').read_text(encoding='utf-8'))
    config['dataset']['model'] = str(model)
    config['dataset']['local_files_only'] = True
    config['model_source'] = {'repo_id':'bert-base-uncased','revision':REVISION,
        'preparation':'downloaded pinned snapshot' if args.download_model else 'user-provided local model directory'}
    config['name'] = 'hetsa_cloud_confirm_all_v1'
    config['federated']['seeds'] = [202,303,404,505,606]
    config['output_dir'] = 'results/hetsa_cloud_confirm_all_v1'
    (ROOT/'configs/cloud_confirm_all_v1.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    config['name'] = 'hetsa_cloud_mechanism_v1'
    config['federated']['seeds'] = [1201,1301,1401,1501,1601]
    config['output_dir'] = 'results/hetsa_cloud_mechanism_v1'
    (ROOT/'configs/cloud_mechanism_v1.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print('Created configs/cloud_confirm_all_v1.json and configs/cloud_mechanism_v1.json.')

if __name__ == '__main__':
    main()
