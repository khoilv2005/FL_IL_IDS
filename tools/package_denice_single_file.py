"""Copy the small Kaggle launcher; supporting source is downloaded at its pinned revision."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'train_incremental_kaggle.py'
    data = source.read_bytes()
    compile(data, str(source), 'exec')
    if b'EMBEDDED_SUPPORT_B85' in data:
        raise ValueError('Remove embedded source; Kaggle uses the pinned GitHub revision')
    output = root / 'output/single_file/train_incremental_kaggle.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'{output}: {len(data)} bytes (identical to root launcher)')


if __name__ == '__main__':
    main()
