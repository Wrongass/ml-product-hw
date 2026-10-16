import argparse
import os

from ml_product_hw.data.s3 import get_bucket, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw CIFAR-10 archive from S3")
    parser.add_argument("--key", default="raw/cifar-10-python.tar.gz")
    parser.add_argument("--out", default="data/raw/cifar-10-python.tar.gz")
    args = parser.parse_args()

    client = get_client()
    bucket = get_bucket()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"downloading s3://{bucket}/{args.key} to {args.out}")
    client.download_file(bucket, args.key, args.out)
    print("done")


if __name__ == "__main__":
    main()
