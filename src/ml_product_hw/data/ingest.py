import argparse
import os
import tempfile
import urllib.request

from ml_product_hw.data.s3 import get_bucket, get_client, object_exists

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload raw CIFAR-10 archive to S3")
    parser.add_argument("--url", default=CIFAR_URL)
    parser.add_argument("--key", default="raw/cifar-10-python.tar.gz")
    args = parser.parse_args()

    client = get_client()
    bucket = get_bucket()

    # raw data is immutable, so never overwrite existing object
    if object_exists(client, bucket, args.key):
        print(f"s3://{bucket}/{args.key} already exists, nothing to do")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "cifar-10-python.tar.gz")
        print(f"downloading {args.url}")
        urllib.request.urlretrieve(args.url, path)
        print(f"uploading to s3://{bucket}/{args.key}")
        client.upload_file(path, bucket, args.key)

    print("done")


if __name__ == "__main__":
    main()
