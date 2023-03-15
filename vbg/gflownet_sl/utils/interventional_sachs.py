import os
import gzip
import urllib.request


def download_sachs_intervention():
    path = 'data'
    if not os.path.exists(path):
        os.mkdir(path)

    url = 'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz'
    out_file = 'data/sachs.interventional.txt'
    if os.path.exists(out_file):
        return

    # Download archive
    try:
        # Read the file inside the .gz archive located at url
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

        # write to file in binary mode 'wb'
        with open(out_file, 'wb') as f:
            f.write(file_content)
            return 0

    except Exception as e:
        print(e)
        raise
        # return 1
