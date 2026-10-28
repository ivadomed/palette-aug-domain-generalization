"""Extract a few members of a large remote zip over HTTP range requests (no full download).

  python scripts/git_annex/fetch_zip_members.py <url> <out_dir> <member-substring> [...]

Every member whose name ends with one of the given substrings is written to <out_dir>/<member name>.
Used to check single source cases against our BIDS copies when the full archive is no longer on disk.
Network only, light on CPU: run on a login node (compute nodes have no internet).
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    url, out = sys.argv[1], Path(sys.argv[2])
    wanted = sys.argv[3:]
    zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 22))
    names = [n for n in zf.namelist() if any(n.endswith(w) for w in wanted)]
    print(f"{len(zf.namelist())} members in archive, {len(names)} selected")
    for n in names:
        dst = out / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as src, open(dst, "wb") as fh:
            while chunk := src.read(1 << 22):
                fh.write(chunk)
        print(f"  {n} -> {dst} ({dst.stat().st_size} B)")


if __name__ == "__main__":
    main()
