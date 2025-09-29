from pathlib import Path, PureWindowsPath

def as_path(p) -> Path:
    # Accept Path or str; convert Windows-style separators to native Path on POSIX.
    if isinstance(p, Path):
        return p
    s = str(p)
    if "\\" in s:                 # looks Windows-y
        return Path(PureWindowsPath(s))
    return Path(s)
