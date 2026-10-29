from pathlib import Path
import os
import shutil
import stat
import subprocess
import time

import numpy as np
import pandas as pd
from urllib.request import Request, urlopen

# Thu muc luu du lieu
BASE = Path(r"D:\NCKH\PTDLKD")
OUTPUT = BASE / "dataset_original"
RSR_SOURCE = BASE / "_RSR_source"  # Bo nho tam cho NASDAQ / NYSE

RSR_GIT = "https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git"
RSR_RAW = "https://raw.githubusercontent.com/fulifeng/Temporal_Relational_Stock_Ranking/master/data"
SP500_RAW = "https://raw.githubusercontent.com/SJTU-DMTai/StockMixer/master/dataset/SP500"

# True: thu xoa _RSR_source sau khi da sao chep NASDAQ va NYSE.
# Neu Windows khoa file .git, script van hoan thanh va giu folder de xoa sau.
REMOVE_RSR_CACHE = True


def download(url, dest, is_npy=False):
    """Tai file bang HTTP, co retry va tranh luu file bi tai do."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if not is_npy:
            print(f"Da co, bo qua: {dest.name}")
            return
        with dest.open("rb") as f:
            if f.read(6) == b"\x93NUMPY":
                print(f"Da co, bo qua: {dest.name}")
                return

    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as response, part.open("wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
            if part.stat().st_size == 0:
                raise ValueError("File tai ve rong")
            if is_npy:
                with part.open("rb") as f:
                    if f.read(6) != b"\x93NUMPY":
                        raise ValueError("Noi dung khong phai file NumPy .npy")
            part.replace(dest)
            size = dest.stat().st_size / (1024 * 1024)
            print(f"Tai xong: {dest.name} ({size:.1f} MB)")
            return
        except Exception as e:
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Tai that bai {url}: {e}") from e
            print(f"Thu lai {dest.name}: {e}")
            time.sleep(2 * attempt)


def prepare_rsr():
    """Dung thu muc Git da co, neu chua co thi sparse clone tu GitHub."""
    if not (RSR_SOURCE / ".git").exists():
        if RSR_SOURCE.exists():
            raise RuntimeError(f"{RSR_SOURCE} ton tai nhung khong phai Git repository")
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            RSR_GIT, str(RSR_SOURCE)
        ], check=True)

    subprocess.run([
        "git", "-C", str(RSR_SOURCE), "sparse-checkout",
        "set", "data/2013-01-01"
    ], check=True)


def download_nasdaq_nyse():
    print("\n=== NASDAQ / NYSE (Feng et al.) ===")
    for market, expected in [("NASDAQ", 1026), ("NYSE", 1737)]:
        folder = OUTPUT / market
        folder.mkdir(parents=True, exist_ok=True)
        ticker_filename = f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv"
        ticker_file = folder / "tickers.csv"
        download(f"{RSR_RAW}/{ticker_filename}", ticker_file)
        tickers = [line.strip().split("\t")[0]
                   for line in ticker_file.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()]

        missing = [f"{market}_{ticker}_1.csv" for ticker in tickers
                   if not (folder / f"{market}_{ticker}_1.csv").is_file()
                   or (folder / f"{market}_{ticker}_1.csv").stat().st_size == 0]

        if missing:
            print(f"{market}: dang lay {len(missing)} file con thieu...")
            prepare_rsr()
            source_dir = RSR_SOURCE / "data" / "2013-01-01"
            for i, name in enumerate(missing, 1):
                src = source_dir / name
                if src.is_file() and src.stat().st_size > 0:
                    shutil.copy2(src, folder / name)
                if i % 200 == 0:
                    print(f"  {market}: da kiem tra {i}/{len(missing)} file")

        valid = sum(1 for ticker in tickers
                    if (folder / f"{market}_{ticker}_1.csv").is_file()
                    and (folder / f"{market}_{ticker}_1.csv").stat().st_size > 0)
        print(f"{market}: {valid}/{len(tickers)} file co du lieu (bai bao: {expected} co phieu)")
        if valid != len(tickers):
            print("CANH BAO: Mot so file goc khong tim thay, hay kiem tra lai GitHub")


def download_sp500():
    print("\n=== SP500 (StockMixer) ===")
    folder = OUTPUT / "SP500"
    for name in ["SP500.npy", "baseline_data_sp500.npy", "sp500_ticker.csv"]:
        download(f"{SP500_RAW}/{name}", folder / name, is_npy=name.endswith(".npy"))


def export_tensor_csv(data, target):
    """Xuat ma tran 2D/3D sang CSV, giu stock_index va time_index."""
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Expected 2D/3D tensor, got {data.shape}")

    n_stocks, n_days, n_features = data.shape
    columns = [f"feature_{i}" for i in range(n_features)]
    for i in range(n_stocks):
        frame = pd.DataFrame(data[i], columns=columns)
        frame.insert(0, "time_index", np.arange(n_days))
        frame.insert(0, "stock_index", i)
        frame.to_csv(target, mode="w" if i == 0 else "a",
                     index=False, header=(i == 0), encoding="utf-8-sig")
        if (i + 1) % 100 == 0:
            print(f"  Da chuyen {i + 1}/{n_stocks} co phieu")
    print(f"Da tao: {target} ({n_stocks} co phieu, {n_days} ngay, {n_features} features)")


def convert_sp500_to_csv():
    """Xuat ca SP500.npy va baseline_data_sp500.npy sang CSV."""
    folder = OUTPUT / "SP500"
    tensor_path = folder / "SP500.npy"
    baseline_path = folder / "baseline_data_sp500.npy"

    if not tensor_path.exists() or not baseline_path.exists():
        raise FileNotFoundError("Thieu file SP500.npy hoac baseline_data_sp500.npy")

    # Tensor da tien xu ly ma StockMixer dung de train.
    data = np.load(tensor_path, allow_pickle=False)
    print(f"SP500.npy shape: {data.shape}")
    export_tensor_csv(data, folder / "SP500.csv")

    # Baseline co the la dict: ticker -> DataFrame gia goc.
    # Chi cho phep unpickle neu file duoc tai tu repo tin cay.
    baseline = np.load(baseline_path, allow_pickle=True)
    if isinstance(baseline, np.ndarray) and baseline.ndim == 0:
        baseline = baseline.item()

    target = folder / "baseline_data_sp500.csv"
    if isinstance(baseline, dict):
        count = 0
        for stock_key, table in baseline.items():
            frame = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
            frame = frame.reset_index()  # giu cot ngay neu ngay la index
            frame.insert(0, "stock_key", str(stock_key))
            frame.to_csv(target, mode="w" if count == 0 else "a",
                         header=(count == 0), index=False, encoding="utf-8-sig")
            count += 1
        print(f"Da tao: {target} ({count} ma co phieu)")
    elif isinstance(baseline, np.ndarray) and baseline.ndim in (2, 3):
        if baseline.dtype == object:
            raise ValueError("Baseline data has object dtype and needs inspection")
        export_tensor_csv(baseline, target)
    else:
        raise TypeError(f"Kieu du lieu baseline chua duoc ho tro: {type(baseline).__name__}")


def unlock_and_delete(func, path, error_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    download_nasdaq_nyse()
    download_sp500()
    print("\n=== CHUYEN SP500 SANG CSV ===")
    convert_sp500_to_csv()

    print("\n=== KIEM TRA ===")
    for market in ["NASDAQ", "NYSE", "SP500"]:
        folder = OUTPUT / market
        files = [p for p in folder.iterdir() if p.is_file() and p.stat().st_size > 0]
        total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
        print(f"{market}: {len(files)} file, {total_mb:.1f} MB -> {folder}")

    if REMOVE_RSR_CACHE and RSR_SOURCE.exists():
        try:
            shutil.rmtree(RSR_SOURCE, onerror=unlock_and_delete)
            print("Da xoa thu muc tam _RSR_source")
        except OSError as e:
            print(f"Khong xoa duoc _RSR_source do Windows dang khoa file: {e}")
            print("Khong anh huong den dataset_original; co the xoa thu cong sau.")

    print("\nHOAN TAT! Du lieu nam tai:", OUTPUT)


if __name__ == "__main__":
    main()
