import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Chatbot PTSP")
TOP50_PATH = BASE / "rekap proyek OSS_top 50 most favorite KBLI_provinsi_updated 310826.xlsx"
MAPPING_PATH = BASE / "KBLI_2025_mapping_cakep.xlsx"
OUT_XLSX = BASE / "data_kbli_top_50.xlsx"
OUT_JSON = BASE / "data_kbli_top_50.json"
OUT_TXT = BASE / "data_kbli_top_50.txt"


def normalize_kbli_code(value):
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"(?i)v\d+$", "", text).strip()
    return text


def main():
    df_top = pd.read_excel(TOP50_PATH, sheet_name="rekap top KBLI")
    codes = []
    seen = set()
    for value in df_top["kbli"].dropna().astype(str).str.strip():
        code = normalize_kbli_code(value)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)

    df = pd.read_excel(MAPPING_PATH, sheet_name="Ketentuan")
    if "Kode KBLI 2025" not in df.columns:
        raise ValueError("Sheet 'Ketentuan' tidak memiliki kolom 'Kode KBLI 2025'.")

    df["Kode KBLI 2025"] = df["Kode KBLI 2025"].astype(str).str.strip()
    filtered = df[df["Kode KBLI 2025"].isin(codes)].copy()
    filtered.reset_index(drop=True, inplace=True)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="data_kbli_top_50")

    OUT_JSON.write_text(
        json.dumps(filtered.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "DATA KBLI TOP 50 - SHEET KETENTUAN",
        "===================================",
        f"Jumlah data: {len(filtered)}",
        "",
    ]
    for idx, row in enumerate(filtered.to_dict(orient="records"), 1):
        lines.append(f"{idx}. {row.get('Kode KBLI 2025', '')} | {row.get('Judul KBLI 2025', '')}")
        for key in [
            "No Ruang Lingkup",
            "Ruang Lingkup",
            "Skala Usaha",
            "Ketentuan",
            "Luas Lahan",
            "Tingkat Risiko",
            "Perizinan Berusaha",
            "Jangka Waktu",
        ]:
            value = row.get(key, "")
            if pd.notna(value) and str(value).strip():
                lines.append(f"   {key}: {value}")
        lines.append("")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"top50_codes={len(codes)}")
    print(f"matched_rows={len(filtered)}")
    print(f"excel={OUT_XLSX}")
    print(f"json={OUT_JSON}")
    print(f"txt={OUT_TXT}")


if __name__ == "__main__":
    main()
