import json
import re
from html import unescape
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Chatbot PTSP")
TOP50_PATH = BASE / "rekap proyek OSS_top 50 most favorite KBLI_provinsi_updated 310826.xlsx"
SOURCE_PATH = BASE / "scraping" / "data_kbli.json"
OUT_JSON = BASE / "data_kbli_top_50_scraping.json"
OUT_TXT = BASE / "data_kbli_top_50_scraping.txt"


def normalize_code(value):
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"(?i)v\d+$", "", text).strip()


def collect_top50_codes():
    df = pd.read_excel(TOP50_PATH, sheet_name="rekap top KBLI")
    codes = []
    seen = set()
    for value in df["kbli"].dropna().astype(str).str.strip():
        code = normalize_code(value)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return codes


def build_lookup(records):
    lookup = {}
    for rec in records:
        code = (
            str(rec.get("kode_kelompok") or "").strip()
            or str(rec.get("kode_subgolongan") or "").strip()
            or str(rec.get("kode_golongan") or "").strip()
            or str(rec.get("kode_golongan_pokok") or "").strip()
        )
        if code:
            lookup.setdefault(code, []).append(rec)
    return lookup


def match_top50_records():
    codes = collect_top50_codes()
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    lookup = build_lookup(records)

    matched = []
    seen_signature = set()
    for code in codes:
        for rec in lookup.get(code, []):
            signature = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            if signature not in seen_signature:
                seen_signature.add(signature)
                matched.append(rec)
    return matched


def clean_html_text(value):
    if value is None:
        return ""
    text = str(value)
    text = unescape(text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = text.replace("</p>", "\n").replace("</li>", "\n").replace("</ol>", "\n").replace("</ul>", "\n")
    text = re.sub(r"<li[^>]*>", "• ", text, flags=re.I)
    text = re.sub(r"<p[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_readable_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_html_text(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = extract_readable_text(item)
            if text:
                parts.append(text)
        return " | ".join(parts)
    if isinstance(value, dict):
        for key in ["uraian", "syarat_perizinan", "nama_dokumen", "parameter", "kewenangan", "nama", "isi", "teks", "judul"]:
            if key in value and value.get(key) not in (None, ""):
                text = extract_readable_text(value.get(key))
                if text:
                    return text
        for key in ["id", "local", "localization", "en"]:
            if key in value:
                text = extract_readable_text(value.get(key))
                if text:
                    return text
        parts = []
        for key, item in value.items():
            if key in {"id", "kode", "jangkaWaktu", "jangka_waktu", "satuan", "jenis", "label", "no", "No"}:
                continue
            text = extract_readable_text(item)
            if text:
                parts.append(text)
        return " | ".join(parts)
    return clean_html_text(value)


def format_detail_list(label, items):
    lines = []
    if not items:
        return lines
    for index, item in enumerate(items, 1):
        lines.append(f"   {label} #{index}:")
        if not isinstance(item, dict):
            cleaned = clean_html_text(item)
            if cleaned:
                lines.append(f"      - {cleaned}")
            continue

        for key in [
            "kode",
            "teks",
            "skala_usaha",
            "tingkat_resiko",
            "risiko",
            "luas_lahan",
            "jangka_waktu",
            "masa_berlaku",
            "nama_dokumen",
        ]:
            value = item.get(key)
            cleaned = clean_html_text(value)
            if cleaned and cleaned != "None":
                lines.append(f"      - {key}: {cleaned}")

        if label == "pb_umku_detail":
            if item.get("parameter_kewenangan"):
                lines.append("      - parameter_kewenangan:")
                for authority in item["parameter_kewenangan"]:
                    if isinstance(authority, dict):
                        parameter = clean_html_text(authority.get("parameter"))
                        kewenangan = clean_html_text(authority.get("kewenangan"))
                        if parameter or kewenangan:
                            lines.append(f"         * {parameter} -> {kewenangan}")
            if item.get("persyaratan"):
                lines.append("      - persyaratan:")
                for index, req in enumerate(item["persyaratan"], 1):
                    if isinstance(req, dict):
                        kode = clean_html_text(req.get("kode"))
                        isi = extract_readable_text(req.get("persyaratan"))
                        jangka = clean_html_text(req.get("jangkaWaktu") or req.get("jangka_waktu") or "")
                        if isi:
                            display_no = str(index).zfill(2)
                            prefix = f"[{display_no}] "
                            suffix = f" | {jangka}" if jangka else ""
                            lines.append(f"         * {prefix}{isi}{suffix}")
            if item.get("kewajiban"):
                lines.append("      - kewajiban:")
                for index, obligation in enumerate(item["kewajiban"], 1):
                    if isinstance(obligation, dict):
                        isi = extract_readable_text(obligation.get("kewajiban"))
                        kode = clean_html_text(obligation.get("kode"))
                        jangka = clean_html_text(obligation.get("jangkaWaktu") or obligation.get("jangka_waktu") or "")
                        if not isi:
                            isi = extract_readable_text(obligation)
                        if isi:
                            display_no = str(index).zfill(2)
                            prefix = f"[{display_no}] "
                            suffix = f" | {jangka}" if jangka else ""
                            lines.append(f"         * {prefix}{isi}{suffix}")
        else:
            if item.get("kewenangan"):
                lines.append("      - kewenangan:")
                for authority in item["kewenangan"]:
                    if isinstance(authority, dict):
                        parameter = clean_html_text(authority.get("parameter", ""))
                        kewenangan = clean_html_text(authority.get("kewenangan", ""))
                        if parameter or kewenangan:
                            lines.append(f"         * {parameter} -> {kewenangan}")
            if item.get("kewajiban"):
                lines.append("      - kewajiban:")
                for obligation in item["kewajiban"]:
                    if isinstance(obligation, dict):
                        uraian = clean_html_text(obligation.get("uraian") or obligation.get("nama") or "")
                        jangka = clean_html_text(obligation.get("jangka_waktu"))
                        satuan = clean_html_text(obligation.get("satuan") or "")
                        if uraian:
                            suff = f" | {jangka} {satuan}".strip() if jangka or satuan else ""
                            lines.append(f"         * {uraian}{suff}")
    return lines


def write_txt(records):
    lines = [
        "TOP 50 KBLI TERFAVORIT - DATA SCRAPING LENGKAP",
        "==============================================",
        f"Jumlah record: {len(records)}",
        "",
    ]
    for idx, rec in enumerate(records, 1):
        code = (
            str(rec.get("kode_kelompok") or "").strip()
            or str(rec.get("kode_subgolongan") or "").strip()
            or str(rec.get("kode_golongan") or "").strip()
            or str(rec.get("kode_golongan_pokok") or "").strip()
        )
        title = (
            str(rec.get("nama_kelompok") or "").strip()
            or str(rec.get("nama_subgolongan") or "").strip()
            or str(rec.get("nama_golongan") or "").strip()
            or str(rec.get("nama_golongan_pokok") or "").strip()
            or str(rec.get("uraian") or "").strip()
        )
        lines.append(f"{idx}. {code} | {title}")

        for key in [
            "url",
            "kode_kategori",
            "nama_kategori",
            "kode_golongan_pokok",
            "nama_golongan_pokok",
            "kode_golongan",
            "nama_golongan",
            "kode_subgolongan",
            "nama_subgolongan",
            "kode_kelompok",
            "nama_kelompok",
            "uraian",
            "ruang_lingkup",
            "pb_umku",
        ]:
            value = rec.get(key)
            if value is not None and str(value).strip():
                lines.append(f"   {key}: {value}")

        if rec.get("ruang_lingkup_detail"):
            lines.append(f"   ruang_lingkup_detail_count: {len(rec.get('ruang_lingkup_detail'))}")
            lines.extend(format_detail_list("ruang_lingkup_detail", rec.get("ruang_lingkup_detail")))
        if rec.get("pb_umku_detail"):
            lines.append(f"   pb_umku_detail_count: {len(rec.get('pb_umku_detail'))}")
            lines.extend(format_detail_list("pb_umku_detail", rec.get("pb_umku_detail")))
        if rec.get("hierarki"):
            lines.append(f"   hierarki: {json.dumps(rec.get('hierarki'), ensure_ascii=False)}")
        lines.append("")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    matched = match_top50_records()
    OUT_JSON.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")
    write_txt(matched)
    print(f"matched_records={len(matched)}")
    print(f"json={OUT_JSON}")
    print(f"txt={OUT_TXT}")


if __name__ == "__main__":
    main()
