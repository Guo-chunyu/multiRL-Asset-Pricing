"""Export read-only research outputs without changing local experiment records."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RUNS = {"ashare", "demo", "walkforward", "comparison_v2", "walkforward_v2",
                "risk0_v2", "risk8_v2", "paper_v2", "research_v2", "research_v3", "research_v4"}
ALLOWED_ROOT_FILES = {"theory.json", "verification.json", "verification_v2.json",
                      "pytest.xml", "pytest_v2.xml", "causality_v2_pytest.xml",
                      "pytest_v3.xml", "verification_v3.json", "pytest_v4.xml", "verification_v4.json"}
WITHDRAWN_ADJUSTED_RUNS = {"ashare", "walkforward", "comparison_v2", "walkforward_v2",
                          "risk0_v2", "risk8_v2", "research_v2"}


def scrub(value):
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        for variant in (str(ROOT), ROOT.as_posix()):
            value = value.replace(variant, "${PROJECT_ROOT}")
        value = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", "${USERPROFILE}", value)
        return value
    return value


def main():
    destination = ROOT / "reports" / "runs"
    records = []
    for path in sorted((ROOT / "runs").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".csv", ".md", ".png", ".xml"}:
            continue
        relative = path.relative_to(ROOT / "runs")
        if not ((len(relative.parts) > 1 and relative.parts[0] in ALLOWED_RUNS)
                or str(relative) in ALLOWED_ROOT_FILES):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            content = scrub(json.loads(path.read_text(encoding="utf-8")))
            target.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
                              encoding="utf-8", newline="\n")
        elif path.suffix == ".xml":
            tree = ET.parse(path)
            for item in tree.iter():
                item.attrib.pop("hostname", None)
                item.attrib = {k: scrub(v) for k, v in item.attrib.items()}
                if item.text:
                    item.text = scrub(item.text)
            tree.write(target, encoding="utf-8", xml_declaration=True)
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
        elif path.suffix in {".md", ".csv"}:
            content = scrub(path.read_text(encoding="utf-8"))
            if path.suffix == ".md" and relative.parts[0] in WITHDRAWN_ADJUSTED_RUNS:
                content = ("> 2026-10-23 correction: the market-performance interpretation of these "
                           "Tencent hfq price-ratio results is withdrawn because additive adjustments "
                           "distort returns. Original numerical records are retained for audit. "
                           "See docs/data-audit-v3.md and the v3 raw-price/corporate-action study.\n\n" + content)
            target.write_text(content, encoding="utf-8", newline="\n")
        else:
            target.write_bytes(path.read_bytes())
        records.append({"path": target.relative_to(ROOT).as_posix(),
                        "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "published_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"purpose": "Read-only research archive, not resumable training checkpoints.",
                "embedded_hash_scope": "Hashes and record seals embedded inside research files refer to original local artifacts before path scrubbing and LF normalization; they are intentionally preserved, not recomputed.",
                "public_integrity": "Verify each published file using files[].published_sha256; files[].original_sha256 identifies its original local bytes. This is an integrity inventory, not a cryptographic signature by an external authority.",
                "transformations": ["Local workspace/profile paths replaced with placeholders.",
                                    "Published text uses canonical LF newlines; original embedded hashes and seals remain provenance of local originals.",
                                    "JUnit machine hostnames removed; numeric research values unchanged.",
                                    "Dated performance-correction notices added to superseded adjusted-price Markdown reports."],
                "excluded": ["Market-data snapshots", "Model ZIPs", "Environment/vendor folders",
                             "Third-party paper full text and source snapshots", "Console logs"],
                "files": records}
    (ROOT / "reports" / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"Exported {len(records)} research files to reports/runs")


if __name__ == "__main__":
    main()
