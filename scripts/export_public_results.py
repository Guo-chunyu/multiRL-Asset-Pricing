"""Export read-only research outputs without changing local experiment records."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RUNS = {"ashare", "demo", "walkforward", "comparison_v2", "walkforward_v2",
                "risk0_v2", "risk8_v2", "paper_v2", "research_v2"}
ALLOWED_ROOT_FILES = {"theory.json", "verification.json", "verification_v2.json",
                      "pytest.xml", "pytest_v2.xml", "causality_v2_pytest.xml"}


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
            target.write_text(scrub(path.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        else:
            target.write_bytes(path.read_bytes())
        records.append({"path": target.relative_to(ROOT).as_posix(),
                        "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "published_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"purpose": "Read-only research archive, not resumable training checkpoints.",
                "transformations": ["Local workspace/profile paths replaced with placeholders.",
                                    "JUnit machine hostnames removed; numeric research values unchanged."],
                "excluded": ["Market-data snapshots", "Model ZIPs", "Environment/vendor folders",
                             "Third-party paper full text and source snapshots", "Console logs"],
                "files": records}
    (ROOT / "reports" / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"Exported {len(records)} research files to reports/runs")


if __name__ == "__main__":
    main()
