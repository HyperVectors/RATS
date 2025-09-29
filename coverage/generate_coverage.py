import os
import sys
import subprocess
import shutil
import glob
from datetime import datetime
from xml.etree import ElementTree as ET


def run(cmd, cwd=None):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    output_lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    process.wait()
    return process.returncode, "".join(output_lines)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_cobertura(xml_path):
    if not os.path.exists(xml_path):
        return {"lines_valid": 0, "lines_covered": 0}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Cobertura-like schema
        lines_valid = root.attrib.get("lines-valid") or root.attrib.get("lines_valid") or "0"
        lines_covered = root.attrib.get("lines-covered") or root.attrib.get("lines_covered") or "0"
        return {
            "lines_valid": int(float(lines_valid)),
            "lines_covered": int(float(lines_covered)),
        }
    except Exception:
        return {"lines_valid": 0, "lines_covered": 0}


def pct(covered, valid):
    if valid == 0:
        return 0.0
    return 100.0 * covered / valid


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    coverage_dir = os.path.join(repo_root, "coverage")
    ensure_dir(coverage_dir)

    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Coverage run at {timestamp}\n")

    # 1) Rust coverage for rats
    log_lines.append("\n=== Running cargo tarpaulin for rats ===\n")
    rc, out = run(
        "cargo tarpaulin --out Xml --output-dir ../coverage --exclude-files \"*/tests/*\" \"*/examples/*\" \"*/main.rs\" \"*/readcsv.rs\"",
        cwd=os.path.join(repo_root, "rats"),
    )
    log_lines.append(out)
    if rc != 0:
        log_lines.append(f"rats coverage command failed with exit code {rc}\n")
    src_xml = os.path.join(coverage_dir, "cobertura.xml")
    rats_xml = os.path.join(coverage_dir, "rats-coverage.xml")
    if os.path.exists(src_xml):
        if os.path.exists(rats_xml):
            os.remove(rats_xml)
        shutil.move(src_xml, rats_xml)

    # 2) Rust coverage for ratspy
    log_lines.append("\n=== Running cargo tarpaulin for ratspy ===\n")
    rc, out = run(
        "cargo tarpaulin --out Xml --output-dir ../coverage --exclude-files \"*/tests/*\" \"*/examples/*\"",
        cwd=os.path.join(repo_root, "ratspy"),
    )
    log_lines.append(out)
    if rc != 0:
        log_lines.append(f"ratspy coverage command failed with exit code {rc}\n")
    src_xml = os.path.join(coverage_dir, "cobertura.xml")
    ratspy_xml = os.path.join(coverage_dir, "ratspy-coverage.xml")
    if os.path.exists(src_xml):
        if os.path.exists(ratspy_xml):
            os.remove(ratspy_xml)
        shutil.move(src_xml, ratspy_xml)

    # 3) Python coverage for ratspy tests
    log_lines.append("\n=== Preparing Python environment and building ratspy ===\n")
    # Install Python dependencies
    req_path = os.path.join(repo_root, "ratspy", "requirements.txt")
    if os.path.exists(req_path):
        rc, out = run(f"pip install -r \"{req_path}\"")
        log_lines.append(out)
        if rc != 0:
            log_lines.append(f"pip install -r requirements.txt failed with exit code {rc}\n")
    rc, out = run("pip install coverage pytest-cov maturin")
    log_lines.append(out)

    # Build and install wheel
    rc, out = run("python -m maturin build --release", cwd=os.path.join(repo_root, "ratspy"))
    log_lines.append(out)
    wheels_glob = os.path.join(repo_root, "ratspy", "target", "wheels", "*.whl")
    wheels = sorted(glob.glob(wheels_glob))
    if wheels:
        wheel = wheels[-1]
        rc, out = run(f"pip install \"{wheel}\"")
        log_lines.append(out)
        if rc != 0:
            log_lines.append(f"pip install wheel failed with exit code {rc}\n")
    else:
        log_lines.append("No wheel found under ratspy/target/wheels; Python tests may fail to import.\n")

    # Run pytest with coverage
    log_lines.append("\n=== Running Python coverage for ratspy tests ===\n")
    testrun_dir = os.path.join(repo_root, "testrun")
    ensure_dir(testrun_dir)
    rc, out = run(
        f"python -m coverage run --source=ratspy -m pytest -q --import-mode=importlib \"{os.path.join(repo_root, 'ratspy', 'tests')}\"",
        cwd=testrun_dir,
    )
    log_lines.append(out)
    if rc != 0:
        log_lines.append(f"pytest run failed with exit code {rc}\n")
    rc, out = run(f"python -m coverage xml -o \"{os.path.join(coverage_dir, 'python-coverage.xml')}\"", cwd=testrun_dir)
    log_lines.append(out)
    if rc != 0:
        log_lines.append(f"coverage xml failed with exit code {rc}\n")

    # 4) Parse and combine coverage
    python_xml = os.path.join(coverage_dir, "python-coverage.xml")
    python_stats = parse_cobertura(python_xml)
    print(f"Python coverage: {python_stats['lines_covered']}/{python_stats['lines_valid']} lines")
    rats_stats = parse_cobertura(rats_xml)
    ratspy_stats = parse_cobertura(ratspy_xml)

    total_valid = rats_stats["lines_valid"] + ratspy_stats["lines_valid"] + python_stats["lines_valid"]
    total_covered = rats_stats["lines_covered"] + ratspy_stats["lines_covered"] + python_stats["lines_covered"]

    # 5) Write report
    report_path = os.path.join(coverage_dir, "coverage_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RATS Project Coverage Report\n")
        f.write("\n--- Per-component Coverage ---\n")
        f.write(
            f"rats:    {rats_stats['lines_covered']}/{rats_stats['lines_valid']} lines "
            f"({pct(rats_stats['lines_covered'], rats_stats['lines_valid']):.2f}%)\n"
        )
        f.write(
            f"ratspy:  {ratspy_stats['lines_covered']}/{ratspy_stats['lines_valid']} lines "
            f"({pct(ratspy_stats['lines_covered'], ratspy_stats['lines_valid']):.2f}%)\n"
        )
        f.write(
            f"python:  {python_stats['lines_covered']}/{python_stats['lines_valid']} lines "
            f"({pct(python_stats['lines_covered'], python_stats['lines_valid']):.2f}%)\n"
        )
        f.write("\n--- Combined Coverage ---\n")
        f.write(f"total:   {total_covered}/{total_valid} lines ({pct(total_covered, total_valid):.2f}%)\n")
        f.write("\n--- Command Output ---\n")
        f.writelines(log_lines)

    print(f"\nCoverage report written to: {report_path}")


if __name__ == "__main__":
    main()


