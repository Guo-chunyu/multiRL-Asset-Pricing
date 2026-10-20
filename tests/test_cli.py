import json
import subprocess
import sys


def test_cli_theory_produces_a_real_json_result(tmp_path):
    target = tmp_path / "theory.json"
    process = subprocess.run([sys.executable, "-m", "rlasset.cli", "theory", "--output", str(target), "--horizon", "100", "--episodes", "2"], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(target.read_text())
    assert set(result["policies"]) == {"analytic", "myopic", "zero"}
    assert result["policies"]["zero"]["mean_reward"] == 0
