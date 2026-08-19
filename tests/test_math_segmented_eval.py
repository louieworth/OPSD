from pathlib import Path
import os
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MATH_SEGMENT_LIB = REPO_ROOT / "scripts/lib/math_segment_common.sh"


class MathSegmentedEvalTests(unittest.TestCase):
    def test_each_checkpoint_is_evaluated_and_only_final_metadata_remains(self):
        output_root = REPO_ROOT / "outputs/math-segment-tests"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            experiment = Path(temporary)
            eval_log = experiment / "eval.log"
            eval_runner = experiment / "fake_math_eval.sh"
            eval_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'test -d "$EVAL_EXPERIMENT_DIR/checkpoint-$CHECKPOINTS"\n'
                'printf "%s\\n" "$CHECKPOINTS" >> "$MATH_FAKE_EVAL_LOG"\n'
            )
            eval_runner.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "MATH_EVAL_RUNNER": str(eval_runner),
                    "MATH_FAKE_EVAL_LOG": str(eval_log),
                }
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; '
                    'fake_train() { '
                    'local target="$1" latest="$2" root="$3"; '
                    '[[ -z "$latest" ]] || rm -rf -- "$latest"; '
                    'mkdir -p "$root/checkpoint-$target"; '
                    'touch "$root/checkpoint-$target/adapter_model.safetensors"; '
                    '}; '
                    'math_run_segmented_training 100 25 "$2" 1.7b "$2/results" '
                    'math_test fake_train "$2"',
                    "math-segment-test",
                    str(MATH_SEGMENT_LIB),
                    str(experiment),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(eval_log.read_text().splitlines(), ["25", "50", "75", "100"])
            self.assertEqual(list(experiment.glob("checkpoint-*")), [])
            self.assertEqual(
                len(list(experiment.glob(".math_eval_complete_checkpoint-*"))),
                4,
            )


if __name__ == "__main__":
    unittest.main()
