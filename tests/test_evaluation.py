"""Offline evaluation checks; no credentials or network calls required."""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_outputs as evaluator
from moi.scoring import aggregate_question_scores, normalize_score


class EvaluationTests(unittest.TestCase):
    def test_hierarchical_mean_and_na(self):
        rows = [{'question_id': str(i), 'criterion': criterion, 'score': score}
                for i, (criterion, score) in enumerate([
                    ('IF-Fidelity', 1), ('IF-Fidelity', 0.6),
                    ('IF-Selection', 1), ('VQ-Integrity', 0.6)])]
        scores = aggregate_question_scores(rows, {'IF': 0.6, 'VQ': 0.4, 'DP': 0})
        self.assertAlmostEqual(scores['group_scores']['IF'], 0.9)
        self.assertNotIn('DP', scores['group_scores'])
        self.assertAlmostEqual(scores['usable_score'], 0.9 ** 0.6 * 0.6 ** 0.4)

    def test_invalid_responses(self):
        for response in ['{"score":0.5}', '{"score":true}', 'API error', 'text "score": 1']:
            with self.assertRaises((ValueError, TypeError)):
                evaluator.parse_score_response(response)
        for response, expected in [('{"score":0}', 0), ('{"score":0.6}', 0.6), ('{"score":1}', 1)]:
            self.assertEqual(evaluator.parse_score_response(response), expected)

    def test_all_question_records_score(self):
        questions = evaluator.load_questions(ROOT / 'data/questions.jsonl')
        self.assertEqual(len(questions), 1150)
        for row in questions.values():
            scores = aggregate_question_scores(
                [dict(q, score=1) for q in row['questions']], row['group_weights'])
            self.assertEqual(scores['usable_score'], 1)
            zero = evaluator.zero_question_results(row, 'unsupported')
            self.assertEqual(aggregate_question_scores(zero, row['group_weights'])['usable_score'], 0)

    def test_dry_run_and_unsupported_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'example'
            sample = model / 'gen/animal/gen_animal_000'
            sample.mkdir(parents=True)
            (sample / 'prompt.json').write_text(json.dumps({'entry_id': 'gen_animal_000', 'input_images': []}))
            (sample / 'seed_0.png').write_bytes(b'not opened during dry-run')
            command = [sys.executable, str(ROOT/'scripts/eval_outputs.py'), '--model-tag', 'example',
                       '--judge', 'openai', '--judge-model', 'gpt-4o', '--output-root', tmp,
                       '--mode', 'gen', '--dry-run']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('pending=1', result.stdout)
            self.assertFalse((model/'eval').exists())
            command = command[:-3] + ['--mode', 'edit', '--task', 'ref/b1r3/mixed',
                                     '--multi-image-support', 'no', '--limit', '1']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = list((model/'eval/gpt-4o/ref/b1r3/mixed').glob('*/seed_0.json'))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text())
            self.assertEqual(record['status'], 'unsupported_capability')
            self.assertEqual(record['scores']['usable_score'], 0)
            self.assertFalse(evaluator.result_is_current(records[0], record['question_record_sha256'], 'other-judge'))


if __name__ == '__main__':
    unittest.main()
