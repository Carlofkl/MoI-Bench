"""Offline checks for grouping, denominator policies and record validation."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import aggregate_scores as aggregate
import eval_outputs as evaluator


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = evaluator.load_questions(ROOT/'data/questions.jsonl')
        cls.samples = json.loads((ROOT/'data/samples.json').read_text())['items']

    def record(self, sid, value):
        question = self.questions[sid]
        return evaluator.build_result_record(
            entry_id=sid, item=self.samples[sid], question_record=question,
            question_hash=evaluator.stable_hash(question), input_images=[],
            results=[dict(q, score=value) for q in question['questions']], status='scored',
            judge_info={'model':'offline-fixture', 'prompt_templates_hash':evaluator.PROMPT_TEMPLATES_HASH},
            seed_image='fixture.png')

    def test_exports_and_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sid, value in [('gen_animal_000', 1), ('gen_animal_001', 0.6)]:
                p=root/'model/eval/test/gen/animal'/sid/'seed_0.json'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(self.record(sid,value)))
            status=root/'status.json'
            status.write_text(json.dumps({'model':{'gen_animal_002':{'status':'unsupported_capability','reason':'fixture'}}}))
            for policy in ['fixed','available']:
                dest=root/policy
                result=subprocess.run([sys.executable,str(ROOT/'scripts/aggregate_scores.py'),
                    '--models','model','--judge-tag','test','--output-root',str(root),
                    '--results-dir',str(dest),'--denominator',policy,'--status-file',str(status)],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                def rows(name):
                    with (dest/(name+'.csv')).open() as handle:
                        return list(csv.DictReader(handle))
                summary=rows('summary')[0]
                self.assertAlmostEqual(float(summary['score']),1.6/(1150 if policy=='fixed' else 3))
                self.assertEqual(summary['n_unsupported_capability'],'1')
                self.assertEqual(summary['n_missing_output'],'1147')
                self.assertEqual(summary['aggregation_status'],'incomplete')
                self.assertEqual(len(rows('subtask_scores')),64)
                self.assertEqual(len(rows('operation_scores')),16)
                self.assertEqual(len(rows('target_scores')),8)
                self.assertEqual(len(rows('criteria_scores')),11)
                cross=rows('operation_target_scores')
                self.assertEqual(len(cross),128)
                for cell in cross:
                    if cell['n_expected']=='0':
                        self.assertEqual(cell['score'],'')
                        self.assertEqual(cell['aggregation_status'],'not_applicable')
                self.assertFalse(any(row['criterion'].startswith('DP') for row in rows('sample_criteria')
                                     if row['sample_id']=='gen_animal_000'))
                if policy=='available':
                    self.assertEqual(summary['coverage_0.6'],'')
                else:
                    self.assertEqual(float(summary['coverage_0.6']),0)

    def test_invalid_and_stale_results(self):
        sid='gen_animal_000'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json'
            record=self.record(sid,1)
            # Cached aggregate values are not trusted over the atomic answers.
            record['scores']['usable_score']=0
            p.write_text(json.dumps(record))
            status,scores,_,_=aggregate.load_result(p,self.questions[sid],sid,'gen/animal')
            self.assertEqual(status,'scored');self.assertEqual(scores['usable_score'],1)
            record['questions'][0]['criterion']='VQ-Integrity'
            p.write_text(json.dumps(record))
            self.assertEqual(aggregate.load_result(p,self.questions[sid],sid,'gen/animal')[0],'invalid_result')
            record=self.record(sid,1);record['question_record_sha256']='old'
            p.write_text(json.dumps(record))
            self.assertEqual(aggregate.load_result(p,self.questions[sid],sid,'gen/animal')[0],'stale_result')

    def test_group_mean_uses_sample_counts(self):
        rows=[{'Q':1,'included':True,'status':'scored'}]*9+[
            {'Q':0,'included':True,'status':'unsupported_capability'}]
        self.assertAlmostEqual(aggregate.summarize(rows)['score'],0.9)
        self.assertIsNone(aggregate.summarize([])['score'])


if __name__=='__main__':
    unittest.main()
