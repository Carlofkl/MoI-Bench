"""Run the real file pipeline with temporary images and an offline judge."""
import contextlib
import csv
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import eval_outputs as evaluator
import aggregate_scores as aggregate


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Install requirements.txt for image pipeline checks')
class PipelineTests(unittest.TestCase):
    def test_images_to_results_to_profiles(self):
        from PIL import Image
        calls = []
        class OfflineJudge:
            def __init__(self, **kwargs):
                pass
            def answer(self, system, prompt, images):
                calls.append((prompt, [image.getpixel((0,0)) for image in images]))
                return 0.6, '{"score":0.6}'
        samples=json.loads((ROOT/'data/samples.json').read_text())['items']
        ids=['gen_animal_000','images/ref_r1_animal_000.jpg','images/ref_b1r3_mixed_000.jpg']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            shutil.copy(ROOT/'data/samples.json',data/'samples.json')
            model=root/'outputs/fixture'
            for sid in ids:
                row=samples[sid]
                inputs=[] if row['task'].startswith('gen/') else [sid,*row.get('reference_images',[])]
                for index, relative in enumerate(inputs):
                    p=data/relative;p.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(12+index,19),(index*40,0,0)).save(p,format='PNG')
                directory=model/row['task']/Path(sid).stem;directory.mkdir(parents=True)
                (directory/'prompt.json').write_text(json.dumps({'entry_id':sid,'input_images':inputs}))
                Image.new('RGB',(17,11),(200,0,0)).save(directory/'seed_0.png')
            argv=['eval_outputs.py','--model-tag','fixture','--judge','openai','--judge-model','offline',
                  '--output-root',str(root/'outputs'),'--eval-json',str(data/'samples.json'),
                  '--multi-image-support','yes']
            with patch.object(sys,'argv',argv),patch.object(evaluator,'OpenAIJudge',OfflineJudge),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(),0)
            self.assertTrue(any('reference image' in p and 'no separate base image' in p for p,images in calls if len(images)==2))
            self.assertTrue(any(images==[(0,0,0),(40,0,0),(80,0,0),(120,0,0),(200,0,0)] for _,images in calls))
            count=len(calls)
            with patch.object(sys,'argv',argv),patch.object(evaluator,'OpenAIJudge',OfflineJudge),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(),0)
            self.assertEqual(len(calls),count,'resuming must not rejudge unchanged results')
            argv=['aggregate_scores.py','--models','fixture','--judge-tag','offline',
                  '--output-root',str(root/'outputs'),'--results-dir',str(root/'results'),'--denominator','available']
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(aggregate.main(),0)
            with (root/'results/summary.csv').open() as handle:summary=next(csv.DictReader(handle))
            self.assertEqual(summary['n_scored'],'3')
            self.assertAlmostEqual(float(summary['score']),0.6)
            self.assertEqual(summary['aggregation_status'],'incomplete')
