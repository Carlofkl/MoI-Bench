#!/usr/bin/env python3
"""Aggregate MoI-Eval records into all tabular profiles without API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from eval_outputs import PROMPT_TEMPLATES_HASH, load_questions, stable_hash
from moi.scoring import EVAL_PROTOCOL_VERSION, GROUP_ORDER, aggregate_question_scores, zero_question_results

ROOT = Path(__file__).resolve().parents[1]
VALID = {'scored', 'unsupported_capability'}
RESOLVED = VALID | {'execution_failed'}
STATUSES = ('scored', 'unsupported_capability', 'execution_failed', 'missing_output',
            'missing_evaluation', 'invalid_result', 'stale_result')
COUNT_FIELDS = ['n_expected', 'n_included', *['n_' + s for s in STATUSES]]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_tag(value):
    if not value or Path(value).name != value or value in {'.', '..'}:
        raise ValueError(f'Expected a single directory name, got {value!r}')
    return value


def load_result(path, question, sample_id, task, judge_model=None):
    """Validate identity and provenance; recompute scores from atomic answers."""
    try:
        record = read_json(path)
        if (record.get('protocol_version') != EVAL_PROTOCOL_VERSION
                or record.get('question_record_sha256') != stable_hash(question)
                or record.get('judge', {}).get('prompt_templates_hash') != PROMPT_TEMPLATES_HASH
                or (judge_model and record.get('judge', {}).get('model') != judge_model)):
            return 'stale_result', None, 'Protocol, questions or judge configuration differs', None
        if record['entry_id'] != sample_id or record['task'] != task or record['seed'] != 0:
            raise ValueError('Sample identity, task or seed differs')
        status = record['status']
        if status not in VALID:
            raise ValueError('Unexpected evaluation status')
        expected = {q['question_id']: q['criterion'] for q in question['questions']}
        answers = record['questions']
        actual = {q['question_id']: q['criterion'] for q in answers}
        if len(answers) != len(expected) or actual != expected:
            raise ValueError('Question IDs or criteria differ')
        scores = aggregate_question_scores(answers, question['group_weights'])
        if status == 'unsupported_capability' and any(scores['per_question'].values()):
            raise ValueError('Unsupported result contains nonzero question scores')
        model = record['judge']['model']
        if not isinstance(model, str) or not model:
            raise ValueError('Missing judge model')
        return status, scores, record.get('skip_reason', ''), model
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return 'invalid_result', None, str(exc), None


def summarize(rows, value='Q'):
    counts = Counter(r['status'] for r in rows)
    included = [r[value] for r in rows if r['included']]
    unresolved = any(counts[s] for s in STATUSES if s not in RESOLVED)
    return {'score': mean(included) if included else None,
            'aggregation_status': ('not_applicable' if not rows else
                                   'no_included_samples' if not included else
                                   'incomplete' if unresolved else 'complete'),
            'n_expected': len(rows), 'n_included': len(included),
            **{'n_' + s: counts[s] for s in STATUSES}}


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def aggregate_model(model, args, samples, targets, questions, overrides):
    root = args.output_root / model
    if not root.is_dir():
        raise ValueError(f'Model output directory does not exist: {root}')
    rows, criterion_rows, provenance, judges = [], [], {}, set()
    for sid, sample in samples.items():
        task = sample['task']
        kind = 'gen' if task.startswith('gen/') else 'edit'
        if args.mode != 'all' and args.mode != kind:
            continue
        if Path(task).is_absolute() or any(part in {'.', '..'} for part in Path(task).parts):
            raise ValueError(f'Unsafe task: {task}')
        question = questions[sid]
        if question['task'] != task or question['instruction'] != sample['instruction'] or targets[sid]['task'] != task:
            raise ValueError(f'Metadata mismatch: {sid}')
        result_path = root / 'eval' / args.judge_tag / task / Path(sid).stem / 'seed_0.json'
        result, judge_model = None, None
        reason = ''
        override = overrides.get(sid)
        if override and result_path.exists():
            raise ValueError(f'Status declaration conflicts with an existing result: {sid}')
        if override:
            status, reason = override['status'], override['reason']
            if status == 'unsupported_capability':
                result = aggregate_question_scores(zero_question_results(question, reason), question['group_weights'])
        elif result_path.exists():
            provenance[str(result_path.relative_to(args.output_root))] = digest(result_path)
            status, result, reason, judge_model = load_result(result_path, question, sid, task, args.judge_model)
        else:
            image = root / task / Path(sid).stem / 'seed_0.png'
            status = 'missing_evaluation' if image.is_file() else 'missing_output'
        if judge_model:
            judges.add(judge_model)
        included = args.denominator == 'fixed' or status in VALID
        scores = result or aggregate_question_scores(zero_question_results(question, status), question['group_weights'])
        labels = targets[sid]
        row = {'model': model, 'judge': args.judge_tag, 'sample_id': sid, 'task': task,
               'operation': labels['operation'], 'target': labels['target'],
               'status': status, 'reason': reason, 'included': included,
               'Q': scores['usable_score'] if included else None}
        rows.append(row)
        for level, values in [('criterion', scores['criterion_scores']), ('group', scores['group_scores'])]:
            for criterion, score in values.items():
                criterion_rows.append({**row, 'level': level, 'criterion': criterion,
                                       'score': score if included else None})
    if len(judges) > 1:
        raise ValueError(f'Mixed judge models under tag {args.judge_tag}: {sorted(judges)}')
    return rows, criterion_rows, provenance, judges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output')
    parser.add_argument('--results-dir', type=Path, help='Default: results/<judge>/<mode>/<denominator>')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--judge-tag', required=True)
    parser.add_argument('--judge-model', help='Optional expected judge model; mismatches are stale')
    parser.add_argument('--mode', choices=['all', 'gen', 'edit'], default='all')
    parser.add_argument('--denominator', choices=['fixed', 'available'], default='available',
                        help=('available: scored and unsupported samples only; '
                              'fixed: every expected sample, including unresolved cases, counts zero'))
    parser.add_argument('--status-file', type=Path, help='Explicit unsupported/failed sample declarations; see docs/AGGREGATION.md')
    args = parser.parse_args()
    try:
        valid_tag(args.judge_tag)
        for model in args.models:
            valid_tag(model)
        if len(args.models) != len(set(args.models)):
            raise ValueError('Duplicate model tags')
        paths = {name: args.data_root / filename for name, filename in
                 [('samples', 'samples.json'), ('targets', 'targets.json'),
                  ('questions', 'questions.jsonl'), ('rubrics', 'rubrics.json')]}
        samples, targets = read_json(paths['samples'])['items'], read_json(paths['targets'])['items']
        questions = load_questions(paths['questions'])
        rubrics = read_json(paths['rubrics'])
        if set(samples) != set(targets) or set(samples) != set(questions):
            raise ValueError('Sample IDs differ across metadata files')
        overrides = read_json(args.status_file) if args.status_file else {}
        if not set(overrides) <= set(args.models):
            raise ValueError('Status file contains unselected model tags')
        for model, declarations in overrides.items():
            if not set(declarations) <= set(samples):
                raise ValueError(f'Unknown sample IDs in status file: {model}')
            for sid, declaration in declarations.items():
                if declaration['status'] not in {'unsupported_capability', 'execution_failed'} or not declaration.get('reason', '').strip():
                    raise ValueError(f'Invalid status/reason for {sid}')
        exports = defaultdict(list)
        records, judge_models = {}, set()
        for model in args.models:
            rows, criteria, hashes, judges = aggregate_model(model, args, samples, targets, questions, overrides.get(model, {}))
            records.update(hashes); judge_models.update(judges)
            exports['sample_scores'].extend(rows)
            exports['sample_criteria'].extend(criteria)
            base = {'model': model, 'judge': args.judge_tag, 'denominator': args.denominator}
            tasks = sorted({r['task'] for r in rows})
            task_summaries = []
            for filename, field in [('subtask_scores', 'task'), ('operation_scores', 'operation'), ('target_scores', 'target')]:
                for key in sorted({r[field] for r in rows}):
                    stat = summarize([r for r in rows if r[field] == key])
                    exports[filename].append({**base, field: key, **stat})
                    if field == 'task':
                        task_summaries.append(stat)
            # Export the full observed-axis grid; empty combinations remain N/A.
            for operation in sorted({r['operation'] for r in rows}):
                for target in sorted({r['target'] for r in rows}):
                    exports['operation_target_scores'].append({**base, 'operation': operation, 'target': target,
                        **summarize([r for r in rows if r['operation'] == operation and r['target'] == target])})
            groups = {}
            for level, names in [('criterion', rubrics['criterion_order']), ('group', GROUP_ORDER)]:
                for name in names:
                    stat = summarize([r for r in criteria if r['level'] == level and r['criterion'] == name], 'score')
                    exports['criteria_scores'].append({**base, 'level': level, 'criterion': name, **stat})
                    if level == 'group':
                        groups[name] = stat['score']
            # Unknown subtask means make coverage provisional rather than silently becoming failures.
            coverage = (sum(s['score'] >= 0.6 for s in task_summaries) / len(tasks)
                        if tasks and all(s['score'] is not None for s in task_summaries) else None)
            exports['summary'].append({**base, **summarize(rows), **groups,
                                      'coverage_0.6': coverage, 'n_subtasks': len(tasks)})
        if len(judge_models) > 1:
            raise ValueError(f'Mixed judge models across selected results: {sorted(judge_models)}')
        destination = args.results_dir or ROOT / 'results' / args.judge_tag / args.mode / args.denominator
        destination.mkdir(parents=True, exist_ok=True)
        common = ['model', 'judge', 'denominator']
        stats = ['score', 'aggregation_status', *COUNT_FIELDS]
        sample_fields = ['model', 'judge', 'sample_id', 'task', 'operation', 'target', 'status', 'reason', 'included', 'Q']
        fields = {
            'summary': common + stats + ['IF', 'VQ', 'DP', 'coverage_0.6', 'n_subtasks'],
            'sample_scores': sample_fields,
            'sample_criteria': sample_fields + ['level', 'criterion', 'score'],
            'subtask_scores': common + ['task'] + stats,
            'operation_scores': common + ['operation'] + stats,
            'target_scores': common + ['target'] + stats,
            'criteria_scores': common + ['level', 'criterion'] + stats,
            'operation_target_scores': common + ['operation', 'target'] + stats,
        }
        for name, columns in fields.items():
            write_csv(destination / (name + '.csv'), exports[name], columns)
        metadata = {'protocol_version': EVAL_PROTOCOL_VERSION, 'denominator': args.denominator,
                    'mode': args.mode, 'models': args.models, 'judge_tag': args.judge_tag,
                    'judge_models': sorted(judge_models), 'expected_judge_model': args.judge_model,
                    'input_sha256': {k: digest(p) for k, p in paths.items()},
                    'result_sha256': records, 'status_file_sha256': digest(args.status_file) if args.status_file else None,
                    'note': 'Empty CSV scores mean N/A. Incomplete profiles are not complete benchmark measurements.'}
        (destination/'aggregation_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for summary in exports['summary']:
            print(f"{summary['model']}: Q={summary['score']} included={summary['n_included']}/{summary['n_expected']} status={summary['aggregation_status']}")
        print(f'Wrote 8 CSV files and aggregation_metadata.json to {destination}')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Aggregation failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
