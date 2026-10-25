"""Local input and CLI regressions; fixtures require no external data or model."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import AsyncMock, patch

from datasets import Dataset, DatasetDict, disable_progress_bars

import longswe_slo as bench


disable_progress_bars()


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def options(self, path, **overrides):
        values = dict(dataset=str(path), split='test', prompt_column=None,
                      tokenizer='', trust_remote_code=False,
                      dataset_kind='swebench', issue_only=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def parquet(self, relative, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).to_parquet(str(path))
        return path

    def test_saved_longswe_dataset_still_selects_test_split(self):
        path = self.root / '128K'
        DatasetDict({
            'test': Dataset.from_dict({'prompt': ['repository context and issue']}),
            'train': Dataset.from_dict({'prompt': ['do not use train']}),
        }).save_to_disk(str(path))
        prompts, info = bench.load_prompts(self.options(path, dataset_kind='longswebench'))
        self.assertEqual(prompts, ['repository context and issue'])
        self.assertEqual(info['source']['format'], 'save_to_disk')
        self.assertEqual(info['column'], 'prompt')

    def test_saved_single_dataset_is_supported(self):
        path = self.root / 'saved'
        Dataset.from_dict({'text': ['prepared input']}).save_to_disk(str(path))
        prompts, _ = bench.load_prompts(self.options(path))
        self.assertEqual(prompts, ['prepared input'])

    def test_downloaded_parquet_shards_exclude_train_and_dev(self):
        self.parquet('data/test-00001-of-00002.parquet', [
            {'text': 'second issue plus code', 'patch': 'GOLD ANSWER'},
        ])
        self.parquet('data/test-00000-of-00002.parquet', [
            {'text': 'first issue plus code', 'patch': 'GOLD ANSWER'},
        ])
        self.parquet('data/train-00000-of-00001.parquet', [{'text': 'TRAIN'}])
        self.parquet('data/dev-00000-of-00001.parquet', [{'text': 'DEV'}])
        with patch('socket.create_connection', side_effect=AssertionError('network access')):
            prompts, info = bench.load_prompts(self.options(self.root))
        self.assertEqual(prompts, ['first issue plus code', 'second issue plus code'])
        self.assertEqual(info['input_mode'], 'prepared_prompt')
        self.assertEqual(len(info['source']['files']), 2)
        self.assertTrue(all('/test-' in name for name in info['source']['files']))

    def test_split_subdirectory_and_explicit_file(self):
        path = self.parquet('data/test/0000.parquet', [{'prompt': 'prepared'}])
        for source in (self.root, path):
            with self.subTest(source=source):
                prompts, _ = bench.load_prompts(self.options(source))
                self.assertEqual(prompts, ['prepared'])

    def test_json_array_and_jsonl(self):
        for extension, contents in (
            ('json', json.dumps([{'text': 'json prompt'}])),
            ('jsonl', json.dumps({'text': 'json prompt'}) + '\n'),
        ):
            with self.subTest(extension=extension):
                path = self.root / f'{extension}/test.{extension}'
                path.parent.mkdir()
                path.write_text(contents)
                prompts, _ = bench.load_prompts(self.options(path.parent))
                self.assertEqual(prompts, ['json prompt'])

    def test_missing_split_never_falls_back_to_train(self):
        self.parquet('data/train-00000-of-00001.parquet', [{'text': 'TRAIN'}])
        with self.assertRaisesRegex(ValueError, 'No local.*test'):
            bench.load_prompts(self.options(self.root))

    def test_nondefault_split_is_respected(self):
        self.parquet('data/test.parquet', [{'text': 'TEST'}])
        self.parquet('data/dev.parquet', [{'text': 'DEV'}])
        prompts, _ = bench.load_prompts(self.options(self.root, split='dev'))
        self.assertEqual(prompts, ['DEV'])

    def test_mixed_formats_are_rejected(self):
        self.parquet('test.parquet', [{'text': 'parquet'}])
        (self.root / 'test.jsonl').write_text('{"text": "json"}\n')
        with self.assertRaisesRegex(ValueError, 'one local data format'):
            bench.load_prompts(self.options(self.root))

    def test_raw_swebench_requires_explicit_issue_only(self):
        path = self.parquet('data/test.parquet', [{
            'problem_statement': 'Fix this bug', 'patch': 'GOLD ANSWER',
            'test_patch': 'TEST ANSWER', 'repo': 'owner/repo',
        }])
        with self.assertRaisesRegex(ValueError, 'SWE-bench_bm25_40K'):
            bench.load_prompts(self.options(path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prompts, info = bench.load_prompts(self.options(path, issue_only=True))
        self.assertTrue(any('not the approximately 40K' in str(item.message) for item in caught))
        self.assertEqual(prompts, ['Fix this bug'])
        self.assertEqual(info['input_mode'], 'issue_only')
        self.assertIn('warnings', info)

    def test_explicit_problem_statement_column_is_also_marked_issue_only(self):
        path = self.parquet('test.parquet', [{'problem_statement': 'issue'}])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, info = bench.load_prompts(self.options(path, prompt_column='problem_statement'))
        self.assertTrue(any('Issue-only SWE-Bench' in str(item.message) for item in caught))
        self.assertEqual(info['input_mode'], 'issue_only')

    def test_ambiguous_and_missing_columns_are_actionable(self):
        path = self.parquet('test.parquet', [{'text': 'text', 'prompt': 'prompt'}])
        with self.assertRaisesRegex(ValueError, '--prompt-column'):
            bench.load_prompts(self.options(path))
        prompts, _ = bench.load_prompts(self.options(path, prompt_column='text'))
        self.assertEqual(prompts, ['text'])
        with self.assertRaisesRegex(ValueError, 'Column.*missing.*not found'):
            bench.load_prompts(self.options(path, prompt_column='missing'))

    def test_invalid_prompts_are_rejected(self):
        for index, values in enumerate(([], [''], ['valid', None], [123])):
            with self.subTest(values=values):
                path = self.root / f'invalid-{index}.json'
                path.write_text(json.dumps([{'text': value} for value in values]))
                with self.assertRaises(ValueError):
                    bench.load_prompts(self.options(path))

    def test_missing_local_path_does_not_resolve_hub_id(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Local dataset does not exist'):
            bench.load_prompts(self.options(self.root / 'princeton-nlp/SWE-bench'))

    def test_shared_entry_point_dispatches_to_the_same_search(self):
        path = self.parquet('test.parquet', [{'text': 'issue plus code'}])
        runner = AsyncMock()
        with patch.object(bench, 'main_async', runner), contextlib.redirect_stdout(io.StringIO()):
            bench.cli_main([
                '--dataset', str(path), '--tokenizer', '', '--output-len', '1024',
                '--output-dir', str(self.root / 'results'),
            ], dataset_kind='swebench')
        runner.assert_awaited_once()
        args, prompts, info = runner.call_args.args
        self.assertEqual((args.budget_seconds, args.start_concurrency, args.max_concurrency),
                         (1200, 8, 64))
        self.assertEqual((args.ttft_slo, args.tps_slo, args.output_len), (10, 15, 1024))
        self.assertEqual(args.confirm_requests, 100)
        self.assertFalse(args.allow_prefix_reuse)
        self.assertFalse(args.keep_cache)
        self.assertEqual(prompts, ['issue plus code'])
        self.assertEqual(info['dataset_kind'], 'swebench')

    def test_inspect_only_cli_uses_local_tokenizer_without_http_or_output(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'fix': 1, 'code': 2}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_dir = self.root / 'tokenizer'
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(tokenizer_dir)
        path = self.parquet('data/test.parquet', [{'text': 'fix code'}])
        output = self.root / 'must-not-exist'
        result = subprocess.run([
            sys.executable, str(Path(bench.__file__).with_name('swebench_slo.py')),
            '--dataset', str(path.parent.parent), '--tokenizer', str(tokenizer_dir),
            '--base-url', 'http://127.0.0.1:1', '--output-len', '1024',
            '--output-dir', str(output), '--inspect-only',
        ], env={**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1'}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual(info['dataset_kind'], 'swebench')
        self.assertEqual(info['base_input_tokens']['mean'], 2)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

