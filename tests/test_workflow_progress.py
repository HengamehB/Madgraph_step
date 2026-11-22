"""Progress remains scoped to CLI invocations and never changes result stdout."""
from contextlib import redirect_stderr, redirect_stdout
import io
import re
import threading
import unittest
from unittest.mock import patch

import pipeline_cli
from workflow_progress import format_count, message, report_progress, stage


class WorkflowProgressTests(unittest.TestCase):
    def test_enormous_counts_use_bounded_integer_only_diagnostics(self):
        self.assertEqual(format_count(1819221039),'1,819,221,039')
        self.assertEqual(format_count(0),'0')
        huge = 10**6000
        text = format_count(huge)
        self.assertLess(len(text),80)
        match = re.fullmatch(r'between 10\^([\d,]+) and 10\^([\d,]+)',text)
        self.assertIsNotNone(match)
        lower, upper = (int(value.replace(',','')) for value in match.groups())
        self.assertLessEqual(10**lower,huge)
        self.assertLess(huge,10**upper)
        output = io.StringIO()
        with report_progress(stream=output):
            with stage('Huge discovery',total=huge,unit='combinations') as progress:
                progress.update(huge)
        self.assertIn(text,output.getvalue())
        self.assertLess(len(output.getvalue()),400)
        self.assert_no_worker_threads()

    def test_library_calls_are_silent_and_stdout_is_untouched(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            message('Unreported summary')
            with stage('Unreported library work', total=5) as progress:
                progress.update(3, detail='quiet counters')
                print('C0001 p p > x x')
        self.assertEqual(out.getvalue(), 'C0001 p p > x x\n')
        self.assertEqual(err.getvalue(), '')

    def test_counter_updates_are_reported_without_database_or_log_reads(self):
        pulse = threading.Event()

        class SignallingStream(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if 'still running' in value and '1,000/5,000 combinations; 900 saved' in value:
                    pulse.set()
                return result

        err = SignallingStream()
        with report_progress(stream=err,interval=.001):
            message('Expected 5,000 combinations')
            with stage('Histories',total=5000,unit='combinations') as progress:
                progress.update(1000,detail='900 saved')
                self.assertTrue(pulse.wait(1))
                progress.update(5000,detail='4,500 saved')
        messages = err.getvalue()
        self.assertIn('Expected 5,000 combinations',messages)
        self.assertIn('Histories... [0/5,000 combinations]',messages)
        self.assertIn('[5,000/5,000 combinations; 4,500 saved]',messages)
        self.assert_no_worker_threads()

    def test_short_stages_report_initial_and_final_counts(self):
        err = io.StringIO()
        with report_progress(stream=err):
            with stage('Export',total=0,unit='rows') as progress:
                progress.update(3,total=3)
        self.assertIn('Export... [0/0 rows]',err.getvalue())
        self.assertIn('[3/3 rows]',err.getvalue())
        self.assertNotIn('still running',err.getvalue())

    def test_cli_scope_reports_log_path_and_restores_quiet_library_calls(self):
        out, err = io.StringIO(), io.StringIO()

        def operation(argv):
            self.assertEqual(argv, ['channels', 'my_model.txt'])
            with stage('Discovering tree-level production channels', log='run folder/channels.log'):
                print('C0001 p p > x x')
            return 0

        with redirect_stdout(out), redirect_stderr(err), patch.object(pipeline_cli, '_main', operation):
            self.assertEqual(pipeline_cli.main(['channels', 'my_model.txt']), 0)
            with stage('Unreported later library work'):
                pass
        self.assertEqual(out.getvalue(), 'C0001 p p > x x\n')
        self.assertIn('Discovering tree-level production channels... (log: run folder/channels.log)', err.getvalue())
        self.assertIn('Discovering tree-level production channels: done (', err.getvalue())
        self.assertNotIn('Unreported', err.getvalue())
        self.assert_no_worker_threads()

    def test_nested_scopes_restore_the_outer_reporter(self):
        outer, inner = io.StringIO(), io.StringIO()
        with report_progress(stream=outer):
            with stage('Outer'):
                with report_progress(stream=inner):
                    with stage('Inner'):
                        pass
                with stage('Outer again'):
                    pass
        self.assertIn('Outer: done (', outer.getvalue())
        self.assertIn('Outer again: done (', outer.getvalue())
        self.assertNotIn('Inner', outer.getvalue())
        self.assertIn('Inner: done (', inner.getvalue())
        self.assertNotIn('Outer', inner.getvalue())
        self.assert_no_worker_threads()

    def test_failure_and_interruption_propagate_and_stop_workers(self):
        for exception, status in ((ValueError('bad input'), 'failed'),
                                  (KeyboardInterrupt(), 'interrupted'),
                                  (EOFError(), 'interrupted')):
            with self.subTest(exception=type(exception).__name__):
                err = io.StringIO()
                with self.assertRaises(type(exception)), report_progress(stream=err):
                    with stage('Work'):
                        raise exception
                self.assertIn(f'Work: {status} (', err.getvalue())
                self.assert_no_worker_threads()
                with redirect_stderr(err), stage('Must be quiet after failure'):
                    pass
                self.assertNotIn('Must be quiet', err.getvalue())

    def test_heartbeat_stops_before_final_message(self):
        pulse = threading.Event()

        class SignallingStream(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if 'still running' in value:
                    pulse.set()
                return result

        err = SignallingStream()
        with report_progress(stream=err, interval=0.001):
            with stage('Long work'):
                self.assertTrue(pulse.wait(1), 'The worker did not report progress')
        lines = err.getvalue().splitlines()
        self.assertIn('Long work...', lines[0])
        self.assertTrue(any('still running (' in line and 's elapsed)' in line for line in lines))
        self.assertTrue(lines[-1].startswith('Long work: done ('))
        self.assert_no_worker_threads()

    def test_invalid_interval_does_not_enable_reporting(self):
        err = io.StringIO()
        with self.assertRaises(ValueError), report_progress(stream=err, interval=0):
            pass
        with redirect_stderr(err), stage('Still quiet'):
            pass
        self.assertEqual(err.getvalue(), '')

    def assert_no_worker_threads(self):
        self.assertFalse([thread for thread in threading.enumerate()
                          if thread.name == 'madgraph-step-progress'])


if __name__ == '__main__':
    unittest.main()
