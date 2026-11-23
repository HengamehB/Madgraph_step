"""Run a saved plain-text model from particle definitions through ranked rates."""
import argparse
from pathlib import Path
from workflow_errors import error_category
import sys


def _history_page(ranking, *, size, cursor, cutoff):
    rows = ranking.rows
    if hasattr(rows, 'page'):
        return rows.page(limit=size, cursor=cursor, min_relative_rate=cutoff, include_uncalculated=True)
    # Small legacy API views remain displayable; durable workflows use SQL.
    if cursor:
        raise ValueError('Pagination cursors require a saved history catalogue')
    import itertools
    maximum = max((row['rate_pb'] for row in rows if row.get('rate_pb') is not None), default=0.)
    selected = (row for row in rows if row.get('rate_pb') is None or row['rate_pb'] >= cutoff * maximum)
    return {'rows': list(itertools.islice(selected, size)), 'next_cursor': None, 'total': len(rows), 'revision': None}


def _show_history(row):
    value, error = row.get('rate_pb'), row.get('error_pb')
    number = 'not calculated' if value is None else f'{value:.6g} +/- {error:.3g} pb' if error is not None else f'{value:.6g} pb (error unavailable)'
    print(f"{row.get('history_id', '')}  {row['production_id']}  {row['final_state']}  {number}  {row['history']}")
    if row.get('flags'):
        print('  ' + '; '.join(row['flags']))
    status = row.get('rate_status', row.get('status'))
    if value is None and status:
        print('  status: ' + status)


def _save_input_failure(input_file, destination, error):
    """Retain diagnostics even when parsing fails before Pipeline can be built."""
    import hashlib
    import re
    from pipeline_config import ROOT
    from pipeline_storage import RunStore, atomic_text, write_json
    source = Path(input_file).expanduser().resolve()
    raw = source.read_bytes() if source.is_file() else b""
    default_output = ROOT / 'output' if source.parent == (ROOT / 'input').resolve() else source.parent / 'output'
    base = Path(destination).expanduser().resolve() if destination else default_output
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._") or "model"
    key = hashlib.sha256(raw + str(source).encode()).hexdigest()[:20]
    folder = base / name / ("invalid-" + key)
    record = {"schema": "madgraph-step-run-v1", "status": "failed", "complete": False,
              "requested_complete": False, "input_file": str(source),
              "failures": [{"stage": "input", "type": type(error).__name__, "category": error_category(error), "message": str(error)}]}
    with RunStore(folder).lock():
        if source.is_file():
            atomic_text(folder / ("input" + source.suffix), raw.decode("utf-8", errors="replace"))
        write_json(folder / "manifest.json", record)
        write_json(folder / "validation.json", record["failures"][0])
        from report_view import write_report
        write_report(folder, record)
    print("Input diagnostics:", folder, file=sys.stderr)


def _setup(args, model_file):
    from pipeline_config import load_config
    from tex_inputs import setup
    config = load_config(args.config, model_file=model_file, output=args.output, read_model=False)
    result = setup(config.model_file)
    print('Saved inputs:', result.path)
    unresolved = result.assignments.get('_input_provenance', {}).get('unresolved', {})
    if unresolved:
        for key, reason in unresolved.items():
            print(f'{key}: {reason}', file=sys.stderr)
        print('Edit the unresolved inputs and repeat setup.', file=sys.stderr)
        return 1
    print('Run with: ./run.sh run', result.path)
    return 0


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "init", "setup", "validate", "run", "check", "configure-tools", "inputs", "ufo", "channels", "decays", "histories", "rates", "explicit-decays"))
    parser.add_argument("input_file", nargs="?", help="model path (defaults to .config); init defaults to input/particles.txt in the package")
    parser.add_argument("--input", dest="input_option", help="input file, as an alternative to the positional path")
    parser.add_argument("--model", help="init destination; defaults to my_model.txt beside the particle source")
    parser.add_argument("--config", help="optional machine tool configuration; defaults to the package .config")
    parser.add_argument("--output", help="output folder; relative paths start at the current directory")
    parser.add_argument("--selected", default="all", help="all or comma-separated production channel IDs")
    parser.add_argument("--show-all", action="store_true", help="print every ranking row; full tables are always saved")
    parser.add_argument('--page-size', type=int, default=100, help='history rows per page (1–1000; default 100)')
    parser.add_argument('--cursor', help='continue a saved history page without production integration')
    parser.add_argument('--report-page', action='store_true', help='save the selected bounded history page as HTML')
    parser.add_argument('--run-dir', help='saved run directory for an offline HTML report')
    args = parser.parse_intermixed_args(argv)
    if args.action == 'report':
        if not args.run_dir or args.input_file or args.input_option or args.model or args.output or args.config or args.cursor or args.report_page or args.show_all or args.selected != 'all':
            parser.error('Use report --run-dir "path/to/saved run" [--page-size N]')
        from report_view import refresh_report
        print('HTML report:', refresh_report(args.run_dir, limit=args.page_size))
        return 0
    if args.run_dir:
        parser.error('--run-dir applies only to report')
    if args.input_file and args.input_option:
        parser.error("Pass either the positional input or --input, not both.")
    if not 1 <= args.page_size <= 1000:
        parser.error('--page-size must be between 1 and 1000')
    if args.cursor and args.action not in {'histories', 'rates'}:
        parser.error('--cursor is available for histories and rates browsing')
    if args.show_all and args.cursor:
        parser.error('--show-all cannot be combined with --cursor')
    model_file = args.input_file or args.input_option
    if args.model is not None and args.action != 'init':
        parser.error('--model specifies the generated destination for init only; use --input to select a model.')
    if args.action == 'init':
        if args.output:
            parser.error('init writes a model file; use --model for its destination and edit output inside that file.')
        from model_initialization import initialize_model
        import shlex
        result = initialize_model(model_file, args.model)
        print('Replaced model:' if result.replaced else 'Created model:', result.path)
        print(f'{result.operator_count} catalogue operators; {result.enabled_count} canonical kinetic/mass terms enabled.')
        for key, reason in result.unresolved.items():
            print(f'Unresolved {key}: {reason}')
        print('Edit the generated coefficients and switches, then run:',
              shlex.join([str(Path(__file__).resolve().with_name('run.sh')), 'channels', str(result.path)]))
        print('Electroweak vacuum: 125 GeV Higgs and zero VEVs for all new scalars.')
        return 0
    if args.action == 'configure-tools':
        from tool_configuration import configure_tools
        configure_tools(args.config)
        return 0
    if args.action == "setup":
        return _setup(args, model_file)
    from pipeline_config import load_config
    try:
        config = load_config(args.config, model_file=model_file, output=args.output,
                             read_model=args.action != "check")
    except (ValueError, OSError) as exc:
        if args.action in {"run", "validate", "histories", "rates", "explicit-decays"}:
            try:
                failed_input = model_file if model_file is not None else load_config(
                    args.config, read_model=False).model_file
                _save_input_failure(failed_input, args.output, exc)
            except (ValueError, OSError, RuntimeError) as storage_error:
                print("Could not save input diagnostics:", storage_error, file=sys.stderr)
        raise
    if args.action == "inputs":
        if config.model_file.suffix.lower() == '.txt':
            from tex_inputs import read_inputs
            from model_inputs_text import show_inputs
            show_inputs(read_inputs(config.model_file))
        else:
            from pipeline import load_model_definition
            load_model_definition(config.model_file).show_inputs()
        return 0
    if args.action == "check":
        from portable_runtime import check_environment
        report = check_environment(config.runtime)
        for check in report["checks"]:
            print(f"{'OK' if check['ok'] else 'MISSING'} {check['name']}: {check['detail']}")
        return 0 if report["ok"] else 1
    from pipeline import Pipeline
    project = Pipeline(config)
    if args.action == "validate":
        prepared = project.validate()
        print("Validated physical model:", prepared.model_name)
        print("Validation and model files:", project.run_directory)
        return 0
    if args.action == "ufo":
        print(project.export_ufo().path)
    elif args.action == "channels":
        channels = project.channels(show=False)
        for row in channels:
            print(row.id, row.process)
        print(f"{len(channels)} separate tree-level production channels.")
        notice = project.production_notice()
        if notice:
            print(notice)
    elif args.action == "decays":
        decays = project.decays(show=False)
        for row in decays.total_width_rows():
            print(f"{row['name']}: estimated total width {row['width_gev']:.6g} GeV; omitted reported probability {row['omitted_probability']:.3g}")
        for row in decays.two_body:
            print(f"{row['process']}: {row['width_gev']:.6g} GeV, BR={row['branching_ratio']:.6g}")
        print(decays.approximation)
        for notice in decays.coverage_notices:
            print(notice)
    elif args.action == 'explicit-decays':
        results = project.explicit_decays(show=False)
        for row in results:
            print(row['history_id'], row.get('cross_section_pb'), '+/-', row.get('error_pb'), 'pb', row['status'])
        print('Explicit decay-chain matrix-element rates are separate from factorized production × branching ratios.')
    else:
        selected = "all" if args.selected == "all" else [s.strip() for s in args.selected.split(",") if s.strip()]
        if args.cursor:
            ranking = project.saved_histories()
        elif args.action == 'histories':
            ranking = project.histories(show=False)
        else:
            action = project.run if args.action == 'run' else project.rates
            ranking = action(selected, show=False)
        page = _history_page(ranking, size=args.page_size, cursor=args.cursor,
                             cutoff=config.ranking['min_relative_rate'])
        count = 0
        for row in ranking.rows if args.show_all else page['rows']:
            _show_history(row)
            count += 1
        print("Separate narrow-width estimates; do not sum overlapping production histories.")
        hidden = len(ranking.rows) - count
        if hidden:
            print(f"{hidden} further histories retained in the saved catalogue; use --show-all to stream every row.")
        if page.get('next_cursor'):
            import shlex
            command = [str(Path(__file__).resolve().with_name('run.sh')), 'histories', str(config.model_file), '--cursor', page['next_cursor'],
                       '--page-size', str(args.page_size), '--report-page']
            if args.output:
                command += ['--output', str(config.output)]
            if args.config:
                command += ['--config', str(config.path)]
            print('Next page:', shlex.join(command))
        if args.report_page:
            from pipeline_reports import write_history_page
            print('History page:', write_history_page(project, page))
        for channel, probability in ranking.omitted_probability.items():
            if probability is not None and probability > 1e-8:
                print(f"{channel}: uncovered decay probability {probability:.6g}")
        if ranking.uncomputed_production_ids:
            print(f"{len(ranking.uncomputed_production_ids)} production channels were not integrated.")
        for notice in ranking.notices:
            print(notice)
    print("Run files:", project.run_directory)
    complete = getattr(project, 'requested_complete', getattr(project, 'complete', True))
    if args.action == 'explicit-decays':
        complete = getattr(project, 'selected_integrations_complete', False)
    elif args.action == 'histories' or args.cursor:
        complete = getattr(project, 'discovery_complete', True) and getattr(project, 'export_complete', True)
    if complete is False and args.action in {"run", "rates", "histories", "explicit-decays"}:
        print("Run incomplete; successful results were saved. Repeat the same command to resume.", file=sys.stderr)
        failures = getattr(project, "failures", ())
        if isinstance(failures, dict):
            failures = [dict(channel=key, message=value) for key, value in failures.items()]
        for failure in failures:
            print(f"{failure.get('channel', failure.get('stage', 'pipeline'))}: {failure.get('message', failure)}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    from workflow_progress import report_progress
    with report_progress():
        return _main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print("Cancelled. Saved checkpoints are retained; setup leaves the original input intact.", file=sys.stderr)
        raise SystemExit(130)
