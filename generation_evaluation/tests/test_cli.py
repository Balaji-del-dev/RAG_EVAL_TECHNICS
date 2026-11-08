from generation_eval.cli import main

from .conftest import SAMPLES_FILE


def test_tree(capsys):
    assert main(["tree"]) == 0
    out = capsys.readouterr().out
    assert "|-- 1. Faithfulness" in out and "`-- 8. ROUGE" in out


def test_metrics_table(capsys):
    assert main(["metrics"]) == 0
    out = capsys.readouterr().out
    assert "Answer Completeness" in out and "deepeval / custom [deepeval]" in out


def test_run_text_metrics(capsys, tmp_path):
    report = tmp_path / "report.json"
    code = main(["run", str(SAMPLES_FILE), "--metrics", "exact_match,bleu,rouge", "--no-track", "--out", str(report)])
    out = capsys.readouterr().out
    assert code == 1  # only low-priority metrics ran, and Exact Match fails the paraphrase
    assert "[PASS] hr-01-correct-complete" in out
    assert "5 overlapping reference unigrams / 11 total reference unigrams" in out  # hr-03-incomplete
    assert '"total": 5' in report.read_text()
