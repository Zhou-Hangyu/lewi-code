"""The documents: their commands parse, their Python examples compile, and the quickstart runs."""
import re
import runpy
import shlex
from pathlib import Path

import pytest

from csfrbench import cli

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "csfrbench" / "README.md", ROOT / "lewi" / "README.md",
             ROOT / "docs" / "REPRODUCE.md", ROOT / "CONTRIBUTING.md"]


def blocks(language: str, document: Path) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", document.read_text(), flags=re.DOTALL)


def commands_of(document: Path) -> list[str]:
    """The `csfrbench` commands of the shell examples, of the tables and of the text of a document."""
    found = [line.split("#")[0].strip() for block in blocks("bash", document) for line in block.splitlines()
             if line.strip().startswith("csfrbench ")]
    return found + re.findall(r"`(csfrbench [^`]*)`", document.read_text())


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_commands_parse(document):
    parser = cli.build_parser()
    verbs = next(action for action in parser._actions if action.dest == "command").choices
    for command in commands_of(document):
        if "..." in command or "<" in command or "[" in command:
            continue
        words = shlex.split(command)[1:]
        if not any(word.startswith("-") for word in words):  # a verb named in the text, as `csfrbench plan / run`
            assert all(word in verbs for word in words if word != "/"), command
            continue
        assert callable(parser.parse_args(words).func), command


def test_documents_name_enough_commands():
    assert len(commands_of(ROOT / "AGENTS.md")) >= 10
    assert len(commands_of(ROOT / "docs" / "REPRODUCE.md")) >= 8


def test_commands_of_the_module_description_parse():
    parser = cli.build_parser()
    lines = [line.strip() for line in cli.__doc__.splitlines() if line.strip().startswith("csfrbench ")]
    assert len(lines) == 10
    for line in lines:
        words = shlex.split(re.sub(r"\[[^\]]*\]", "", line.split("   ")[0]))[1:]
        assert callable(parser.parse_args(words).func), line


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_examples_compile(document):
    for example in blocks("python", document):
        compile(example, str(document), "exec")


@pytest.mark.slow
def test_readme_examples_run(capsys):
    """The Python examples of README.md, as a reader copies them (about a minute)."""
    for example in blocks("python", ROOT / "README.md"):
        exec(compile(example, "README.md", "exec"), {"__name__": "__main__"})
    assert "AULC" in capsys.readouterr().out


@pytest.mark.slow
def test_quickstart_runs(capsys):
    """examples/quickstart.py, as a reader runs it (under a minute)."""
    runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "test NMSE" in out and "AULC" in out and "LeWi" in out


def test_versions_agree():
    """One version for the distribution and both packages."""
    import lewi
    import csfrbench
    declared = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), flags=re.MULTILINE).group(1)
    assert declared == csfrbench.__version__ == lewi.__version__
    assert csfrbench.BENCHMARK_VERSION == "1.5"
