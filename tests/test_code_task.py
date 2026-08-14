"""Code task: reward correctness + the sandbox actually containing things."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mlx_rl.tasks.base import Example, get_task

HAVE_SEATBELT = shutil.which("sandbox-exec") is not None
needs_seatbelt = pytest.mark.skipif(not HAVE_SEATBELT,
                                    reason="sandbox-exec not available")


def _example(test_list, test_imports=()):
    return Example(messages=[], meta={"task_id": 0, "test_list": test_list,
                                      "test_imports": list(test_imports)})


def _completion(body):
    return f"reasoning...\n```python\n{body}\n```"


@pytest.fixture(scope="module")
def task():
    if not HAVE_SEATBELT:
        pytest.skip("sandbox-exec not available")
    return get_task("code")


def test_split_is_deterministic_and_disjoint():
    t = get_task("code", sandbox=HAVE_SEATBELT)
    eval_ids = {r["task_id"] for r in t._eval}
    train_ids = {r["task_id"] for r in t._train}
    assert len(eval_ids) == 80
    assert not (eval_ids & train_ids)
    t2 = get_task("code", sandbox=HAVE_SEATBELT)
    assert {r["task_id"] for r in t2._eval} == eval_ids


def test_correct_solution_scores_one(task):
    ex = _example(["assert add(2, 3) == 5", "assert add(-1, 1) == 0"])
    r = task.reward(ex, _completion("def add(a, b):\n    return a + b"))
    assert r.total == 1.0
    assert r.parts["correct"] == 1.0


def test_wrong_solution_scores_zero(task):
    ex = _example(["assert add(2, 3) == 5"])
    r = task.reward(ex, _completion("def add(a, b):\n    return a - b"))
    assert r.total == 0.0


def test_no_code_scores_zero(task):
    r = task.reward(_example(["assert True"]), "I cannot solve this.")
    assert r.total == 0.0
    assert r.parts.get("nopatch") == 1.0


def test_writes_inside_cwd_still_allowed(task):
    # Legit scratch-file use inside the candidate's own temp dir must survive
    # the sandbox, or we'd be punishing valid programs.
    body = ("def roundtrip():\n"
            "    open('x.txt', 'w').write('hi')\n"
            "    return open('x.txt').read()")
    ex = _example(["assert roundtrip() == 'hi'"])
    assert task.reward(ex, _completion(body)).total == 1.0


@needs_seatbelt
def test_sandbox_blocks_writes_outside_tmpdir(task):
    with tempfile.TemporaryDirectory() as outside:
        target = Path(outside).resolve() / "escaped.txt"
        body = ("def f():\n"
                f"    open({str(target)!r}, 'w').write('escaped')\n"
                "    return True")
        r = task.reward(_example(["assert f()"]), _completion(body))
        assert r.total == 0.0
        assert not target.exists()


@needs_seatbelt
def test_sandbox_blocks_network(task):
    body = ("def f():\n"
            "    import socket\n"
            "    socket.create_connection(('127.0.0.1', 9), timeout=2)\n"
            "    return True")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 0.0


def test_infinite_loop_times_out(task, monkeypatch):
    monkeypatch.setattr("mlx_rl.tasks.code._TIMEOUT_S", 2)
    ex = _example(["assert f()"])
    r = task.reward(ex, _completion("def f():\n    while True:\n        pass"))
    assert r.total == 0.0


def test_sandbox_false_runs_plain(monkeypatch):
    t = get_task("code", sandbox=False)
    assert t._sandbox_exec is None
    ex = _example(["assert add(1, 1) == 2"])
    r = t.reward(ex, _completion("def add(a, b):\n    return a + b"))
    assert r.total == 1.0


def test_missing_sandbox_exec_raises(monkeypatch):
    monkeypatch.setattr("mlx_rl.tasks.code.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="sandbox-exec"):
        get_task("code", sandbox=True)


# --- adversarial sandbox probes ---------------------------------------------
# Each runs a hostile candidate through the real reward() path and asserts
# both the score and the absence of side effects.


@needs_seatbelt
def test_sandbox_env_is_scrubbed(task, monkeypatch):
    # A secret in the trainer's environment must not reach candidate code.
    monkeypatch.setenv("MLX_RL_FAKE_SECRET", "hunter2")
    body = ("def f():\n"
            "    import os\n"
            "    return 'MLX_RL_FAKE_SECRET' not in os.environ")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 1.0


@needs_seatbelt
def test_sandbox_home_and_tmpdir_point_into_sandbox(task):
    # HOME and TMPDIR are redirected, so expanduser()/tempfile land inside
    # the writable dir instead of touching the real home.
    body = ("def f():\n"
            "    import os, tempfile\n"
            "    cwd = os.getcwd()\n"
            "    ok_home = os.path.expanduser('~') == cwd\n"
            "    with tempfile.NamedTemporaryFile() as fh:\n"
            "        ok_tmp = fh.name.startswith(cwd)\n"
            "    return ok_home and ok_tmp")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 1.0


@needs_seatbelt
def test_sandbox_blocks_write_to_real_home(task):
    target = Path.home().resolve() / ".mlx_rl_sandbox_escape_test"
    body = ("def f():\n"
            "    try:\n"
            f"        open({str(target)!r}, 'w').write('escaped')\n"
            "        return False\n"
            "    except OSError:\n"
            "        return True")
    try:
        r = task.reward(_example(["assert f()"]), _completion(body))
        assert r.total == 1.0
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


@needs_seatbelt
def test_sandbox_blocks_symlink_escape(task):
    # Seatbelt checks the resolved path, so writing through a symlink that
    # points outside the temp dir must still be denied.
    home = Path.home().resolve()
    target = home / ".mlx_rl_symlink_escape_test"
    body = ("def f():\n"
            "    import os\n"
            f"    os.symlink({str(home)!r}, 'link')\n"
            "    try:\n"
            "        open('link/.mlx_rl_symlink_escape_test', 'w').write('x')\n"
            "        return False\n"
            "    except OSError:\n"
            "        return True")
    try:
        r = task.reward(_example(["assert f()"]), _completion(body))
        assert r.total == 1.0
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


@needs_seatbelt
def test_sandbox_blocks_listening_socket(task):
    body = ("def f():\n"
            "    import socket\n"
            "    s = socket.socket()\n"
            "    s.bind(('127.0.0.1', 0))\n"
            "    return True")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 0.0


@needs_seatbelt
def test_sandbox_inherited_by_grandchildren(task):
    # The profile survives fork+exec: a subprocess spawned by the candidate
    # is under the same sandbox and still can't write outside.
    with tempfile.TemporaryDirectory() as outside:
        target = Path(outside).resolve() / "grandchild.txt"
        inner = f"open({str(target)!r}, 'w').write('x')"
        body = ("def f():\n"
                "    import subprocess, sys\n"
                f"    r = subprocess.run([sys.executable, '-c', {inner!r}],\n"
                "                       capture_output=True)\n"
                "    return r.returncode != 0")
        r = task.reward(_example(["assert f()"]), _completion(body))
        assert r.total == 1.0
        assert not target.exists()


@needs_seatbelt
def test_sandbox_still_allows_reads(task):
    # Deliberate design: the profile is allow-default (reads permitted) —
    # only network and stray writes are denied. Guards against a profile
    # change that would break interpreter startup or stdlib imports.
    body = ("def f():\n"
            "    return len(open('/etc/hosts').read()) > 0")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 1.0


def test_rlimits_applied_in_child(task):
    from mlx_rl.tasks.code import _TIMEOUT_S
    body = ("def f():\n"
            "    import resource\n"
            f"    assert resource.getrlimit(resource.RLIMIT_CPU)[0] == {_TIMEOUT_S + 2}\n"
            "    assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 16 * 2**20\n"
            "    assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 256\n"
            "    assert resource.getrlimit(resource.RLIMIT_NPROC)[0] == 1024\n"
            "    return True")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 1.0


def test_fsize_limit_kills_oversized_write(task):
    # 20MB > the 16MB RLIMIT_FSIZE cap → SIGXFSZ kills the process mid-write.
    body = ("def f():\n"
            "    with open('big.bin', 'wb') as fh:\n"
            "        fh.write(b'\\0' * (20 * 2**20))\n"
            "    return True")
    assert task.reward(_example(["assert f()"]), _completion(body)).total == 0.0
