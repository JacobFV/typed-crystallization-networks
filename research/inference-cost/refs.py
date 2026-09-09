"""Plain-Python references for the two artifacts whose reference is not already
measured in `visual.py` / `inproc.py`, plus the matched-NN figures track 6 recorded.

The computer reference is the whole decision the frozen agent makes: read the
digit at the computed address, emit its successor, and choose the verb from
whether the terminal is showing file content or a command result.
"""
import sys, json, statistics
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/research/inference-cost')
sys.path.insert(0, ROOT + '/research/computer-capability')
from harness import timed, dump


def python_agent(terminal, wrote):
    """`terminal` is the raw text. Same rule as the frozen 23-node program:
    pos = length - 1 (the searched `sub(length, k0)`), shift = value + 1,
    verb from byte 0 against '{' (`eq(pbyte, b0)` at ppos = q0 = 0)."""
    if wrote:
        return 'wait', None
    if terminal[:1] == '{':
        return 'read', None
    return 'write', chr(ord(terminal[len(terminal) - 1]) + 1)


if __name__ == '__main__':
    import task as T
    from tcn.generation import read_text
    host = T.host('k', 7, 3, objective_path=T.TASK_PATH, seed=0, index=5000, split='test')
    host.step((T.read_action(),))
    terminal = read_text(host.view('agent_0').observations['terminal'])
    ms, mn, out = timed(lambda: python_agent(terminal, False), repeats=1001)
    row = {'terminal_repr': terminal[:60], 'decision': out,
           'plain_python_ms': ms, 'plain_python_min_ms': mn}
    print(json.dumps(row, indent=1))
    dump('refs', row)
