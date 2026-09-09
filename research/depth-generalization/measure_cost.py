"""Program text against depth: the interpreter is flat where an unroll is not."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tcn.generation import Host
from tcn.search import space_size
import interpreter as I

WIDTH, CAPACITY, HORIZON = 4, 8, 12
BASE = {'inputs': WIDTH, 'gate_capacity': CAPACITY, 'nondegenerate': True, 'min_relevant_inputs': 2}


def symbols(program):
    """Nodes plus constants plus state ports, with each module definition once."""
    return len(program.nodes) + len(program.constants) + len(program.state)


def main():
    rows = {}
    for depth in (1, 2, 3, 4, 6, 8):
        host = Host.create('logic', seed=0, index=0, split='test',
                           configuration=BASE | {'depth': depth, 'horizon': HORIZON})
        names, program, registry = I.interpreter_scaffold(host, WIDTH, CAPACITY, HORIZON, wire_choice=True)
        rows[str(depth)] = {
            'nodes': len(program.nodes), 'structural_symbols': symbols(program),
            'space_size': space_size(program),
            'execution_cost_per_tick': program.execution_cost(registry),
            'description_bits': program.description_bits(registry),
            'digest': program.digest,
            'gates_flat_width': host.view().observations['gates'].type.width,
            'program_flat_width': host.view().observations['program'].type.width,
            'ticks_to_settle': depth - 1,
        }
    rows['one_program_text'] = len({v['digest'] for v in rows.values() if isinstance(v, dict)}) == 1
    Path(__file__).parent.joinpath('out/cost.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
