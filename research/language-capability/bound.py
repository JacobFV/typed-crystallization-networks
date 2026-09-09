"""Step 1: bound what is recoverable before searching.

For every implemented lesson, over N seeds, measure:
  - the answer set, its majority-class baseline, and the random-choice baseline
  - whether `answer` is determined by the observation `text` (prompt)
  - whether `answer` is determined by the latent `construction` (metadata.hidden)
  - prompt size in bytes, against the generator's declared text capacity

Determinism is measured as: group episodes by the exact key; a key seen with two
different answers is a violation. `best_given_key` is the accuracy of the optimal
deterministic predictor from that key -- an upper bound on any program reading it.
"""
from __future__ import annotations
import json, sys, collections
sys.path.insert(0, '.')
from generators.language.engine.registry import lesson_ids, get

def summarize(rows):
    ans = collections.Counter(a for _, _, a in rows)
    n = len(rows)
    def bound(keyidx):
        groups = collections.defaultdict(collections.Counter)
        for r in rows: groups[r[keyidx]][r[2]] += 1
        best = sum(c.most_common(1)[0][1] for c in groups.values())
        conflicts = sum(1 for c in groups.values() if len(c) > 1)
        return {'best_given_key': best / n, 'distinct_keys': len(groups),
                'keys_with_conflicting_answers': conflicts,
                'episodes_in_conflicted_keys': sum(sum(c.values()) for c in groups.values() if len(c) > 1)}
    return {'n': n, 'distinct_answers': len(ans),
            'majority_answer': ans.most_common(1)[0][0], 'majority_baseline': ans.most_common(1)[0][1] / n,
            'answer_counts': dict(ans.most_common(8)),
            'from_text': bound(0), 'from_construction': bound(1)}

def main(n=300):
    out = {}
    for lid in lesson_ids(implemented_only=True):
        lesson = get(lid)
        rows = []; choices = collections.Counter(); lens = []
        try:
            for s in range(n):
                ex = lesson.example(seed=s)
                hidden = json.dumps(ex.metadata.get('hidden', {}), sort_keys=True, default=str)
                rows.append((ex.prompt, hidden, ex.answer))
                choices[len(ex.choices)] += 1
                lens.append(len(ex.prompt.encode('utf8')))
        except Exception as e:
            out[lid] = {'error': f'{type(e).__name__}: {e}'}; continue
        d = summarize(rows)
        d['choice_sizes'] = dict(choices)
        d['random_choice_baseline'] = sum(k and 1.0/k for k in choices.elements()) / max(1, len(lens))
        d['prompt_bytes'] = {'min': min(lens), 'mean': sum(lens)/len(lens), 'max': max(lens)}
        out[lid] = d
    return out

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    res = main(n)
    json.dump(res, open('research/language-capability/bounds.json', 'w'), indent=1, sort_keys=True)
    print(f'{len(res)} lessons, n={n}')
