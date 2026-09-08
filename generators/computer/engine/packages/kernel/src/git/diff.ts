/**
 * Line diffing, unified-diff rendering and three-way merge.
 *
 * The matcher is a classic LCS over lines (with a shared prefix/suffix trim so
 * the quadratic table only ever sees the genuinely differing middle). Output is
 * byte-compatible with `git diff --no-color` for the cases the simulator
 * produces: identical hunk headers, three lines of context, `\ No newline at
 * end of file` markers.
 */

export interface DiffChange {
  /** Index into the base (a) sequence where the change starts. */
  baseStart: number;
  baseLength: number;
  /** Index into the other (b) sequence where the replacement starts. */
  otherStart: number;
  otherLength: number;
}

export function splitLines(content: string): string[] {
  if (content === '') return [];
  const lines = content.split('\n');
  if (lines.at(-1) === '') lines.pop();
  return lines;
}

export function endsWithNewline(content: string): boolean {
  return content === '' || content.endsWith('\n');
}

/** Longest common subsequence over lines, returned as aligned index pairs. */
function lcsPairs(a: readonly string[], b: readonly string[]): Array<[number, number]> {
  const n = a.length;
  const m = b.length;
  // Table is (n+1) x (m+1); files in the simulator are small enough for this.
  const table: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i]![j] = a[i] === b[j] ? table[i + 1]![j + 1]! + 1 : Math.max(table[i + 1]![j]!, table[i]![j + 1]!);
    }
  }
  const pairs: Array<[number, number]> = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { pairs.push([i, j]); i++; j++; }
    else if (table[i + 1]![j]! >= table[i]![j + 1]!) i++;
    else j++;
  }
  return pairs;
}

/** Changed regions turning `a` into `b`. */
export function diffChanges(a: readonly string[], b: readonly string[]): DiffChange[] {
  let prefix = 0;
  while (prefix < a.length && prefix < b.length && a[prefix] === b[prefix]) prefix++;
  let suffix = 0;
  while (
    suffix < a.length - prefix
    && suffix < b.length - prefix
    && a[a.length - 1 - suffix] === b[b.length - 1 - suffix]
  ) suffix++;

  const midA = a.slice(prefix, a.length - suffix);
  const midB = b.slice(prefix, b.length - suffix);
  const pairs = lcsPairs(midA, midB);

  const changes: DiffChange[] = [];
  let i = 0;
  let j = 0;
  const flush = (endA: number, endB: number) => {
    if (endA > i || endB > j) {
      changes.push({ baseStart: prefix + i, baseLength: endA - i, otherStart: prefix + j, otherLength: endB - j });
    }
  };
  for (const [pa, pb] of pairs) {
    flush(pa, pb);
    i = pa + 1;
    j = pb + 1;
  }
  flush(midA.length, midB.length);
  return changes;
}

export interface UnifiedDiffOptions {
  /** Path rendered after `a/`. */
  fromPath: string;
  /** Path rendered after `b/`. */
  toPath: string;
  fromLabel?: string;
  toLabel?: string;
  context?: number;
}

interface HunkLine { prefix: ' ' | '-' | '+'; text: string; }

/** Unified diff body (hunks only; callers prepend the `diff --git` header). */
export function unifiedDiff(before: string, after: string, options: UnifiedDiffOptions): string {
  const context = options.context ?? 3;
  const a = splitLines(before);
  const b = splitLines(after);
  const changes = diffChanges(a, b);
  if (!changes.length) return '';

  const noNewlineBefore = before !== '' && !endsWithNewline(before);
  const noNewlineAfter = after !== '' && !endsWithNewline(after);

  interface Hunk { aStart: number; aCount: number; bStart: number; bCount: number; lines: HunkLine[]; }
  const hunks: Hunk[] = [];
  let current: Hunk | undefined;
  let lastCopied = 0;

  for (const change of changes) {
    const contextStart = Math.max(0, change.baseStart - context);
    if (current && contextStart <= lastCopied) {
      for (let line = lastCopied; line < change.baseStart; line++) {
        current.lines.push({ prefix: ' ', text: a[line]! });
        current.aCount++;
        current.bCount++;
      }
    } else {
      current = {
        aStart: contextStart,
        aCount: change.baseStart - contextStart,
        bStart: change.otherStart - (change.baseStart - contextStart),
        bCount: change.baseStart - contextStart,
        lines: [],
      };
      for (let line = contextStart; line < change.baseStart; line++) current.lines.push({ prefix: ' ', text: a[line]! });
      hunks.push(current);
    }
    for (let line = change.baseStart; line < change.baseStart + change.baseLength; line++) {
      current.lines.push({ prefix: '-', text: a[line]! });
      current.aCount++;
    }
    for (let line = change.otherStart; line < change.otherStart + change.otherLength; line++) {
      current.lines.push({ prefix: '+', text: b[line]! });
      current.bCount++;
    }
    lastCopied = change.baseStart + change.baseLength;
    const trailing = Math.min(a.length, lastCopied + context);
    for (let line = lastCopied; line < trailing; line++) {
      current.lines.push({ prefix: ' ', text: a[line]! });
      current.aCount++;
      current.bCount++;
    }
    lastCopied = trailing;
  }

  const output: string[] = [];
  const fromLabel = options.fromLabel ?? `a/${options.fromPath}`;
  const toLabel = options.toLabel ?? `b/${options.toPath}`;
  output.push(`--- ${fromLabel}`);
  output.push(`+++ ${toLabel}`);
  for (const hunk of hunks) {
    const aRange = hunk.aCount === 1 ? `${hunk.aStart + 1}` : `${hunk.aCount === 0 ? hunk.aStart : hunk.aStart + 1},${hunk.aCount}`;
    const bRange = hunk.bCount === 1 ? `${hunk.bStart + 1}` : `${hunk.bCount === 0 ? hunk.bStart : hunk.bStart + 1},${hunk.bCount}`;
    output.push(`@@ -${aRange} +${bRange} @@`);
    for (const line of hunk.lines) {
      output.push(`${line.prefix}${line.text}`);
      const isLastOfA = line.prefix !== '+' && hunk.aStart + hunk.aCount >= a.length && line.text === a.at(-1);
      const isLastOfB = line.prefix !== '-' && hunk.bStart + hunk.bCount >= b.length && line.text === b.at(-1);
      if (line.prefix === '-' && noNewlineBefore && isLastOfA) output.push('\\ No newline at end of file');
      else if (line.prefix === '+' && noNewlineAfter && isLastOfB) output.push('\\ No newline at end of file');
    }
  }
  return `${output.join('\n')}\n`;
}

export type MergeRegion =
  | { ok: string[] }
  | { conflict: { ours: string[]; base: string[]; theirs: string[] } };

/**
 * diff3-style three-way merge. Regions changed on one side only are taken from
 * that side; regions changed on both sides are taken once when the two sides
 * agree, and reported as conflicts when they do not.
 */
export function merge3(ours: readonly string[], base: readonly string[], theirs: readonly string[]): MergeRegion[] {
  const ourChanges = diffChanges(base, ours);
  const theirChanges = diffChanges(base, theirs);
  const regions: MergeRegion[] = [];

  let baseCursor = 0;
  let ourCursor = 0;
  let theirCursor = 0;
  let oi = 0;
  let ti = 0;

  const pushOk = (lines: string[]) => {
    if (!lines.length) return;
    const last = regions.at(-1);
    if (last && 'ok' in last) last.ok.push(...lines);
    else regions.push({ ok: lines });
  };

  while (oi < ourChanges.length || ti < theirChanges.length) {
    const ourChange = ourChanges[oi];
    const theirChange = theirChanges[ti];
    const nextStart = Math.min(ourChange?.baseStart ?? Infinity, theirChange?.baseStart ?? Infinity);

    // Copy the stable base region preceding the next change.
    if (nextStart > baseCursor) {
      const stable = base.slice(baseCursor, nextStart);
      pushOk(stable);
      ourCursor += nextStart - baseCursor;
      theirCursor += nextStart - baseCursor;
      baseCursor = nextStart;
    }

    // Grow a combined window while the two sides' changed base ranges overlap.
    let windowEnd = nextStart;
    let takeOurs = false;
    let takeTheirs = false;
    let oj = oi;
    let tj = ti;
    let grew = true;
    while (grew) {
      grew = false;
      while (oj < ourChanges.length && ourChanges[oj]!.baseStart <= windowEnd) {
        const change = ourChanges[oj]!;
        windowEnd = Math.max(windowEnd, change.baseStart + change.baseLength);
        takeOurs = true;
        oj++;
        grew = true;
      }
      while (tj < theirChanges.length && theirChanges[tj]!.baseStart <= windowEnd) {
        const change = theirChanges[tj]!;
        windowEnd = Math.max(windowEnd, change.baseStart + change.baseLength);
        takeTheirs = true;
        tj++;
        grew = true;
      }
    }

    const lastOur = ourChanges[oj - 1];
    const lastTheir = theirChanges[tj - 1];
    const ourEnd = takeOurs && lastOur
      ? lastOur.otherStart + lastOur.otherLength + (windowEnd - (lastOur.baseStart + lastOur.baseLength))
      : ourCursor + (windowEnd - baseCursor);
    const theirEnd = takeTheirs && lastTheir
      ? lastTheir.otherStart + lastTheir.otherLength + (windowEnd - (lastTheir.baseStart + lastTheir.baseLength))
      : theirCursor + (windowEnd - baseCursor);

    const baseSlice = base.slice(baseCursor, windowEnd);
    const ourSlice = ours.slice(ourCursor, ourEnd);
    const theirSlice = theirs.slice(theirCursor, theirEnd);

    if (!takeTheirs) pushOk(ourSlice);
    else if (!takeOurs) pushOk(theirSlice);
    else if (sameLines(ourSlice, theirSlice)) pushOk(ourSlice);
    else regions.push({ conflict: { ours: ourSlice, base: baseSlice, theirs: theirSlice } });

    baseCursor = windowEnd;
    ourCursor = ourEnd;
    theirCursor = theirEnd;
    oi = oj;
    ti = tj;
  }

  pushOk(base.slice(baseCursor));
  return regions;
}

function sameLines(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((line, index) => line === b[index]);
}

export interface MergeTextResult {
  content: string;
  conflicted: boolean;
}

/** Render a three-way merge, using Git's default conflict marker style. */
export function mergeText(
  ours: string,
  base: string,
  theirs: string,
  labels: { ours: string; base?: string; theirs: string; diff3?: boolean },
): MergeTextResult {
  const regions = merge3(splitLines(ours), splitLines(base), splitLines(theirs));
  const lines: string[] = [];
  let conflicted = false;
  for (const region of regions) {
    if ('ok' in region) { lines.push(...region.ok); continue; }
    conflicted = true;
    lines.push(`<<<<<<< ${labels.ours}`);
    lines.push(...region.conflict.ours);
    if (labels.diff3) {
      lines.push(`||||||| ${labels.base ?? 'base'}`);
      lines.push(...region.conflict.base);
    }
    lines.push('=======');
    lines.push(...region.conflict.theirs);
    lines.push(`>>>>>>> ${labels.theirs}`);
  }
  const content = lines.length ? `${lines.join('\n')}\n` : '';
  return { content, conflicted };
}
