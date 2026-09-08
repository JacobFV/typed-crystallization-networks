/**
 * `.gitignore` pattern matching: comments, negation, directory-only rules,
 * anchoring, `**` spans, character classes, and per-directory ignore files.
 * Ancestor exclusion wins over negation, exactly like Git — a file inside an
 * ignored directory cannot be re-included.
 */

export interface IgnoreRule {
  /** Directory the rule was declared in, relative to the repository root (`''` = root). */
  base: string;
  pattern: string;
  negated: boolean;
  dirOnly: boolean;
  regex: RegExp;
  source: string;
}

function escapeRegex(char: string): string {
  return /[.*+?^${}()|[\]\\]/.test(char) ? `\\${char}` : char;
}

function globToRegexBody(pattern: string): string {
  let body = '';
  for (let i = 0; i < pattern.length; i++) {
    const char = pattern[i]!;
    if (char === '\\' && i + 1 < pattern.length) { body += escapeRegex(pattern[++i]!); continue; }
    if (char === '*') {
      const doubled = pattern[i + 1] === '*';
      if (doubled) {
        i++;
        if (pattern[i + 1] === '/') { i++; body += '(?:[^/]+/)*'; }
        else body += '.*';
      } else body += '[^/]*';
      continue;
    }
    if (char === '?') { body += '[^/]'; continue; }
    if (char === '[') {
      const close = pattern.indexOf(']', i + 1);
      if (close === -1) { body += '\\['; continue; }
      let set = pattern.slice(i + 1, close);
      if (set.startsWith('!')) set = `^${set.slice(1)}`;
      body += `[${set}]`;
      i = close;
      continue;
    }
    body += escapeRegex(char);
  }
  return body;
}

export function compileIgnorePattern(rawPattern: string, base: string, source: string): IgnoreRule | undefined {
  let pattern = rawPattern;
  if (!pattern.trim()) return undefined;
  if (pattern.startsWith('#')) return undefined;
  // Trailing whitespace is insignificant unless escaped.
  pattern = pattern.replace(/(?<!\\)\s+$/, '');
  if (!pattern) return undefined;
  let negated = false;
  if (pattern.startsWith('!')) { negated = true; pattern = pattern.slice(1); }
  else if (pattern.startsWith('\\!') || pattern.startsWith('\\#')) pattern = pattern.slice(1);
  let dirOnly = false;
  if (pattern.endsWith('/')) { dirOnly = true; pattern = pattern.slice(0, -1); }
  if (!pattern) return undefined;
  const anchored = pattern.includes('/');
  if (pattern.startsWith('/')) pattern = pattern.slice(1);
  const body = globToRegexBody(pattern);
  const regex = new RegExp(`^${anchored ? '' : '(?:.*/)?'}${body}$`);
  return { base, pattern: rawPattern, negated, dirOnly, regex, source };
}

export function parseIgnoreFile(text: string, base: string, source: string): IgnoreRule[] {
  const rules: IgnoreRule[] = [];
  for (const line of text.split('\n')) {
    const rule = compileIgnorePattern(line.replace(/\r$/, ''), base, source);
    if (rule) rules.push(rule);
  }
  return rules;
}

export class IgnoreMatcher {
  constructor(private readonly rules: readonly IgnoreRule[] = []) {}

  get size(): number { return this.rules.length; }

  /** Decision for one path with no ancestor consideration. */
  private decide(relativePath: string, isDirectory: boolean): boolean | undefined {
    let decision: boolean | undefined;
    for (const rule of this.rules) {
      if (rule.base && !relativePath.startsWith(`${rule.base}/`)) continue;
      const scoped = rule.base ? relativePath.slice(rule.base.length + 1) : relativePath;
      if (rule.dirOnly && !isDirectory) continue;
      if (!rule.regex.test(scoped)) continue;
      decision = !rule.negated;
    }
    return decision;
  }

  /**
   * Whether Git would ignore `relativePath`. Directories are tested along the
   * way, because an ignored directory is never descended into.
   */
  ignores(relativePath: string, isDirectory = false): boolean {
    if (!this.rules.length) return false;
    const parts = relativePath.split('/').filter(Boolean);
    let decision = false;
    for (let i = 0; i < parts.length; i++) {
      const candidate = parts.slice(0, i + 1).join('/');
      const last = i === parts.length - 1;
      const verdict = this.decide(candidate, last ? isDirectory : true);
      if (verdict === true && !last) return true;
      if (verdict !== undefined) decision = verdict;
      else if (!last) decision = false;
    }
    return decision;
  }
}
