/**
 * Shell grammar: lexer and parser.
 *
 * The old shell split statements with regular expressions, so `echo "a; b"`
 * split inside the quotes and `echo "a > b"` created a file. Everything is
 * tokenized here *first* — quoting, escapes, expansions, heredocs — and only
 * then split on operators, which is the only way those bugs stay fixed.
 */

export type QuoteKind = 'none' | 'single' | 'double';

export interface LiteralPart { kind: 'literal'; text: string; quote: QuoteKind }
export interface VariablePart {
  kind: 'variable';
  name: string;
  /** `:-` `:=` `:+` `:?` `-` `+` from `${VAR:-default}` forms. */
  operator?: string;
  word?: string;
  /** `${#VAR}` */
  length?: boolean;
  quote: QuoteKind;
}
export interface CommandPart { kind: 'command'; source: string; quote: QuoteKind }
/** `$(( expression ))`. Lexed separately from `$( command )` so the two cannot be confused. */
export interface ArithmeticPart { kind: 'arithmetic'; source: string; quote: QuoteKind }
export type WordPart = LiteralPart | VariablePart | CommandPart | ArithmeticPart;

export interface Word { parts: WordPart[]; raw: string }

export type OperatorValue = '|' | '||' | '&&' | ';' | '&' | '\n' | '>' | '>>' | '<' | '<<' | '2>' | '2>>' | '&>' | '&>>' | '(' | ')';

export interface HeredocBody { text: string; expand: boolean }

export type Token =
  | { kind: 'word'; word: Word }
  | { kind: 'operator'; value: OperatorValue; heredoc?: HeredocBody };

export interface Redirection {
  fd: 1 | 2 | 'both';
  operator: '>' | '>>' | '<' | '<<';
  target?: Word;
  heredoc?: HeredocBody;
}

export interface SimpleCommand { kind: 'simple'; assignments: Word[]; words: Word[]; redirections: Redirection[] }
export interface SubshellCommand { kind: 'subshell'; statements: Statement[]; redirections: Redirection[] }
export type CommandNode = SimpleCommand | SubshellCommand;
export interface Pipeline { commands: CommandNode[] }
export interface AndOrEntry { operator: 'first' | '&&' | '||'; pipeline: Pipeline }
export interface Statement { entries: AndOrEntry[]; background: boolean }

export class ShellSyntaxError extends Error {}

const OPERATOR_CHARACTERS = new Set(['|', '&', ';', '<', '>', '(', ')', '\n']);

/** Read a `${...}` body into a variable part. */
function parseBracedVariable(body: string, quote: QuoteKind): VariablePart {
  if (body.startsWith('#')) return { kind: 'variable', name: body.slice(1), length: true, quote };
  const match = body.match(/^([A-Za-z_][A-Za-z0-9_]*|[?$!#@*0-9]|env:[A-Za-z_][A-Za-z0-9_]*)(:?[-=+?])?([\s\S]*)$/);
  if (!match) throw new ShellSyntaxError(`bad substitution: \${${body}}`);
  const [, name, operator, word] = match;
  return { kind: 'variable', name: name!, ...(operator ? { operator, word: word ?? '' } : {}), quote };
}

/** Scan a balanced `(...)` or `` `...` `` body starting after the opener. */
function readBalanced(source: string, start: number, open: string, close: string): { body: string; end: number } {
  let depth = 1;
  let index = start;
  let body = '';
  let quote = '';
  while (index < source.length) {
    const char = source[index]!;
    if (quote) {
      body += char;
      if (char === '\\' && index + 1 < source.length) { body += source[++index]; }
      else if (char === quote) quote = '';
      index += 1;
      continue;
    }
    if (char === '"' || char === "'") { quote = char; body += char; index += 1; continue; }
    if (open !== close && char === open) depth += 1;
    if (char === close) { depth -= 1; if (depth === 0) return { body, end: index + 1 }; }
    body += char;
    index += 1;
  }
  throw new ShellSyntaxError(`unterminated ${open === '`' ? 'backquote' : `'${open}'`} substitution`);
}

export interface LexOptions {
  /**
   * POSIX escapes with `\\` and substitutes with backticks; PowerShell escapes
   * with a backtick and treats `\\` as an ordinary path separator, which is why
   * `C:\\Users\\agent` survives only when the dialect is honoured here.
   */
  dialect?: 'posix' | 'powershell';
}

/**
 * Tokenize a command line. Heredoc bodies are pulled from the lines that
 * follow the operator, exactly as an interactive shell does.
 */
export function lex(source: string, options: LexOptions = {}): Token[] {
  const powershell = options.dialect === 'powershell';
  const escapeCharacter = powershell ? '`' : '\\';
  const tokens: Token[] = [];
  let parts: WordPart[] = [];
  let raw = '';
  let buffer = '';
  let bufferQuote: QuoteKind = 'none';
  let bufferPending = false;
  let index = 0;
  const pendingHeredocs: Array<{ token: Extract<Token, { kind: 'operator' }>; delimiter: string; expand: boolean; strip: boolean }> = [];

  const flushLiteral = (): void => {
    if (buffer !== '' || bufferPending) parts.push({ kind: 'literal', text: buffer, quote: bufferQuote });
    buffer = '';
    bufferPending = false;
    bufferQuote = 'none';
  };
  const flushWord = (): void => {
    flushLiteral();
    if (parts.length) tokens.push({ kind: 'word', word: { parts, raw } });
    parts = [];
    raw = '';
  };
  const append = (text: string, quote: QuoteKind): void => {
    if (buffer !== '' && bufferQuote !== quote) flushLiteral();
    bufferQuote = quote;
    buffer += text;
  };
  /** True when the pending word is exactly the unquoted digit `value`. */
  const wordIsDigit = (value: string): boolean => parts.length === 0 && buffer === value && bufferQuote === 'none';

  const readExpansion = (quote: QuoteKind): boolean => {
    const next = source[index + 1];
    if (next === '(' && source[index + 2] === '(') {
      // `$((…))` must be recognised before `$(…)`, otherwise arithmetic is lexed
      // as a subshell running the expression as a command.
      const { body, end } = readBalanced(source, index + 3, '(', ')');
      if (source[end] === ')') {
        flushLiteral();
        parts.push({ kind: 'arithmetic', source: body, quote });
        index = end + 1;
        return true;
      }
    }
    if (next === '(') {
      const { body, end } = readBalanced(source, index + 2, '(', ')');
      flushLiteral();
      parts.push({ kind: 'command', source: body, quote });
      index = end;
      return true;
    }
    if (next === '{') {
      const { body, end } = readBalanced(source, index + 2, '{', '}');
      flushLiteral();
      parts.push(parseBracedVariable(body, quote));
      index = end;
      return true;
    }
    const rest = source.slice(index + 1);
    const named = rest.match(/^(env:[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*|[?$!#0-9])/);
    if (!named) return false;
    flushLiteral();
    parts.push({ kind: 'variable', name: named[1]!, quote });
    index += 1 + named[1]!.length;
    return true;
  };

  const consumeHeredocBodies = (): void => {
    if (!pendingHeredocs.length) return;
    const lines: string[] = [];
    while (pendingHeredocs.length) {
      const pending = pendingHeredocs.shift()!;
      lines.length = 0;
      for (;;) {
        if (index >= source.length) break;
        const lineEnd = source.indexOf('\n', index);
        const line = lineEnd === -1 ? source.slice(index) : source.slice(index, lineEnd);
        index = lineEnd === -1 ? source.length : lineEnd + 1;
        if ((pending.strip ? line.trimStart() : line) === pending.delimiter) break;
        lines.push(pending.strip ? line.replace(/^\t+/, '') : line);
        if (lineEnd === -1) break;
      }
      pending.token.heredoc = { text: lines.length ? `${lines.join('\n')}\n` : '', expand: pending.expand };
    }
  };

  while (index < source.length) {
    const char = source[index]!;
    if (char === escapeCharacter) {
      const next = source[index + 1];
      if (next === undefined) { append(escapeCharacter, 'single'); index += 1; continue; }
      if (next === '\n') { index += 2; continue; } // line continuation
      raw += char + next;
      append(next, 'single');
      index += 2;
      continue;
    }
    if (char === "'") {
      const end = source.indexOf("'", index + 1);
      if (end === -1) throw new ShellSyntaxError('unterminated single quote');
      if (buffer !== '' && bufferQuote !== 'single') flushLiteral();
      bufferQuote = 'single';
      bufferPending = true;
      buffer += source.slice(index + 1, end);
      raw += source.slice(index, end + 1);
      index = end + 1;
      continue;
    }
    if (char === '"') {
      if (buffer !== '' && bufferQuote !== 'double') flushLiteral();
      bufferQuote = 'double';
      bufferPending = true;
      index += 1;
      let closed = false;
      while (index < source.length) {
        const inner = source[index]!;
        if (inner === '"') { closed = true; index += 1; break; }
        if (inner === escapeCharacter) {
          const next = source[index + 1];
          if (next !== undefined && ['"', escapeCharacter, '$', '`', '\n'].includes(next)) {
            if (next !== '\n') append(next, 'double');
            index += 2;
            continue;
          }
          append(escapeCharacter, 'double');
          index += 1;
          continue;
        }
        if (inner === '$' && readExpansion('double')) continue;
        if (!powershell && inner === '`') {
          const { body, end } = readBalanced(source, index + 1, '`', '`');
          flushLiteral();
          parts.push({ kind: 'command', source: body, quote: 'double' });
          index = end;
          continue;
        }
        append(inner, 'double');
        index += 1;
      }
      if (!closed) throw new ShellSyntaxError('unterminated double quote');
      continue;
    }
    if (char === '$' && readExpansion(bufferQuote === 'none' ? 'none' : bufferQuote)) continue;
    if (!powershell && char === '`') {
      const { body, end } = readBalanced(source, index + 1, '`', '`');
      flushLiteral();
      parts.push({ kind: 'command', source: body, quote: 'none' });
      index = end;
      continue;
    }
    if (char === '#' && buffer === '' && parts.length === 0) {
      const lineEnd = source.indexOf('\n', index);
      index = lineEnd === -1 ? source.length : lineEnd;
      continue;
    }
    if (/[ \t]/.test(char)) { flushWord(); index += 1; continue; }
    if (OPERATOR_CHARACTERS.has(char)) {
      if (char === '(' || char === ')') {
        if (char === '(' && (buffer !== '' || parts.length)) { append(char, 'none'); index += 1; continue; }
        flushWord();
        tokens.push({ kind: 'operator', value: char });
        index += 1;
        continue;
      }
      if (char === '>' || char === '<') {
        const fd = wordIsDigit('2') ? 2 : wordIsDigit('1') ? 1 : undefined;
        if (fd !== undefined) { buffer = ''; bufferQuote = 'none'; bufferPending = false; }
        flushWord();
        const doubled = source[index + 1] === char;
        if (char === '>') {
          tokens.push({ kind: 'operator', value: fd === 2 ? (doubled ? '2>>' : '2>') : doubled ? '>>' : '>' });
          index += doubled ? 2 : 1;
          continue;
        }
        if (doubled) {
          index += 2;
          const strip = source[index] === '-';
          if (strip) index += 1;
          while (/[ \t]/.test(source[index] ?? '')) index += 1;
          const delimiterMatch = source.slice(index).match(/^(?:'([^']*)'|"([^"]*)"|([^\s;&|<>]+))/);
          if (!delimiterMatch) throw new ShellSyntaxError('expected a here-document delimiter after <<');
          const quotedDelimiter = delimiterMatch[1] ?? delimiterMatch[2];
          const delimiter = quotedDelimiter ?? delimiterMatch[3]!;
          index += delimiterMatch[0]!.length;
          const token: Extract<Token, { kind: 'operator' }> = { kind: 'operator', value: '<<' };
          tokens.push(token);
          pendingHeredocs.push({ token, delimiter, expand: quotedDelimiter === undefined, strip });
          continue;
        }
        tokens.push({ kind: 'operator', value: '<' });
        index += 1;
        continue;
      }
      if (char === '&') {
        flushWord();
        if (source[index + 1] === '&') { tokens.push({ kind: 'operator', value: '&&' }); index += 2; continue; }
        if (source[index + 1] === '>') {
          const doubled = source[index + 2] === '>';
          tokens.push({ kind: 'operator', value: doubled ? '&>>' : '&>' });
          index += doubled ? 3 : 2;
          continue;
        }
        tokens.push({ kind: 'operator', value: '&' });
        index += 1;
        continue;
      }
      if (char === '|') {
        flushWord();
        const doubled = source[index + 1] === '|';
        tokens.push({ kind: 'operator', value: doubled ? '||' : '|' });
        index += doubled ? 2 : 1;
        continue;
      }
      if (char === ';') { flushWord(); tokens.push({ kind: 'operator', value: ';' }); index += 1; continue; }
      // newline
      flushWord();
      tokens.push({ kind: 'operator', value: '\n' });
      index += 1;
      consumeHeredocBodies();
      continue;
    }
    raw += char;
    append(char, 'none');
    index += 1;
  }
  flushWord();
  consumeHeredocBodies();
  return tokens;
}

/* -------------------------------------------------------------------------- *
 * Parser
 * -------------------------------------------------------------------------- */

export function parse(tokens: Token[]): Statement[] {
  let position = 0;

  const peek = (): Token | undefined => tokens[position];
  const isOperator = (...values: OperatorValue[]): boolean => {
    const token = peek();
    return token?.kind === 'operator' && values.includes(token.value);
  };

  const parseRedirection = (redirections: Redirection[]): boolean => {
    const token = peek();
    if (token?.kind !== 'operator') return false;
    const map: Partial<Record<OperatorValue, Redirection>> = {
      '>': { fd: 1, operator: '>' }, '>>': { fd: 1, operator: '>>' },
      '2>': { fd: 2, operator: '>' }, '2>>': { fd: 2, operator: '>>' },
      '&>': { fd: 'both', operator: '>' }, '&>>': { fd: 'both', operator: '>>' },
      '<': { fd: 1, operator: '<' },
    };
    if (token.value === '<<') {
      position += 1;
      redirections.push({ fd: 1, operator: '<<', heredoc: token.heredoc ?? { text: '', expand: true } });
      return true;
    }
    const base = map[token.value];
    if (!base) return false;
    position += 1;
    const target = peek();
    if (target?.kind !== 'word') throw new ShellSyntaxError(`syntax error near unexpected token \`${token.value}'`);
    position += 1;
    redirections.push({ ...base, target: target.word });
    return true;
  };

  const parseCommand = (): CommandNode => {
    const redirections: Redirection[] = [];
    if (isOperator('(')) {
      position += 1;
      const statements = parseList([')']);
      if (!isOperator(')')) throw new ShellSyntaxError("syntax error: expected `)'");
      position += 1;
      while (parseRedirection(redirections)) { /* trailing redirections */ }
      return { kind: 'subshell', statements, redirections };
    }
    const assignments: Word[] = [];
    const words: Word[] = [];
    for (;;) {
      if (parseRedirection(redirections)) continue;
      const token = peek();
      if (!token || token.kind === 'operator') break;
      position += 1;
      const first = token.word.parts[0];
      const assignment = words.length === 0 && first?.kind === 'literal' && first.quote === 'none' && /^[A-Za-z_][A-Za-z0-9_]*=/.test(first.text);
      if (assignment) assignments.push(token.word); else words.push(token.word);
    }
    if (!words.length && !assignments.length && !redirections.length) throw new ShellSyntaxError('syntax error: empty command');
    return { kind: 'simple', assignments, words, redirections };
  };

  const parsePipeline = (): Pipeline => {
    const commands = [parseCommand()];
    while (isOperator('|')) {
      position += 1;
      while (isOperator('\n')) position += 1;
      commands.push(parseCommand());
    }
    return { commands };
  };

  function parseList(stop: OperatorValue[] = []): Statement[] {
    const statements: Statement[] = [];
    for (;;) {
      while (isOperator('\n', ';')) position += 1;
      const token = peek();
      if (!token) break;
      if (token.kind === 'operator' && stop.includes(token.value)) break;
      const entries: AndOrEntry[] = [{ operator: 'first', pipeline: parsePipeline() }];
      while (isOperator('&&', '||')) {
        const operator = (peek() as Extract<Token, { kind: 'operator' }>).value as '&&' | '||';
        position += 1;
        while (isOperator('\n')) position += 1;
        entries.push({ operator, pipeline: parsePipeline() });
      }
      let background = false;
      if (isOperator('&')) { background = true; position += 1; }
      else if (isOperator(';')) position += 1;
      statements.push({ entries, background });
      const next = peek();
      if (next?.kind === 'operator' && stop.includes(next.value)) break;
      if (next && next.kind === 'operator' && !['\n', ';', '&'].includes(next.value)) {
        throw new ShellSyntaxError(`syntax error near unexpected token \`${next.value}'`);
      }
    }
    return statements;
  }

  const statements = parseList();
  if (position < tokens.length) {
    const token = tokens[position]!;
    throw new ShellSyntaxError(`syntax error near unexpected token \`${token.kind === 'operator' ? token.value : token.word.raw}'`);
  }
  return statements;
}
