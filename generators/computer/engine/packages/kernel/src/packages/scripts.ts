import type { PackageManagerKind } from '@tcn-computer/protocol';
import type { VirtualFileSystem } from '../vfs.js';
import { parseDescriptor } from './executables.js';
import { binDirectory, executableFileName } from './layout.js';
import type { ManagerContext, SeedExecutableDescriptor } from './types.js';

/**
 * Script running.
 *
 * `npm run build` used to print a package list and exit 0, which meant nothing
 * in the simulation could be built or tested. A script now resolves its programs
 * through the same executable descriptors `install` writes, runs `pre`/`post`
 * hooks, propagates a non-zero exit, and fails honestly with
 * `command not found` when the tool was never installed.
 */

export interface ProgramResult {
  stdout: string;
  exitCode: number;
}

/** Injected by the shell when its program registry is available. */
export type ProgramRunner = (input: {
  argv: string[];
  cwd: string;
  descriptor: SeedExecutableDescriptor;
  path: string;
}) => Promise<ProgramResult>;

export interface ScriptEnvironment {
  vfs: VirtualFileSystem;
  context: ManagerContext;
  /** Where the shell's PATH lookup would search, most specific first. */
  searchPath: string[];
  runner?: ProgramRunner;
}

export interface ResolvedProgram {
  path: string;
  descriptor: SeedExecutableDescriptor;
}

export class ScriptError extends Error {
  constructor(message: string, readonly exitCode: number) { super(message); this.name = 'ScriptError'; }
}

/** Directories a project-scoped script searches, in PATH order. */
export function scriptSearchPath(context: ManagerContext, manager: PackageManagerKind): string[] {
  const project = binDirectory({ ...context, manager }, 'project');
  const user = binDirectory({ ...context, manager }, 'user');
  const extras = context.os === 'windows'
    ? ['/C/ProgramData/chocolatey/bin', `${context.home}/scoop/shims`, `${context.home}/AppData/Local/Microsoft/WinGet/Links`, `${context.home}/.cargo/bin`, `${context.home}/go/bin`, `${context.home}/.dotnet/tools`]
    : ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/snap/bin', `${context.home}/.local/bin`, `${context.home}/.cargo/bin`, `${context.home}/go/bin`, `${context.home}/.dotnet/tools`];
  return [...new Set([project, user, ...extras])];
}

export async function resolveProgram(environment: ScriptEnvironment, name: string): Promise<ResolvedProgram | undefined> {
  const candidates = name.includes('/')
    ? [name.startsWith('/') ? name : `${environment.context.cwd}/${name}`]
    : environment.searchPath.map((directory) => `${directory}/${executableFileName(environment.context, name)}`);
  for (const candidate of candidates) {
    if (!environment.vfs.exists(candidate)) continue;
    const descriptor = parseDescriptor(await environment.vfs.readFile(candidate));
    if (descriptor) return { path: candidate, descriptor };
  }
  return undefined;
}

/* --------------------------------------------------------------- behaviors */

const version = (name: string, descriptor: SeedExecutableDescriptor): string => `${name} ${descriptor.version}`;

/**
 * Built-in behaviors for the programs a build script actually calls. Anything
 * outside this table is reported as unmodelled rather than faked as success
 * output, and the shell can replace the whole table via {@link ProgramRunner}.
 */
async function builtinBehavior(
  environment: ScriptEnvironment,
  program: ResolvedProgram,
  argv: string[],
): Promise<ProgramResult> {
  const { descriptor } = program;
  const { vfs, context } = environment;
  const cwd = context.cwd;
  if (argv.includes('--version') || argv.includes('-V') || argv.includes('-v')) {
    return { stdout: version(descriptor.behavior, descriptor), exitCode: 0 };
  }
  switch (descriptor.behavior) {
    case 'tsc': {
      if (argv.includes('--noEmit')) return { stdout: '', exitCode: 0 };
      await vfs.writeFile(`${cwd}/dist/.tsbuildinfo`, JSON.stringify({ version: descriptor.version, program: { fileNames: [] } }));
      return { stdout: '', exitCode: 0 };
    }
    case 'vite': {
      if (argv[0] !== 'build') return { stdout: `  VITE v${descriptor.version}  ready in 214 ms\n\n  ➜  Local:   http://localhost:5173/`, exitCode: 0 };
      await vfs.writeFile(`${cwd}/dist/index.html`, '<!doctype html>\n<html><head><script type="module" src="/assets/index.js"></script></head><body><div id="root"></div></body></html>\n');
      await vfs.writeFile(`${cwd}/dist/assets/index.js`, '// built by vite\n');
      return {
        stdout: [`vite v${descriptor.version} building for production...`, '✓ 34 modules transformed.',
          'dist/index.html                   0.46 kB │ gzip:  0.30 kB',
          'dist/assets/index.js            143.21 kB │ gzip: 46.12 kB',
          '✓ built in 1.21s'].join('\n'),
        exitCode: 0,
      };
    }
    case 'esbuild': case 'rollup': case 'tsup': case 'webpack': case 'webpack-cli': {
      await vfs.writeFile(`${cwd}/dist/bundle.js`, `// bundled by ${descriptor.behavior}\n`);
      return { stdout: `${descriptor.behavior}: dist/bundle.js  142.0kb\n⚡ Done in 187ms`, exitCode: 0 };
    }
    case 'next': {
      if (argv[0] !== 'build') return { stdout: `▲ Next.js ${descriptor.version}\n- Local: http://localhost:3000`, exitCode: 0 };
      await vfs.writeFile(`${cwd}/.next/BUILD_ID`, 'seed-build\n');
      return { stdout: [`▲ Next.js ${descriptor.version}`, '', '   Creating an optimized production build ...', ' ✓ Compiled successfully', ' ✓ Generating static pages (4/4)'].join('\n'), exitCode: 0 };
    }
    case 'vitest': {
      return { stdout: [` RUN  v${descriptor.version} ${cwd}`, '', ' ✓ test/example.test.ts (2 tests) 4ms', '',
        ' Test Files  1 passed (1)', '      Tests  2 passed (2)', '   Duration  312ms'].join('\n'), exitCode: 0 };
    }
    case 'jest': {
      return { stdout: ['PASS test/example.test.js', '', 'Test Suites: 1 passed, 1 total', 'Tests:       2 passed, 2 total',
        'Snapshots:   0 total', 'Time:        0.412 s'].join('\n'), exitCode: 0 };
    }
    case 'pytest': case 'py.test': {
      return { stdout: ['============================= test session starts ==============================',
        `platform linux -- Python 3.13.7, pytest-${descriptor.version}`, `rootdir: ${cwd}`, 'collected 2 items', '',
        'tests/test_example.py ..                                                 [100%]', '',
        '============================== 2 passed in 0.04s =============================='].join('\n'), exitCode: 0 };
    }
    case 'eslint': return { stdout: '', exitCode: 0 };
    case 'prettier': return { stdout: argv.includes('--check') ? 'Checking formatting...\nAll matched files use Prettier code style!' : '', exitCode: 0 };
    case 'biome': return { stdout: 'Checked 34 files in 12ms. No fixes applied.', exitCode: 0 };
    case 'ruff': return { stdout: argv[0] === 'format' ? '12 files left unchanged' : 'All checks passed!', exitCode: 0 };
    case 'black': return { stdout: 'All done! ✨ 🍰 ✨\n12 files left unchanged.', exitCode: 0 };
    case 'mypy': return { stdout: 'Success: no issues found in 12 source files', exitCode: 0 };
    case 'tailwindcss': {
      await vfs.writeFile(`${cwd}/dist/output.css`, '/* generated by tailwindcss */\n');
      return { stdout: `\nRebuilding...\n\nDone in 142ms.`, exitCode: 0 };
    }
    case 'rimraf': {
      for (const target of argv.filter((argument) => !argument.startsWith('-'))) {
        await vfs.remove(target.startsWith('/') ? target : `${cwd}/${target}`, { recursive: true });
      }
      return { stdout: '', exitCode: 0 };
    }
    case 'turbo': return { stdout: `• Packages in scope: seed-project\n• Running ${argv[0] ?? 'build'}\n\n Tasks:    1 successful, 1 total\n  Time:    412ms`, exitCode: 0 };
    case 'nx': return { stdout: `> nx run seed-project:${argv[0] ?? 'build'}\n\n Successfully ran target ${argv[0] ?? 'build'} for project seed-project`, exitCode: 0 };
    case 'tsx': case 'ts-node': case 'node': {
      const script = argv.find((argument) => !argument.startsWith('-'));
      if (script && !vfs.exists(script.startsWith('/') ? script : `${cwd}/${script}`)) {
        return { stdout: `${descriptor.behavior}: cannot find module '${script}'`, exitCode: 1 };
      }
      return { stdout: '', exitCode: 0 };
    }
    case 'echo': return { stdout: argv.join(' '), exitCode: 0 };
    default:
      // Honest: the program is installed, but this kernel models no output for it.
      return { stdout: `${descriptor.name}: installed from ${descriptor.manager}, no simulated behavior for this program`, exitCode: 0 };
  }
}

/** Splits a script body on `&&`, honoring quotes. */
function splitChain(body: string): string[] {
  const parts: string[] = [];
  let current = '';
  let quote = '';
  for (let index = 0; index < body.length; index += 1) {
    const character = body[index]!;
    if (quote) {
      current += character;
      if (character === quote) quote = '';
      continue;
    }
    if (character === '"' || character === "'") { quote = character; current += character; continue; }
    if (character === '&' && body[index + 1] === '&') { parts.push(current); current = ''; index += 1; continue; }
    current += character;
  }
  parts.push(current);
  return parts.map((part) => part.trim()).filter(Boolean);
}

function tokenize(command: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let quote = '';
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index]!;
    if (quote) {
      if (character === quote) quote = ''; else current += character;
      continue;
    }
    if (character === '"' || character === "'") { quote = character; continue; }
    if (/\s/.test(character)) { if (current) { tokens.push(current); current = ''; } continue; }
    current += character;
  }
  if (current) tokens.push(current);
  return tokens;
}

export interface CommandOutcome { stdout: string; exitCode: number; failedCommand?: string }

/** Runs one `&&`-joined script body. Leading `VAR=value` assignments are consumed. */
export async function runCommandChain(environment: ScriptEnvironment, body: string): Promise<CommandOutcome> {
  const chunks: string[] = [];
  for (const command of splitChain(body)) {
    let tokens = tokenize(command);
    while (tokens.length && /^[A-Za-z_][A-Za-z0-9_]*=/.test(tokens[0]!)) tokens = tokens.slice(1);
    if (!tokens.length) continue;
    const [name, ...argv] = tokens as [string, ...string[]];
    const program = await resolveProgram(environment, name);
    if (!program) {
      chunks.push(`sh: 1: ${name}: not found`);
      return { stdout: chunks.join('\n'), exitCode: 127, failedCommand: command };
    }
    const result = environment.runner
      ? await environment.runner({ argv, cwd: environment.context.cwd, descriptor: program.descriptor, path: program.path })
      : await builtinBehavior(environment, program, argv);
    if (result.stdout) chunks.push(result.stdout);
    if (result.exitCode !== 0) return { stdout: chunks.join('\n'), exitCode: result.exitCode, failedCommand: command };
  }
  return { stdout: chunks.join('\n'), exitCode: 0 };
}

export interface ScriptManifest {
  name: string;
  version: string;
  scripts: Record<string, string>;
}

export async function readScriptManifest(vfs: VirtualFileSystem, cwd: string): Promise<ScriptManifest | undefined> {
  try {
    const manifest = JSON.parse(await vfs.readFile(`${cwd}/package.json`)) as Partial<ScriptManifest>;
    return {
      name: typeof manifest.name === 'string' ? manifest.name : 'seed-project',
      version: typeof manifest.version === 'string' ? manifest.version : '1.0.0',
      scripts: (manifest.scripts && typeof manifest.scripts === 'object' ? manifest.scripts : {}) as Record<string, string>,
    };
  } catch { return undefined; }
}

const MAX_DEPTH = 8;

/**
 * Runs `<manager> run <script>` including its `pre`/`post` hooks. Nested
 * `npm run other` inside a script recurses through the same path, bounded so a
 * self-referential script cannot spin.
 */
export async function runScript(
  environment: ScriptEnvironment,
  manager: PackageManagerKind,
  scriptName: string,
  extraArgs: readonly string[],
  depth = 0,
): Promise<string> {
  const manifest = await readScriptManifest(environment.vfs, environment.context.cwd);
  if (!manifest) {
    throw new ScriptError(`${manager} error code ENOENT\n${manager} error Could not read package.json in ${environment.context.cwd}`, 254);
  }
  if (depth > MAX_DEPTH) throw new ScriptError(`${manager} error Maximum script recursion depth exceeded at "${scriptName}"`, 1);
  const body = manifest.scripts[scriptName];
  if (!body) {
    // `npm test`/`npm start` have implicit defaults; everything else is missing.
    if (scriptName === 'test') throw new ScriptError(`${manager} error Missing script: "test"\n${manager} error\n${manager} error To see a list of scripts, run:\n${manager} error   ${manager} run`, 1);
    throw new ScriptError([
      `${manager} error Missing script: "${scriptName}"`,
      `${manager} error`,
      `${manager} error Did you mean one of these?`,
      ...Object.keys(manifest.scripts).slice(0, 5).map((name) => `${manager} error     ${manager} run ${name}`),
      `${manager} error`,
      `${manager} error To see a list of scripts, run:`,
      `${manager} error   ${manager} run`,
    ].join('\n'), 1);
  }
  const output: string[] = [];
  for (const stage of [`pre${scriptName}`, scriptName, `post${scriptName}`]) {
    const stageBody = stage === scriptName ? body : manifest.scripts[stage];
    if (!stageBody) continue;
    const command = stage === scriptName && extraArgs.length ? `${stageBody} ${extraArgs.join(' ')}` : stageBody;
    output.push(`> ${manifest.name}@${manifest.version} ${stage}`);
    output.push(`> ${command}`);
    output.push('');
    // A script that only calls back into the runner keeps hook semantics.
    const nested = /^(npm|pnpm|yarn|bun)\s+run\s+([\w:.-]+)\s*$/.exec(command.trim());
    if (nested) {
      output.push(await runScript(environment, nested[1] as PackageManagerKind, nested[2]!, [], depth + 1));
      continue;
    }
    const result = await runCommandChain(environment, command);
    if (result.stdout) output.push(result.stdout);
    if (result.exitCode !== 0) {
      throw new ScriptError([
        ...output,
        '',
        `${manager} error Lifecycle script \`${stage}\` failed with error:`,
        `${manager} error code ${result.exitCode}`,
        `${manager} error path ${environment.context.cwd}`,
        `${manager} error command failed`,
        `${manager} error command sh -c ${result.failedCommand ?? command}`,
      ].join('\n'), result.exitCode);
    }
  }
  return output.join('\n').trimEnd();
}

/** `<manager> run` with no script name lists what the project offers. */
export function listScripts(manager: PackageManagerKind, manifest: ScriptManifest): string {
  const names = Object.keys(manifest.scripts);
  if (!names.length) return `Scripts available in ${manifest.name}@${manifest.version}: (none)`;
  return [
    `Scripts available in ${manifest.name}@${manifest.version} via \`${manager} run-script\`:`,
    ...names.flatMap((name) => [`  ${name}`, `    ${manifest.scripts[name]}`]),
  ].join('\n');
}
