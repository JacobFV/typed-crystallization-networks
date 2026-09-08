import type { PackageManagerKind } from '@tcn-computer/protocol';
import { availableVersions, findEntry } from './catalog.js';
import { rangeOptionsFor } from './specifier.js';
import type { CatalogPackage, PackageSpecifier } from './types.js';
import { compareVersions, maxSatisfyingAll, satisfies } from './versions.js';

/**
 * Dependency resolution.
 *
 * Requirements accumulate per package name; a package is only resolvable if one
 * published version satisfies *every* accumulated requirement at once. Picking a
 * version can introduce new requirements, so the closure is computed to a
 * fixpoint rather than one level deep, and a version change invalidates the
 * requirements its previous choice contributed.
 */

export interface Requirement {
  range: string;
  /** `<root>` for a user request, otherwise `name@version` of the dependent. */
  by: string;
}

export interface ResolvedPackage {
  name: string;
  version: string;
  entry: CatalogPackage;
  direct: boolean;
  /** Range recorded on the record; the user's ask for direct packages. */
  requestedRange: string;
  dependencyRanges: Record<string, string>;
  /** Names this package pulled in. */
  dependencies: string[];
  /** Already present at this exact version, so the plan is a no-op for it. */
  satisfied: boolean;
}

export type ResolutionProblem =
  | { kind: 'missing'; name: string; requiredBy: string[] }
  | { kind: 'unsatisfiable'; name: string; requirements: Requirement[]; available: string[] }
  | { kind: 'conflict'; name: string; version: string; conflictsWith: string; range: string; requiredBy: string };

export interface ResolutionResult {
  /** Install order: dependencies before dependents. */
  plan: ResolvedPackage[];
  problems: ResolutionProblem[];
}

export interface ResolveOptions {
  manager: PackageManagerKind;
  /** Packages the user asked for. */
  requests: readonly PackageSpecifier[];
  /** Already installed in the same manager/scope/root: name → version. */
  installed?: ReadonlyMap<string, string>;
  /** Follow `dependencies` transitively. `false` installs only what was named. */
  transitive?: boolean;
}

const MAX_ITERATIONS = 200;

/** Human-readable explanation of a resolution failure, in the manager's voice. */
export function describeProblem(manager: PackageManagerKind, problem: ResolutionProblem): string {
  if (problem.kind === 'missing') {
    const by = problem.requiredBy.filter((item) => item !== '<root>');
    return by.length
      ? `${manager}: unable to locate package ${problem.name} (required by ${by.join(', ')})`
      : `${manager}: unable to locate package ${problem.name}`;
  }
  if (problem.kind === 'unsatisfiable') {
    const lines = problem.requirements.map((requirement) => `    ${requirement.by === '<root>' ? 'requested' : requirement.by} requires ${problem.name}@${requirement.range}`);
    return [
      `${manager}: could not resolve dependencies for ${problem.name}`,
      ...lines,
      `    available versions: ${problem.available.join(', ') || 'none'}`,
    ].join('\n');
  }
  return `${manager}: ${problem.name}@${problem.version} conflicts with ${problem.conflictsWith}${problem.range === '*' ? '' : `@${problem.range}`} (pulled in by ${problem.requiredBy})`;
}

export function resolve(options: ResolveOptions): ResolutionResult {
  const { manager, requests } = options;
  const transitive = options.transitive !== false;
  const installed = options.installed ?? new Map<string, string>();
  const grammar = rangeOptionsFor(manager);

  const requirements = new Map<string, Requirement[]>();
  const chosen = new Map<string, { version: string; entry: CatalogPackage }>();
  const direct = new Set<string>();
  const problems: ResolutionProblem[] = [];
  const missing = new Map<string, Set<string>>();

  const addRequirement = (name: string, requirement: Requirement): void => {
    const key = name.toLowerCase();
    const list = requirements.get(key) ?? [];
    if (!list.some((item) => item.range === requirement.range && item.by === requirement.by)) list.push(requirement);
    requirements.set(key, list);
  };

  for (const request of requests) {
    if (!request.name) continue;
    const entry = findEntry(manager, request.name);
    direct.add((entry?.name ?? request.name).toLowerCase());
    addRequirement(entry?.name ?? request.name, { range: request.range, by: '<root>' });
  }

  // Fixpoint: choose a version for every open requirement, then expand it.
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
    let changed = false;
    for (const [key, list] of [...requirements]) {
      const entry = findEntry(manager, key);
      if (!entry) {
        const by = missing.get(key) ?? new Set<string>();
        for (const requirement of list) by.add(requirement.by);
        missing.set(key, by);
        continue;
      }
      const versions = availableVersions(entry).sort(compareVersions);
      const ranges = list.map((requirement) => requirement.range);
      // Keep what is already installed when it still satisfies everything, the
      // way a real manager avoids gratuitous churn.
      const current = installed.get(entry.name) ?? installed.get(key);
      const pick = current && ranges.every((range) => satisfies(current, range, grammar))
        ? current
        : maxSatisfyingAll(versions, ranges, grammar);
      if (!pick) {
        if (!problems.some((problem) => problem.kind === 'unsatisfiable' && problem.name === entry.name)) {
          problems.push({ kind: 'unsatisfiable', name: entry.name, requirements: [...list], available: versions });
        }
        chosen.delete(key);
        continue;
      }
      const previous = chosen.get(key);
      if (previous?.version === pick) continue;
      chosen.set(key, { version: pick, entry });
      changed = true;
      if (!transitive) continue;
      // A version change re-contributes this package's dependency ranges. Stale
      // requirements from the previous pick are dropped by rebuilding the `by`.
      const by = `${entry.name}@${pick}`;
      for (const [, list2] of requirements) {
        for (let index = list2.length - 1; index >= 0; index -= 1) {
          if (list2[index]!.by.startsWith(`${entry.name}@`) && list2[index]!.by !== by) list2.splice(index, 1);
        }
      }
      for (const [dependency, range] of Object.entries(entry.dependencies ?? {})) addRequirement(dependency, { range, by });
    }
    if (!changed) break;
  }

  for (const [name, by] of missing) {
    const entryName = requests.find((request) => request.name.toLowerCase() === name)?.name ?? name;
    problems.push({ kind: 'missing', name: entryName, requiredBy: [...by] });
  }

  // Explicit incompatibilities, checked against the whole resolved set plus
  // whatever is already on the machine.
  const finalVersions = new Map<string, string>(installed);
  for (const [, value] of chosen) finalVersions.set(value.entry.name, value.version);
  for (const [, value] of chosen) {
    for (const [other, range] of Object.entries(value.entry.conflicts ?? {})) {
      const otherVersion = finalVersions.get(other) ?? finalVersions.get(findEntry(manager, other)?.name ?? other);
      if (otherVersion === undefined) continue;
      if (range !== '*' && !satisfies(otherVersion, range, grammar)) continue;
      problems.push({
        kind: 'conflict', name: value.entry.name, version: value.version,
        conflictsWith: other, range: range === '*' ? otherVersion : range,
        requiredBy: direct.has(value.entry.name.toLowerCase()) ? 'command line' : 'dependency graph',
      });
    }
  }

  const plan: ResolvedPackage[] = [];
  const visiting = new Set<string>();
  const emitted = new Set<string>();
  const emit = (key: string): void => {
    if (emitted.has(key) || visiting.has(key)) return;
    const value = chosen.get(key);
    if (!value) return;
    visiting.add(key);
    const dependencyNames: string[] = [];
    for (const dependency of Object.keys(value.entry.dependencies ?? {})) {
      const dependencyKey = (findEntry(manager, dependency)?.name ?? dependency).toLowerCase();
      if (chosen.has(dependencyKey)) {
        emit(dependencyKey);
        dependencyNames.push(chosen.get(dependencyKey)!.entry.name);
      }
    }
    visiting.delete(key);
    emitted.add(key);
    const requestedRange = direct.has(key)
      ? requirements.get(key)?.find((requirement) => requirement.by === '<root>')?.range ?? '*'
      : requirements.get(key)?.[0]?.range ?? '*';
    plan.push({
      name: value.entry.name, version: value.version, entry: value.entry, direct: direct.has(key), requestedRange,
      dependencyRanges: { ...(value.entry.dependencies ?? {}) }, dependencies: dependencyNames,
      satisfied: installed.get(value.entry.name) === value.version,
    });
  };
  for (const key of chosen.keys()) emit(key);

  return { plan, problems };
}

/**
 * Reference counting for autoremove: a transitive package is orphaned when no
 * remaining direct package reaches it through the dependency graph.
 */
export function findOrphans(
  packages: readonly { name: string; dependencies: readonly string[]; dependencyType: 'direct' | 'transitive' }[],
): string[] {
  const byName = new Map(packages.map((item) => [item.name, item]));
  const reachable = new Set<string>();
  const walk = (name: string): void => {
    if (reachable.has(name)) return;
    reachable.add(name);
    for (const dependency of byName.get(name)?.dependencies ?? []) walk(dependency);
  };
  for (const item of packages) if (item.dependencyType === 'direct') walk(item.name);
  return packages.filter((item) => item.dependencyType === 'transitive' && !reachable.has(item.name)).map((item) => item.name);
}
