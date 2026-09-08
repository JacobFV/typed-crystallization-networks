import { randomUUID } from './determinism.js';
import { lookup } from 'node:dns/promises';
import { Agent as HttpAgent, request as httpRequest } from 'node:http';
import { Agent as HttpsAgent, request as httpsRequest } from 'node:https';
import type {
  AddressFamily, ComputerSpec, DatagramMessage, DatagramResult, DnsNegativeRecord, DnsRecord, DnsResolution,
  DnsResourceRecord, DnsRecordType, FabricNetworkConfig, GatewayRule, LinkProfile, LinkRule, NatBindingRecord,
  NeighborRecord, NetworkInterfaceRecord, NetworkPacketTrace, PortBindingRecord, Protocol, RouteRecord,
  SocketRecord, StreamConnectionInfo, VirtualHttpResponse,
} from '@tcn-computer/protocol';

/* -------------------------------------------------------------------------- *
 * Address arithmetic (IPv4 + IPv6)
 * -------------------------------------------------------------------------- */

function ipv4Number(value: string): number | undefined {
  const octets = value.split('.');
  if (octets.length !== 4) return undefined;
  const parsed = octets.map((octet) => Number(octet));
  if (parsed.some((octet) => !Number.isInteger(octet) || octet < 0 || octet > 255)) return undefined;
  if (octets.some((octet) => octet.trim() === '' || !/^\d+$/.test(octet))) return undefined;
  return parsed.reduce((result, octet) => (result * 256 + octet) >>> 0, 0);
}

function ipv4Text(value: number): string {
  return [24, 16, 8, 0].map((shift) => (value >>> shift) & 0xff).join('.');
}

/** Strips brackets and a zone id so `[fe80::1%eth0]` parses like `fe80::1`. */
function stripAddressDecorations(value: string): string {
  let text = value.trim().toLowerCase();
  if (text.startsWith('[') && text.endsWith(']')) text = text.slice(1, -1);
  const zone = text.indexOf('%');
  return zone === -1 ? text : text.slice(0, zone);
}

/** Full RFC 4291 textual parser: `::` compression and embedded IPv4 included. */
export function ipv6Number(value: string): bigint | undefined {
  let text = stripAddressDecorations(value);
  if (!text.includes(':')) return undefined;
  const lastColon = text.lastIndexOf(':');
  const tail = text.slice(lastColon + 1);
  if (tail.includes('.')) {
    const embedded = ipv4Number(tail);
    if (embedded === undefined) return undefined;
    text = `${text.slice(0, lastColon + 1)}${((embedded >>> 16) & 0xffff).toString(16)}:${(embedded & 0xffff).toString(16)}`;
  }
  const halves = text.split('::');
  if (halves.length > 2) return undefined;
  const head = halves[0] ? halves[0].split(':') : [];
  const rest = halves.length === 2 ? (halves[1] ? halves[1].split(':') : []) : undefined;
  let groups: string[];
  if (rest === undefined) {
    if (head.length !== 8) return undefined;
    groups = head;
  } else {
    if (head.length + rest.length > 7) return undefined;
    groups = [...head, ...Array.from({ length: 8 - head.length - rest.length }, () => '0'), ...rest];
  }
  let result = 0n;
  for (const group of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(group)) return undefined;
    result = (result << 16n) | BigInt(Number.parseInt(group, 16));
  }
  return result;
}

/** RFC 5952 canonical text form. */
export function formatIpv6(value: bigint): string {
  const groups = Array.from({ length: 8 }, (_, index) => Number((value >> BigInt((7 - index) * 16)) & 0xffffn));
  let bestStart = -1;
  let bestLength = 0;
  let runStart = -1;
  for (let index = 0; index <= groups.length; index += 1) {
    if (index < groups.length && groups[index] === 0) {
      if (runStart === -1) runStart = index;
      continue;
    }
    if (runStart !== -1) {
      const length = index - runStart;
      if (length > bestLength) { bestLength = length; bestStart = runStart; }
      runStart = -1;
    }
  }
  if (bestLength < 2) return groups.map((group) => group.toString(16)).join(':');
  const head = groups.slice(0, bestStart).map((group) => group.toString(16)).join(':');
  const tail = groups.slice(bestStart + bestLength).map((group) => group.toString(16)).join(':');
  return `${head}::${tail}`;
}

export function addressFamily(value: string): AddressFamily | undefined {
  if (ipv4Number(value) !== undefined) return 4;
  return ipv6Number(value) === undefined ? undefined : 6;
}

/** Canonical text for either family, or `undefined` when `value` is not an IP. */
export function normalizeAddress(value: string): string | undefined {
  const stripped = stripAddressDecorations(value);
  const v4 = ipv4Number(stripped);
  if (v4 !== undefined) return ipv4Text(v4);
  const v6 = ipv6Number(stripped);
  return v6 === undefined ? undefined : formatIpv6(v6);
}

const V4_MAPPED_PREFIX = 0xffff00000000n;

/** `::ffff:a.b.c.d` → the embedded IPv4, so a v4 rule still governs it. */
function mappedIpv4(value: string): number | undefined {
  const v6 = ipv6Number(value);
  if (v6 === undefined) return undefined;
  if (v6 >> 32n !== V4_MAPPED_PREFIX >> 32n) return undefined;
  return Number(v6 & 0xffffffffn) >>> 0;
}

function ipv6Mask(prefix: number): bigint {
  if (prefix <= 0) return 0n;
  return ((1n << BigInt(prefix)) - 1n) << BigInt(128 - prefix);
}

/**
 * Pure helper used by the gateway policy and its acceptance tests.
 *
 * IPv4 behaviour is byte-for-byte what it always was. IPv6 networks are
 * compared with real 128-bit arithmetic, and an IPv4-mapped address is
 * unwrapped so a v4 CIDR still constrains it — a mapped address must never be
 * able to slip past a rule that would have rejected its v4 form.
 */
export function cidrContains(cidr: string, address: string): boolean {
  const slash = cidr.lastIndexOf('/');
  const network = slash === -1 ? cidr : cidr.slice(0, slash);
  const prefixText = slash === -1 ? undefined : cidr.slice(slash + 1);
  const networkV4 = ipv4Number(network);
  if (networkV4 !== undefined) {
    const prefix = prefixText === undefined ? 32 : Number(prefixText);
    const addressNumber = ipv4Number(address) ?? mappedIpv4(address);
    if (addressNumber === undefined || !Number.isInteger(prefix) || prefix < 0 || prefix > 32) return false;
    if (prefix === 0) return true;
    const mask = (0xffffffff << (32 - prefix)) >>> 0;
    return (networkV4 & mask) === (addressNumber & mask);
  }
  const networkV6 = ipv6Number(network);
  if (networkV6 === undefined) return false;
  const prefix = prefixText === undefined ? 128 : Number(prefixText);
  if (!Number.isInteger(prefix) || prefix < 0 || prefix > 128) return false;
  let addressNumber = ipv6Number(address);
  if (addressNumber === undefined) {
    // Only ::ffff:0:0/96 networks may match a bare IPv4 literal.
    const v4 = ipv4Number(address);
    if (v4 === undefined) return false;
    if ((networkV6 & ipv6Mask(96)) !== V4_MAPPED_PREFIX) return false;
    addressNumber = V4_MAPPED_PREFIX | BigInt(v4);
  }
  const mask = ipv6Mask(prefix);
  return (networkV6 & mask) === (addressNumber & mask);
}

function hostnameMatches(pattern: string, hostname: string): boolean {
  const expected = pattern.toLowerCase();
  const actual = hostname.toLowerCase();
  if (expected === '*') return true;
  if (expected.startsWith('*.')) return actual.endsWith(expected.slice(1)) && actual !== expected.slice(2);
  return expected === actual;
}

const LOOPBACK_NAMES = new Set(['localhost', 'localhost.local', 'localhost.localdomain']);

function normalizeHost(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/\.$/, '');
  return normalized.startsWith('[') && normalized.endsWith(']') ? normalized.slice(1, -1) : normalized;
}

function isLoopbackHost(value: string): boolean {
  const host = normalizeHost(value);
  return LOOPBACK_NAMES.has(host) || host === '::1' || host.startsWith('127.');
}

function isWildcardHost(value: string): boolean {
  const host = normalizeHost(value);
  return host === '*' || host === '0.0.0.0' || host === '::';
}

function loopbackAddress(value: string): string {
  const host = normalizeHost(value);
  if (host === '::1') return '::1';
  if (host.startsWith('127.')) return host;
  return '127.0.0.1';
}

type ServiceBinding = 'loopback' | 'wildcard' | 'network';

function serviceBinding(host: string): ServiceBinding {
  if (isLoopbackHost(host)) return 'loopback';
  if (isWildcardHost(host)) return 'wildcard';
  return 'network';
}

/* -------------------------------------------------------------------------- *
 * Deterministic randomness
 * -------------------------------------------------------------------------- */

/**
 * mulberry32. Every stochastic decision in the fabric (jitter, loss, initial
 * sequence numbers) draws from a seeded stream so a run is reproducible;
 * `Math.random` is deliberately never used.
 */
export class SeededRandom {
  private state: number;

  constructor(seed: number) { this.state = (seed >>> 0) || 0x9e3779b9; }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  int(bound: number): number { return Math.floor(this.next() * bound); }
}

/* -------------------------------------------------------------------------- *
 * Transport model
 * -------------------------------------------------------------------------- */

/** Today's deterministic behaviour: 0.42 ms RTT, no jitter, no loss. */
export const DEFAULT_LINK_PROFILE: LinkProfile = {
  latencyMs: 0.42, jitterMs: 0, lossRate: 0, bandwidthBps: 0, mtu: 1500, fragment: false,
};

export const DEFAULT_NETWORK_CONFIG: FabricNetworkConfig = {
  cidr: '10.42.0.0/24', gateway: '10.42.0.1', dns: '10.42.0.2', domain: 'seed.local',
  publicAddress: '198.51.100.1', ipv6Prefix: 'fd42::',
};

const RECEIVE_BUFFER = 65535;
const DEFAULT_MSS = 1460;

/** Mutable TCP connection state so traces carry real seq/ack/window values. */
class TcpFlow {
  clientSeq: number;
  serverSeq: number;
  clientWindow = RECEIVE_BUFFER;
  serverWindow = RECEIVE_BUFFER;

  constructor(clientIsn: number, serverIsn: number) {
    this.clientSeq = clientIsn;
    this.serverSeq = serverIsn;
  }

  advanceClient(bytes: number): void {
    this.clientSeq = (this.clientSeq + bytes) >>> 0;
    this.serverWindow = Math.max(1024, RECEIVE_BUFFER - bytes);
  }

  advanceServer(bytes: number): void {
    this.serverSeq = (this.serverSeq + bytes) >>> 0;
    this.clientWindow = Math.max(1024, RECEIVE_BUFFER - bytes);
  }
}

/* -------------------------------------------------------------------------- *
 * Services
 * -------------------------------------------------------------------------- */

export interface VirtualService {
  id: string;
  computerId: string;
  host: string;
  port: number;
  protocol: 'http' | 'https';
  pid: number;
  handle(path: string, method: string, body?: string): Promise<Omit<VirtualHttpResponse, 'traceId'>>;
}

/** A UDP listener. Returning a string sends a reply datagram to the sender. */
export interface DatagramService {
  id: string;
  computerId: string;
  host: string;
  port: number;
  pid?: number;
  handle(message: DatagramMessage): Promise<string | undefined> | string | undefined;
}

/** A byte-stream (non-HTTP) TCP listener. */
export interface StreamService {
  id: string;
  computerId: string;
  host: string;
  port: number;
  pid?: number;
  onConnect?(connection: StreamConnectionInfo): Promise<string | undefined> | string | undefined;
  handle(data: string, connection: StreamConnectionInfo): Promise<string | undefined> | string | undefined;
  onClose?(connection: StreamConnectionInfo): void;
}

export interface StreamConnection extends StreamConnectionInfo {
  send(data: string): Promise<string | undefined>;
  close(): void;
  readonly closed: boolean;
}

export interface NetworkBindingHandle { close(): void }

/* -------------------------------------------------------------------------- *
 * Pinned egress transport
 * -------------------------------------------------------------------------- */

export interface PinnedEgressRequest {
  /** The single address that was validated by the gateway policy. */
  address: string;
  hostname: string;
  port: number;
  protocol: 'http' | 'https';
  method: string;
  path: string;
  body?: string;
  headers: Record<string, string>;
  timeoutMs: number;
}

export interface PinnedEgressResponse {
  status: number;
  statusText: string;
  headers: Record<string, string>;
  body: string;
}

export type EgressTransport = (request: PinnedEgressRequest) => Promise<PinnedEgressResponse>;
export type ExternalResolver = (hostname: string) => Promise<string[]>;

type LookupCallback = (error: NodeJS.ErrnoException | null, address: string | Array<{ address: string; family: number }>, family?: number) => void;
type PinnedLookup = (hostname: string, options: unknown, callback: LookupCallback) => void;

/**
 * The heart of the rebinding fix: a `lookup` implementation that ignores the
 * hostname entirely and always yields the address the gateway validated. It is
 * installed on the outbound Agent, so the socket provably connects to the
 * checked address — the resolver cannot be consulted a second time.
 */
export function createPinnedLookup(address: string): PinnedLookup {
  const family = addressFamily(address) ?? 4;
  return (_hostname, options, callback) => {
    const all = typeof options === 'object' && options !== null && (options as { all?: boolean }).all === true;
    if (all) callback(null, [{ address, family }], family);
    else callback(null, address, family);
  };
}

async function nodeEgressTransport(request: PinnedEgressRequest): Promise<PinnedEgressResponse> {
  const secure = request.protocol === 'https';
  const lookupFn = createPinnedLookup(request.address) as unknown as never;
  const agent = secure
    ? new HttpsAgent({ keepAlive: false, lookup: lookupFn })
    : new HttpAgent({ keepAlive: false, lookup: lookupFn });
  return await new Promise<PinnedEgressResponse>((resolve, reject) => {
    const send = secure ? httpsRequest : httpRequest;
    const outbound = send({
      // `hostname` drives the Host header and (for TLS) SNI + certificate
      // validation; only the address resolution step is overridden.
      hostname: request.hostname,
      servername: secure ? request.hostname : undefined,
      port: request.port,
      path: request.path,
      method: request.method,
      headers: request.headers,
      agent,
      timeout: request.timeoutMs,
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on('data', (chunk: Buffer) => chunks.push(chunk));
      response.on('end', () => {
        const headers: Record<string, string> = {};
        for (const [key, value] of Object.entries(response.headers)) {
          if (value === undefined) continue;
          headers[key] = Array.isArray(value) ? value.join(', ') : String(value);
        }
        resolve({
          status: response.statusCode ?? 0,
          statusText: response.statusMessage ?? '',
          headers,
          body: Buffer.concat(chunks).toString('utf8'),
        });
      });
      response.on('error', reject);
    });
    outbound.on('timeout', () => outbound.destroy(new Error(`egress timeout: ${request.hostname}:${request.port}`)));
    outbound.on('error', reject);
    if (request.body !== undefined) outbound.write(request.body);
    outbound.end();
  }).finally(() => agent.destroy());
}

/* -------------------------------------------------------------------------- *
 * Internal records
 * -------------------------------------------------------------------------- */

interface InternalSocket extends SocketRecord {
  closedAt?: number;
  ephemeral?: boolean;
  owner?: string;
}

interface CachedRecord extends DnsResourceRecord { expiresAt: number }

interface ZoneRecordSet {
  A: DnsResourceRecord[];
  AAAA: DnsResourceRecord[];
  CNAME: DnsResourceRecord[];
  other: DnsResourceRecord[];
}

export interface NetworkRequestOptions {
  /** Follow 3xx responses, re-validating each hop through `canEgress`. */
  maxRedirects?: number;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

export interface InternetFabricOptions {
  /** Seed for jitter, loss and initial sequence numbers. */
  seed?: number;
  /** Injectable clock so TTL expiry and socket retention are testable. */
  clock?: () => number;
  network?: Partial<FabricNetworkConfig>;
  defaultLink?: Partial<LinkProfile>;
  /** Resolver consulted for non-virtual destinations. Defaults to node DNS. */
  externalResolver?: ExternalResolver;
  /** Transport used for validated egress. Defaults to a pinned-lookup Agent. */
  egressTransport?: EgressTransport;
  ephemeralPortRange?: [number, number];
  maxClosedSockets?: number;
  closedSocketTtlMs?: number;
  maxTraces?: number;
  negativeTtlSeconds?: number;
}

const CNAME_HOP_LIMIT = 8;

/* -------------------------------------------------------------------------- *
 * InternetFabric
 * -------------------------------------------------------------------------- */

export class InternetFabric {
  private readonly computers = new Map<string, ComputerSpec>();
  /** Authoritative zone data. Never expires: the fabric is the zone authority. */
  private readonly zoneRecords = new Map<string, ZoneRecordSet>();
  private readonly zones = new Set<string>();
  /** Cached (non-authoritative) answers, subject to TTL expiry. */
  private readonly cache = new Map<string, CachedRecord[]>();
  private readonly negativeCache = new Map<string, DnsNegativeRecord>();
  private readonly services = new Map<string, VirtualService>();
  private readonly datagramServices = new Map<string, DatagramService>();
  private readonly streamServices = new Map<string, StreamService>();
  private readonly socketRecords: InternalSocket[] = [];
  private readonly traces: NetworkPacketTrace[] = [];
  private readonly interfaces: NetworkInterfaceRecord[] = [];
  private readonly routes: RouteRecord[] = [];
  private readonly neighbors = new Map<string, NeighborRecord>();
  private readonly natBindings: NatBindingRecord[] = [];
  private readonly links: LinkRule[] = [];
  /** (computerId, transport, address, port) → binding. Enforces EADDRINUSE. */
  private readonly portBindings = new Map<string, PortBindingRecord>();
  private readonly ephemeralCursor = new Map<string, number>();
  private readonly random: SeededRandom;
  private readonly now: () => number;
  private readonly options: Required<Omit<InternetFabricOptions, 'network' | 'defaultLink' | 'externalResolver' | 'egressTransport' | 'clock' | 'seed'>>;
  private readonly externalResolver: ExternalResolver;
  private readonly egressTransport: EgressTransport;
  private config: FabricNetworkConfig;
  private defaultLink: LinkProfile;

  readonly gateways: GatewayRule[] = [];

  constructor(options: InternetFabricOptions = {}) {
    this.random = new SeededRandom(options.seed ?? 0x5eed2026);
    this.now = options.clock ?? (() => Date.now());
    this.config = { ...DEFAULT_NETWORK_CONFIG, ...options.network };
    this.defaultLink = { ...DEFAULT_LINK_PROFILE, ...options.defaultLink };
    this.externalResolver = options.externalResolver ?? (async (hostname) => {
      try { return (await lookup(hostname, { all: true, verbatim: true })).map((result) => result.address); }
      catch { return []; }
    });
    this.egressTransport = options.egressTransport ?? nodeEgressTransport;
    this.options = {
      ephemeralPortRange: options.ephemeralPortRange ?? [49152, 65535],
      maxClosedSockets: options.maxClosedSockets ?? 128,
      closedSocketTtlMs: options.closedSocketTtlMs ?? 5 * 60_000,
      maxTraces: options.maxTraces ?? 300,
      negativeTtlSeconds: options.negativeTtlSeconds ?? 30,
    };
    this.zones.add(normalizeHost(this.config.domain));
  }

  /* ---------------------------------------------------------------------- *
   * Topology / interfaces / routing
   * ---------------------------------------------------------------------- */

  /** Applies topology-level network parameters. Safe to call after `attach`. */
  configureNetwork(config: Partial<FabricNetworkConfig>): FabricNetworkConfig {
    this.config = { ...this.config, ...config };
    this.zones.add(normalizeHost(this.config.domain));
    for (const spec of this.computers.values()) this.provisionInterfaces(spec);
    return { ...this.config };
  }

  networkConfig(): FabricNetworkConfig { return { ...this.config }; }

  /** Registers (or replaces) a link profile. Later rules win ties. */
  configureLink(rule: { id?: string; from?: string; to?: string; profile: Partial<LinkProfile> }): LinkRule {
    const entry: LinkRule = {
      id: rule.id ?? `link-${this.links.length + 1}`,
      from: rule.from ?? '*',
      to: rule.to ?? '*',
      profile: { ...this.defaultLink, ...rule.profile },
    };
    const existing = this.links.findIndex((candidate) => candidate.id === entry.id);
    if (existing === -1) this.links.push(entry); else this.links[existing] = entry;
    return { ...entry, profile: { ...entry.profile } };
  }

  setDefaultLink(profile: Partial<LinkProfile>): LinkProfile {
    this.defaultLink = { ...this.defaultLink, ...profile };
    return { ...this.defaultLink };
  }

  listLinks(): LinkRule[] { return this.links.map((link) => ({ ...link, profile: { ...link.profile } })); }

  /** Most specific match wins: exact pair, then one wildcard, then default. */
  linkProfile(from: string, to: string): LinkProfile {
    let best: LinkRule | undefined;
    let bestScore = -1;
    for (const link of this.links) {
      const fromMatch = link.from === '*' ? 0 : link.from === from ? 2 : -1;
      const toMatch = link.to === '*' ? 0 : link.to === to ? 2 : -1;
      if (fromMatch < 0 || toMatch < 0) continue;
      const score = fromMatch + toMatch;
      if (score >= bestScore) { bestScore = score; best = link; }
    }
    return best ? { ...best.profile } : { ...this.defaultLink };
  }

  attach(spec: ComputerSpec, domain = this.config.domain): void {
    this.computers.set(spec.id, spec);
    this.zones.add(normalizeHost(domain));
    this.provisionInterfaces(spec);
    this.addDns(spec.hostname, spec.ipv4);
    this.addDns(`${spec.hostname}.${domain}`, spec.ipv4);
    const v6 = this.ipv6For(spec.ipv4);
    if (v6) {
      this.addRecord({ name: spec.hostname, type: 'AAAA', value: v6 });
      this.addRecord({ name: `${spec.hostname}.${domain}`, type: 'AAAA', value: v6 });
    }
  }

  /** Deterministic ULA address derived from the computer's IPv4. */
  private ipv6For(ipv4: string): string | undefined {
    const value = ipv4Number(ipv4);
    if (value === undefined) return undefined;
    const prefix = ipv6Number(this.config.ipv6Prefix);
    if (prefix === undefined) return undefined;
    return formatIpv6((prefix & ipv6Mask(64)) | BigInt(value));
  }

  private macFor(ipv4: string): string {
    const value = ipv4Number(ipv4) ?? 0;
    const octets = [24, 16, 8, 0].map((shift) => ((value >>> shift) & 0xff).toString(16).padStart(2, '0'));
    return ['02', '42', ...octets].join(':');
  }

  private provisionInterfaces(spec: ComputerSpec): void {
    for (let index = this.interfaces.length - 1; index >= 0; index -= 1) {
      if (this.interfaces[index]!.computerId === spec.id) this.interfaces.splice(index, 1);
    }
    for (let index = this.routes.length - 1; index >= 0; index -= 1) {
      if (this.routes[index]!.computerId === spec.id && this.routes[index]!.source !== 'static') this.routes.splice(index, 1);
    }
    const primaryName = spec.os === 'windows' ? 'Ethernet' : spec.os === 'macos' ? 'en0' : 'seed0';
    const prefix = Number(this.config.cidr.split('/')[1] ?? 24);
    const v6 = this.ipv6For(spec.ipv4);
    const v6Prefix = ipv6Number(this.config.ipv6Prefix);
    this.interfaces.push({
      id: `${spec.id}:lo`, computerId: spec.id, name: 'lo', mac: '00:00:00:00:00:00', mtu: 65536, up: true, loopback: true,
      addresses: [
        { address: '127.0.0.1', prefix: 8, family: 4, scope: 'host' },
        { address: '::1', prefix: 128, family: 6, scope: 'host' },
      ],
    });
    this.interfaces.push({
      id: `${spec.id}:${primaryName}`, computerId: spec.id, name: primaryName, mac: this.macFor(spec.ipv4),
      mtu: this.defaultLink.mtu, up: true, loopback: false,
      addresses: [
        { address: spec.ipv4, prefix, family: 4, scope: 'global' },
        ...(v6 ? [{ address: v6, prefix: 64, family: 6 as AddressFamily, scope: 'global' as const }] : []),
      ],
    });
    this.routes.push({ id: `${spec.id}:lo4`, computerId: spec.id, destination: '127.0.0.0/8', dev: 'lo', metric: 0, family: 4, source: 'kernel' });
    this.routes.push({ id: `${spec.id}:lo6`, computerId: spec.id, destination: '::1/128', dev: 'lo', metric: 0, family: 6, source: 'kernel' });
    this.routes.push({ id: `${spec.id}:onlink4`, computerId: spec.id, destination: this.config.cidr, dev: primaryName, metric: 100, family: 4, source: 'kernel' });
    if (v6 && v6Prefix !== undefined) this.routes.push({ id: `${spec.id}:onlink6`, computerId: spec.id, destination: `${formatIpv6(v6Prefix & ipv6Mask(64))}/64`, dev: primaryName, metric: 100, family: 6, source: 'kernel' });
    this.routes.push({ id: `${spec.id}:default4`, computerId: spec.id, destination: '0.0.0.0/0', via: this.config.gateway, dev: primaryName, metric: 200, family: 4, source: 'dhcp' });
    this.neighbors.set(`${spec.id}\u0000${this.config.gateway}`, {
      computerId: spec.id, address: this.config.gateway, mac: this.macFor(this.config.gateway), dev: primaryName,
      family: 4, state: 'PERMANENT', resolvedBy: 'static',
    });
  }

  listInterfaces(computerId?: string): NetworkInterfaceRecord[] {
    return this.interfaces
      .filter((entry) => !computerId || entry.computerId === computerId)
      .map((entry) => ({ ...entry, addresses: entry.addresses.map((address) => ({ ...address })) }));
  }

  /**
   * `ifconfig`-style lines built from the real interface, route and resolver
   * model. Shells should render this instead of hard-coding a gateway address.
   */
  interfaceSummary(computerId: string): string {
    const entries = this.interfaces.filter((entry) => entry.computerId === computerId && !entry.loopback);
    return entries.map((entry) => {
      const route = this.routes.find((candidate) => candidate.computerId === computerId && candidate.destination === '0.0.0.0/0' && candidate.dev === entry.name);
      const lines = [entry.name, `  ether ${entry.mac}`];
      for (const address of entry.addresses) lines.push(`  ${address.family === 4 ? 'inet' : 'inet6'} ${address.address}/${address.prefix}`);
      if (route?.via) lines.push(`  gateway ${route.via}`);
      lines.push(`  dns ${this.config.dns}`, `  mtu ${entry.mtu}`, `  state ${entry.up ? 'UP' : 'DOWN'}`);
      return lines.join('\n');
    }).join('\n');
  }

  setInterfaceUp(computerId: string, name: string, up: boolean): NetworkInterfaceRecord {
    const entry = this.interfaces.find((candidate) => candidate.computerId === computerId && candidate.name === name);
    if (!entry) throw new Error(`unknown interface: ${computerId}/${name}`);
    entry.up = up;
    return { ...entry, addresses: entry.addresses.map((address) => ({ ...address })) };
  }

  addRoute(route: Omit<RouteRecord, 'id' | 'source'> & { id?: string; source?: RouteRecord['source'] }): RouteRecord {
    const entry: RouteRecord = { ...route, id: route.id ?? randomUUID(), source: route.source ?? 'static' };
    if (!this.computers.has(entry.computerId)) throw new Error(`unknown computer: ${entry.computerId}`);
    this.routes.push(entry);
    return { ...entry };
  }

  removeRoute(id: string): boolean {
    const index = this.routes.findIndex((route) => route.id === id);
    if (index === -1) return false;
    this.routes.splice(index, 1);
    return true;
  }

  listRoutes(computerId?: string): RouteRecord[] {
    return this.routes.filter((route) => !computerId || route.computerId === computerId).map((route) => ({ ...route }));
  }

  /** Longest-prefix match, tie-broken by metric. */
  routeFor(computerId: string, address: string): RouteRecord | undefined {
    const family = addressFamily(address);
    if (family === undefined) return undefined;
    let best: RouteRecord | undefined;
    let bestPrefix = -1;
    for (const route of this.routes) {
      if (route.computerId !== computerId || route.family !== family) continue;
      const device = this.interfaces.find((entry) => entry.computerId === computerId && entry.name === route.dev);
      if (device && !device.up) continue;
      if (!cidrContains(route.destination, address)) continue;
      const prefix = Number(route.destination.split('/')[1] ?? (family === 4 ? 32 : 128));
      if (prefix > bestPrefix || (prefix === bestPrefix && best && route.metric < best.metric)) { bestPrefix = prefix; best = route; }
    }
    return best ? { ...best } : undefined;
  }

  /**
   * ARP (IPv4) / NDP (IPv6) resolution for the next hop. On-link peers answer
   * with their own MAC; off-link destinations resolve to the default gateway.
   */
  resolveNeighbor(computerId: string, address: string): NeighborRecord | undefined {
    const route = this.routeFor(computerId, address);
    if (!route) return undefined;
    const nextHop = route.via ?? address;
    const key = `${computerId}\u0000${nextHop}`;
    const cached = this.neighbors.get(key);
    if (cached && cached.state !== 'INCOMPLETE') return { ...cached };
    const family = addressFamily(nextHop) ?? 4;
    const peer = this.computerByAddress(nextHop);
    const record: NeighborRecord = {
      computerId, address: nextHop, dev: route.dev, family,
      mac: peer ? this.macFor(peer.ipv4) : this.macFor(nextHop),
      state: 'REACHABLE', resolvedBy: family === 4 ? 'arp' : 'ndp',
    };
    this.neighbors.set(key, record);
    return { ...record };
  }

  listNeighbors(computerId?: string): NeighborRecord[] {
    return [...this.neighbors.values()].filter((entry) => !computerId || entry.computerId === computerId).map((entry) => ({ ...entry }));
  }

  private computerByAddress(address: string): ComputerSpec | undefined {
    const canonical = normalizeAddress(address) ?? address;
    for (const spec of this.computers.values()) {
      if (spec.ipv4 === canonical) return spec;
      const owned = this.interfaces.some((entry) => entry.computerId === spec.id && !entry.loopback
        && entry.addresses.some((candidate) => candidate.address === canonical));
      if (owned) return spec;
    }
    return undefined;
  }

  /* ---------------------------------------------------------------------- *
   * DNS
   * ---------------------------------------------------------------------- */

  private zoneOf(name: string): string | undefined {
    let best: string | undefined;
    for (const zone of this.zones) {
      if (name === zone || name.endsWith(`.${zone}`)) { if (!best || zone.length > best.length) best = zone; }
    }
    return best;
  }

  addZone(origin: string): void { this.zones.add(normalizeHost(origin)); }

  listZones(): string[] { return [...this.zones]; }

  /**
   * Publishes an authoritative A record. Loopback names are resolved from each
   * computer's hosts namespace. They must never enter the shared DNS zone,
   * where one computer could overwrite another computer's localhost mapping.
   */
  addDns(name: string, value: string, ttl = 300): void {
    this.addRecord({ name, type: 'A', value, ttl });
  }

  /** General record publication. `authoritative: false` makes the entry expire. */
  addRecord(record: { name: string; type: DnsRecordType; value: string; ttl?: number; authoritative?: boolean }): void {
    const normalized = normalizeHost(record.name);
    if (isLoopbackHost(normalized) || isWildcardHost(normalized)) return;
    // A bare address is already its own answer; it never becomes a zone name.
    if (normalizeAddress(normalized) !== undefined) return;
    const ttl = record.ttl ?? 300;
    const authoritative = record.authoritative ?? true;
    this.negativeCache.delete(normalized);
    if (!authoritative) {
      const entry: CachedRecord = { name: normalized, type: record.type, value: record.value, ttl, authoritative: false, expiresAt: this.now() + ttl * 1000 };
      const bucket = this.cache.get(normalized) ?? [];
      this.cache.set(normalized, [...bucket.filter((candidate) => candidate.type !== record.type || candidate.value !== record.value), entry]);
      return;
    }
    const zone = this.zoneOf(normalized);
    const entry: DnsResourceRecord = { name: normalized, type: record.type, value: record.value, ttl, authoritative: true, zone };
    const set = this.zoneRecords.get(normalized) ?? { A: [], AAAA: [], CNAME: [], other: [] };
    if (record.type === 'A') set.A = [entry];
    else if (record.type === 'AAAA') set.AAAA = [entry];
    else if (record.type === 'CNAME') { set.CNAME = [entry]; set.A = []; set.AAAA = []; }
    else set.other = [...set.other.filter((candidate) => candidate.type !== record.type || candidate.value !== record.value), entry];
    this.zoneRecords.set(normalized, set);
    this.cache.delete(normalized);
  }

  /** Convenience for resolver-style entries that must honour their TTL. */
  cacheRecord(record: { name: string; type: DnsRecordType; value: string; ttl: number }): void {
    this.addRecord({ ...record, authoritative: false });
  }

  /** Remembers an NXDOMAIN for `negativeTtlSeconds` (RFC 2308 style). */
  cacheNegative(name: string, ttlSeconds = this.options.negativeTtlSeconds, reason: DnsNegativeRecord['reason'] = 'nxdomain'): void {
    const normalized = normalizeHost(name);
    if (isLoopbackHost(normalized) || isWildcardHost(normalized)) return;
    if (this.zoneRecords.has(normalized)) return;
    this.negativeCache.set(normalized, { name: normalized, type: 'ANY', ttl: ttlSeconds, expiresAt: this.now() + ttlSeconds * 1000, reason });
  }

  removeDns(name: string): void {
    const normalized = normalizeHost(name);
    this.zoneRecords.delete(normalized);
    this.cache.delete(normalized);
    this.negativeCache.delete(normalized);
  }

  /** Drops every expired cache and negative-cache entry. Returns the count. */
  expireCache(): number {
    const at = this.now();
    let removed = 0;
    for (const [name, records] of this.cache) {
      const live = records.filter((record) => record.expiresAt > at);
      removed += records.length - live.length;
      if (live.length === 0) this.cache.delete(name); else this.cache.set(name, live);
    }
    for (const [name, record] of this.negativeCache) {
      if (record.expiresAt <= at) { this.negativeCache.delete(name); removed += 1; }
    }
    return removed;
  }

  private liveCache(name: string): CachedRecord[] {
    const records = this.cache.get(name);
    if (!records) return [];
    const at = this.now();
    const live = records.filter((record) => record.expiresAt > at);
    if (live.length === records.length) return live;
    if (live.length === 0) this.cache.delete(name); else this.cache.set(name, live);
    return live;
  }

  private recordsFor(name: string): DnsResourceRecord[] {
    const set = this.zoneRecords.get(name);
    const authoritative = set ? [...set.CNAME, ...set.A, ...set.AAAA, ...set.other] : [];
    return [...authoritative, ...this.liveCache(name)];
  }

  /**
   * Full resolution: literal addresses, per-computer loopback, authoritative
   * zone data, TTL-bounded cache, CNAME chasing with loop detection and
   * negative caching.
   */
  resolveDetailed(name: string, options: { computerId?: string; family?: AddressFamily } = {}): DnsResolution {
    const normalized = normalizeHost(name);
    const chain: string[] = [normalized];
    if (isLoopbackHost(normalized)) {
      if (options.computerId && !this.computers.has(options.computerId)) throw new Error(`unknown computer: ${options.computerId}`);
      const address = loopbackAddress(normalized);
      return { name: normalized, status: 'ok', source: 'hosts', chain, records: [], addresses: [address] };
    }
    const literal = normalizeAddress(normalized);
    if (literal !== undefined) return { name: normalized, status: 'ok', source: 'literal', chain, records: [], addresses: [literal] };

    let current = normalized;
    const seen = new Set<string>([current]);
    for (let hop = 0; hop <= CNAME_HOP_LIMIT; hop += 1) {
      const records = this.recordsFor(current);
      if (records.length === 0) {
        const negative = this.negativeCache.get(current);
        if (negative && negative.expiresAt > this.now()) {
          return { name: normalized, status: 'nxdomain', source: 'negative-cache', chain, records: [], addresses: [], ttl: negative.ttl };
        }
        if (negative) this.negativeCache.delete(current);
        this.cacheNegative(current);
        return { name: normalized, status: 'nxdomain', source: 'none', chain, records: [], addresses: [] };
      }
      const cname = records.find((record) => record.type === 'CNAME');
      const addresses = records.filter((record) => record.type === 'A' || record.type === 'AAAA');
      if (addresses.length > 0) {
        const wanted = options.family === undefined
          ? addresses
          : addresses.filter((record) => record.type === (options.family === 6 ? 'AAAA' : 'A'));
        if (wanted.length === 0) {
          return { name: normalized, status: 'nodata', source: addresses[0]!.authoritative ? 'authoritative' : 'cache', chain, records: addresses, addresses: [] };
        }
        // Prefer A first so existing IPv4 callers keep their exact answer.
        const ordered = [...wanted.filter((record) => record.type === 'A'), ...wanted.filter((record) => record.type === 'AAAA')];
        return {
          name: normalized, status: 'ok', source: ordered[0]!.authoritative ? 'authoritative' : 'cache', chain,
          records: ordered.map((record) => ({ ...record })), addresses: ordered.map((record) => record.value),
          ttl: Math.min(...ordered.map((record) => record.ttl)),
        };
      }
      if (!cname) {
        return { name: normalized, status: 'nodata', source: records[0]!.authoritative ? 'authoritative' : 'cache', chain, records: records.map((record) => ({ ...record })), addresses: [] };
      }
      const target = normalizeHost(cname.value);
      chain.push(target);
      if (seen.has(target)) return { name: normalized, status: 'loop', source: 'authoritative', chain, records: [], addresses: [] };
      seen.add(target);
      current = target;
      const asAddress = normalizeAddress(target);
      if (asAddress !== undefined) return { name: normalized, status: 'ok', source: 'authoritative', chain, records: [{ ...cname }], addresses: [asAddress] };
    }
    return { name: normalized, status: 'loop', source: 'authoritative', chain, records: [], addresses: [] };
  }

  /**
   * Unchanged signature and semantics: the first IPv4 answer, per-computer
   * loopback, or `undefined`.
   */
  resolve(name: string, computerId?: string): string | undefined {
    const resolution = this.resolveDetailed(name, { computerId });
    return resolution.status === 'ok' ? resolution.addresses[0] : undefined;
  }

  resolveAll(name: string, options: { computerId?: string; family?: AddressFamily } = {}): string[] {
    const resolution = this.resolveDetailed(name, options);
    return resolution.status === 'ok' ? [...resolution.addresses] : [];
  }

  /** Backwards-compatible view: authoritative A/CNAME records only. */
  listDns(): DnsRecord[] {
    return this.listDnsRecords()
      .filter((record) => record.authoritative && (record.type === 'A' || record.type === 'CNAME'))
      .map((record) => ({ name: record.name, type: record.type as 'A' | 'CNAME', value: record.value, ttl: record.ttl }));
  }

  /** Every live record, authoritative and cached alike. */
  listDnsRecords(): DnsResourceRecord[] {
    this.expireCache();
    const result: DnsResourceRecord[] = [];
    for (const set of this.zoneRecords.values()) result.push(...[...set.A, ...set.AAAA, ...set.CNAME, ...set.other].map((record) => ({ ...record })));
    for (const records of this.cache.values()) result.push(...records.map((record) => ({ ...record })));
    return result;
  }

  listNegativeCache(): DnsNegativeRecord[] {
    this.expireCache();
    return [...this.negativeCache.values()].map((record) => ({ ...record }));
  }

  /* ---------------------------------------------------------------------- *
   * Port bindings and the ephemeral allocator
   * ---------------------------------------------------------------------- */

  private bindingKey(computerId: string, transport: 'tcp' | 'udp', address: string, port: number): string {
    return `${computerId}\u0000${transport}\u0000${address}\u0000${port}`;
  }

  /**
   * Real bind semantics. A wildcard bind conflicts with every address on the
   * port and vice versa; a concrete address only conflicts with itself. Extra
   * name-based virtual hosts on an address that this owner already holds are
   * allowed, exactly like one listening socket serving many `server_name`s.
   */
  private assertBindable(computerId: string, transport: 'tcp' | 'udp', address: string, port: number, host: string, owner: string): void {
    for (const binding of this.portBindings.values()) {
      if (binding.computerId !== computerId || binding.transport !== transport || binding.port !== port) continue;
      const wildcardCollision = binding.address === '0.0.0.0' || address === '0.0.0.0';
      if (!wildcardCollision && binding.address !== address) continue;
      if (binding.owner === owner && binding.host === host) continue;
      // One listening socket may serve several name-based virtual hosts, so a
      // distinct hostname on an address another *service* already holds is
      // legal. An ephemeral client port is never shareable that way.
      const nameBasedVirtualHost = binding.kind !== 'ephemeral' && binding.host !== host
        && serviceBinding(host) === 'network' && serviceBinding(binding.host) === 'network';
      if (!wildcardCollision && binding.address === address && nameBasedVirtualHost) continue;
      throw new Error(`EADDRINUSE: ${transport} bind ${address}:${port} on ${computerId} is already held by ${binding.owner}`);
    }
  }

  private bindPort(binding: PortBindingRecord): void {
    this.portBindings.set(`${this.bindingKey(binding.computerId, binding.transport, binding.address, binding.port)}\u0000${binding.host}`, binding);
  }

  private releasePorts(predicate: (binding: PortBindingRecord) => boolean): void {
    for (const [key, binding] of this.portBindings) if (predicate(binding)) this.portBindings.delete(key);
  }

  listPortBindings(computerId?: string): PortBindingRecord[] {
    return [...this.portBindings.values()].filter((binding) => !computerId || binding.computerId === computerId).map((binding) => ({ ...binding }));
  }

  /**
   * Real ephemeral allocation: a per-computer rotating cursor over the
   * configured range that skips every port currently in use, so a live socket
   * can never be shadowed. Throws EADDRNOTAVAIL when the range is exhausted.
   */
  allocateEphemeralPort(computerId: string, transport: 'tcp' | 'udp' = 'tcp', address = '0.0.0.0'): number {
    const [low, high] = this.options.ephemeralPortRange;
    const span = high - low + 1;
    const key = `${computerId}\u0000${transport}`;
    let cursor = this.ephemeralCursor.get(key) ?? 0;
    for (let attempt = 0; attempt < span; attempt += 1) {
      const port = low + ((cursor + attempt) % span);
      if (this.isPortFree(computerId, transport, address, port)) {
        this.ephemeralCursor.set(key, (cursor + attempt + 1) % span);
        this.bindPort({ computerId, transport, address, port, host: address, owner: `ephemeral:${port}`, kind: 'ephemeral' });
        return port;
      }
    }
    throw new Error(`EADDRNOTAVAIL: no ephemeral ${transport} port available on ${computerId}`);
  }

  private isPortFree(computerId: string, transport: 'tcp' | 'udp', address: string, port: number): boolean {
    for (const binding of this.portBindings.values()) {
      if (binding.computerId !== computerId || binding.transport !== transport || binding.port !== port) continue;
      if (binding.address === '0.0.0.0' || address === '0.0.0.0' || binding.address === address) return false;
    }
    return true;
  }

  releaseEphemeralPort(computerId: string, transport: 'tcp' | 'udp', address: string, port: number): void {
    this.releasePorts((binding) => binding.kind === 'ephemeral' && binding.computerId === computerId
      && binding.transport === transport && binding.address === address && binding.port === port);
  }

  /* ---------------------------------------------------------------------- *
   * Service registration
   * ---------------------------------------------------------------------- */

  registerService(service: VirtualService): void {
    const host = normalizeHost(service.host);
    const binding = serviceBinding(host);
    const ip = this.computers.get(service.computerId)?.ipv4;
    if (!ip) throw new Error(`cannot register service ${service.id} on unknown computer: ${service.computerId}`);
    const localAddress = binding === 'loopback' ? loopbackAddress(host) : binding === 'wildcard' ? '0.0.0.0' : ip;
    this.assertBindable(service.computerId, 'tcp', localAddress, service.port, host, service.id);
    const key = `${service.computerId}\u0000${host}\u0000${service.port}`;
    this.services.set(key, { ...service, host });
    this.bindPort({ computerId: service.computerId, transport: 'tcp', address: localAddress, port: service.port, host, owner: service.id, pid: service.pid, kind: 'service' });
    if (binding === 'network') this.addDns(host, ip);
    if (!this.socketRecords.some((socket) => socket.computerId === service.computerId && socket.protocol !== 'udp' && socket.localAddress === localAddress && socket.localPort === service.port && socket.state === 'LISTEN')) {
      this.socketRecords.push({
        id: randomUUID(), protocol: service.protocol, computerId: service.computerId,
        localAddress, localPort: service.port, state: 'LISTEN', rxBytes: 0, txBytes: 0, owner: service.id,
      });
    }
  }

  unregisterService(host: string, port: number, computerId?: string): void {
    const normalized = normalizeHost(host);
    const affectedComputers = new Set<string>();
    for (const [key, service] of this.services) {
      if (normalizeHost(service.host) !== normalized || service.port !== port || (computerId && service.computerId !== computerId)) continue;
      this.services.delete(key);
      this.releasePorts((binding) => binding.owner === service.id && binding.port === port && binding.computerId === service.computerId);
      affectedComputers.add(service.computerId);
    }
    for (const affectedComputer of affectedComputers) this.closeUnusedListeners(affectedComputer, port);
  }

  unregisterServicesForProcess(computerId: string, pid: number): string[] {
    const removed: string[] = [];
    const affectedPorts = new Set<number>();
    for (const [key, service] of this.services.entries()) {
      if (service.computerId !== computerId || service.pid !== pid) continue;
      this.services.delete(key);
      this.releasePorts((binding) => binding.owner === service.id && binding.computerId === computerId);
      removed.push(service.id);
      affectedPorts.add(service.port);
    }
    for (const [key, service] of this.datagramServices.entries()) {
      if (service.computerId !== computerId || service.pid !== pid) continue;
      this.datagramServices.delete(key);
      this.releasePorts((binding) => binding.owner === service.id && binding.computerId === computerId);
      this.closeListenerSocket(computerId, service.port, 'udp');
      removed.push(service.id);
    }
    for (const [key, service] of this.streamServices.entries()) {
      if (service.computerId !== computerId || service.pid !== pid) continue;
      this.streamServices.delete(key);
      this.releasePorts((binding) => binding.owner === service.id && binding.computerId === computerId);
      this.closeListenerSocket(computerId, service.port, 'tcp');
      removed.push(service.id);
    }
    for (const port of affectedPorts) this.closeUnusedListeners(computerId, port);
    return removed;
  }

  /** Binds a UDP listener. Enforces EADDRINUSE against the UDP port space. */
  bindDatagram(service: DatagramService): NetworkBindingHandle {
    const host = normalizeHost(service.host);
    const spec = this.computers.get(service.computerId);
    if (!spec) throw new Error(`cannot bind udp service ${service.id} on unknown computer: ${service.computerId}`);
    const binding = serviceBinding(host);
    const localAddress = binding === 'loopback' ? loopbackAddress(host) : binding === 'wildcard' ? '0.0.0.0' : spec.ipv4;
    this.assertBindable(service.computerId, 'udp', localAddress, service.port, host, service.id);
    const key = `${service.computerId}\u0000${host}\u0000${service.port}`;
    this.datagramServices.set(key, { ...service, host });
    this.bindPort({ computerId: service.computerId, transport: 'udp', address: localAddress, port: service.port, host, owner: service.id, pid: service.pid, kind: 'datagram' });
    if (binding === 'network') this.addDns(host, spec.ipv4);
    this.socketRecords.push({
      id: randomUUID(), protocol: 'udp', computerId: service.computerId, localAddress,
      localPort: service.port, state: 'LISTEN', rxBytes: 0, txBytes: 0, owner: service.id,
    });
    return {
      close: () => {
        this.datagramServices.delete(key);
        this.releasePorts((candidate) => candidate.owner === service.id && candidate.transport === 'udp');
        this.closeListenerSocket(service.computerId, service.port, 'udp');
      },
    };
  }

  /** Binds a raw TCP stream listener for non-HTTP services. */
  bindStream(service: StreamService): NetworkBindingHandle {
    const host = normalizeHost(service.host);
    const spec = this.computers.get(service.computerId);
    if (!spec) throw new Error(`cannot bind stream service ${service.id} on unknown computer: ${service.computerId}`);
    const binding = serviceBinding(host);
    const localAddress = binding === 'loopback' ? loopbackAddress(host) : binding === 'wildcard' ? '0.0.0.0' : spec.ipv4;
    this.assertBindable(service.computerId, 'tcp', localAddress, service.port, host, service.id);
    const key = `${service.computerId}\u0000${host}\u0000${service.port}`;
    this.streamServices.set(key, { ...service, host });
    this.bindPort({ computerId: service.computerId, transport: 'tcp', address: localAddress, port: service.port, host, owner: service.id, pid: service.pid, kind: 'stream' });
    if (binding === 'network') this.addDns(host, spec.ipv4);
    this.socketRecords.push({
      id: randomUUID(), protocol: 'tcp', computerId: service.computerId, localAddress,
      localPort: service.port, state: 'LISTEN', rxBytes: 0, txBytes: 0, owner: service.id,
    });
    return {
      close: () => {
        this.streamServices.delete(key);
        this.releasePorts((candidate) => candidate.owner === service.id && candidate.transport === 'tcp');
        this.closeListenerSocket(service.computerId, service.port, 'tcp');
      },
    };
  }

  private closeListenerSocket(computerId: string, port: number, protocol: 'tcp' | 'udp'): void {
    for (const socket of this.socketRecords) {
      if (socket.computerId !== computerId || socket.localPort !== port || socket.state !== 'LISTEN') continue;
      const matches = protocol === 'udp' ? socket.protocol === 'udp' : socket.protocol !== 'udp';
      if (matches) this.closeSocket(socket);
    }
  }

  /* ---------------------------------------------------------------------- *
   * Sockets
   * ---------------------------------------------------------------------- */

  private closeSocket(socket: InternalSocket): void {
    if (socket.state !== 'CLOSED') {
      socket.state = 'CLOSED';
      socket.closedAt = this.now();
      if (socket.ephemeral) {
        this.releaseEphemeralPort(socket.computerId, socket.protocol === 'udp' ? 'udp' : 'tcp', socket.localAddress, socket.localPort);
      }
    }
    this.pruneSockets();
  }

  /**
   * Retention policy: live sockets are never dropped; closed ones survive until
   * they age out or the newest `maxClosedSockets` entries push them out, so a
   * long-running simulation cannot grow without bound.
   */
  private pruneSockets(): void {
    const at = this.now();
    const closed: number[] = [];
    for (let index = 0; index < this.socketRecords.length; index += 1) {
      if (this.socketRecords[index]!.state === 'CLOSED') closed.push(index);
    }
    const expired = new Set<number>();
    for (const index of closed) {
      const socket = this.socketRecords[index]!;
      if (socket.closedAt !== undefined && at - socket.closedAt > this.options.closedSocketTtlMs) expired.add(index);
    }
    const survivors = closed.filter((index) => !expired.has(index));
    const overflow = survivors.length - this.options.maxClosedSockets;
    for (let index = 0; index < overflow; index += 1) expired.add(survivors[index]!);
    if (expired.size === 0) return;
    for (let index = this.socketRecords.length - 1; index >= 0; index -= 1) {
      if (expired.has(index)) this.socketRecords.splice(index, 1);
    }
  }

  listSockets(computerId?: string): SocketRecord[] {
    return this.socketRecords
      .filter((socket) => !computerId || socket.computerId === computerId)
      .map(({ closedAt: _closedAt, ephemeral: _ephemeral, owner: _owner, ...socket }) => ({ ...socket }));
  }

  /* ---------------------------------------------------------------------- *
   * Tracing and the link model
   * ---------------------------------------------------------------------- */

  private trace(
    protocol: Protocol, source: string, destination: string, summary: string, bytes = 0,
    sourcePort?: number, destinationPort?: number, flags?: string[], extra: Partial<NetworkPacketTrace> = {},
  ): NetworkPacketTrace {
    const packet: NetworkPacketTrace = {
      id: randomUUID(), at: new Date(this.now()).toISOString(), protocol, source, destination,
      summary, bytes, sourcePort, destinationPort, flags, ...extra,
    };
    this.traces.push(packet);
    if (this.traces.length > this.options.maxTraces) this.traces.shift();
    return packet;
  }

  /** Seeded jitter + serialisation delay. Defaults keep this exactly `latencyMs`. */
  private latencyFor(profile: LinkProfile, bytes: number): number {
    const jitter = profile.jitterMs > 0 ? this.random.next() * profile.jitterMs : 0;
    const serialisation = profile.bandwidthBps > 0 ? (bytes * 8 * 1000) / profile.bandwidthBps : 0;
    return profile.latencyMs + jitter + serialisation;
  }

  private drops(profile: LinkProfile): boolean {
    return profile.lossRate > 0 && this.random.next() < profile.lossRate;
  }

  /** Splits a payload across the path MTU when the link enables fragmentation. */
  private segmentsFor(profile: LinkProfile, bytes: number): number {
    if (!profile.fragment) return 1;
    const mss = Math.max(1, profile.mtu - 40);
    return Math.max(1, Math.ceil(bytes / mss));
  }

  listPackets(): NetworkPacketTrace[] { return this.traces.map((trace) => ({ ...trace })); }

  listNatBindings(): NatBindingRecord[] { return this.natBindings.map((binding) => ({ ...binding })); }

  /* ---------------------------------------------------------------------- *
   * ICMP
   * ---------------------------------------------------------------------- */

  ping(computerId: string, host: string, count = 1): string {
    const source = this.computers.get(computerId);
    if (!source) throw new Error(`unknown computer: ${computerId}`);
    const destination = this.resolve(host, computerId);
    if (!destination) return `ping: cannot resolve ${host}: unknown host`;
    const loopback = isLoopbackHost(host);
    const sourceAddress = loopback ? destination : source.ipv4;
    const peer = this.computerByAddress(destination);
    const profile = loopback ? this.linkProfile(computerId, computerId) : this.linkProfile(computerId, peer?.id ?? 'internet');
    if (!loopback) this.resolveNeighbor(computerId, destination);
    const lines: string[] = [`PING ${host} (${destination}): 56 data bytes`];
    let received = 0;
    for (let sequence = 0; sequence < Math.max(1, count); sequence += 1) {
      const dropped = this.drops(profile);
      const rtt = this.latencyFor(profile, 64);
      this.trace('icmp', sourceAddress, destination, `echo request ${sourceAddress} → ${destination}`, 64, undefined, undefined, undefined, { sequence, ttl: 64, latencyMs: rtt, dropped });
      if (dropped) { lines.push(`Request timeout for icmp_seq ${sequence}`); continue; }
      this.trace('icmp', destination, sourceAddress, `echo reply ${destination} → ${sourceAddress}`, 64, undefined, undefined, undefined, { sequence, ttl: 64, latencyMs: rtt });
      received += 1;
      lines.push(`64 bytes from ${destination}: icmp_seq=${sequence} ttl=64 time=${rtt.toFixed(2)} ms`);
    }
    const transmitted = Math.max(1, count);
    const loss = ((transmitted - received) / transmitted) * 100;
    lines.push(`--- ${host} ping statistics ---`);
    lines.push(`${transmitted} packets transmitted, ${received} received, ${loss.toFixed(1)}% packet loss`);
    return lines.join('\n');
  }

  /* ---------------------------------------------------------------------- *
   * UDP
   * ---------------------------------------------------------------------- */

  private findDatagramService(sourceComputerId: string, destinationComputerId: string, requestedHost: string, port: number): DatagramService | undefined {
    const candidates = [...this.datagramServices.values()].filter((service) => service.computerId === destinationComputerId && service.port === port);
    if (isLoopbackHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      const requestedAddress = loopbackAddress(requestedHost);
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'loopback' && loopbackAddress(candidate.host) === requestedAddress)
        ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    if (isWildcardHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    return candidates.find((candidate) => serviceBinding(candidate.host) !== 'loopback' && normalizeHost(candidate.host) === requestedHost)
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard')
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'network');
  }

  /**
   * Sends a UDP datagram inside the fabric. Loopback isolation matches the TCP
   * path: a loopback destination is only reachable from the same computer.
   */
  async sendDatagram(computerId: string, host: string, port: number, payload: string): Promise<DatagramResult> {
    const source = this.computers.get(computerId);
    if (!source) throw new Error(`unknown computer: ${computerId}`);
    const requestedHost = normalizeHost(host);
    const loopback = isLoopbackHost(requestedHost);
    const wildcard = isWildcardHost(requestedHost);
    const destination = wildcard ? source.ipv4 : this.resolve(requestedHost, computerId);
    if (!destination) throw new Error(`udp: cannot resolve ${host}`);
    const destinationComputer = loopback || wildcard ? source : this.computerByAddress(destination);
    const routedDestination = loopback ? loopbackAddress(requestedHost) : destination;
    const sourceAddress = loopback ? routedDestination : source.ipv4;
    const localPort = this.allocateEphemeralPort(computerId, 'udp', sourceAddress);
    const socket: InternalSocket = {
      id: randomUUID(), protocol: 'udp', computerId, localAddress: sourceAddress, localPort,
      remoteAddress: routedDestination, remotePort: port, state: 'ESTABLISHED', rxBytes: 0, txBytes: 0, ephemeral: true,
    };
    this.socketRecords.push(socket);
    const profile = loopback ? this.linkProfile(computerId, computerId) : this.linkProfile(computerId, destinationComputer?.id ?? 'internet');
    const dropped = this.drops(profile);
    const latencyMs = this.latencyFor(profile, payload.length);
    const requestTrace = this.trace('udp', sourceAddress, routedDestination, `datagram ${payload.length} bytes`, payload.length, localPort, port, undefined, { latencyMs, dropped, ttl: 64 });
    socket.txBytes += payload.length;
    if (dropped) { this.closeSocket(socket); return { delivered: false, dropped: true, traceId: requestTrace.id, latencyMs }; }
    const service = destinationComputer ? this.findDatagramService(computerId, destinationComputer.id, requestedHost, port) : undefined;
    if (!service) {
      this.trace('icmp', routedDestination, sourceAddress, `destination port unreachable ${port}`, 0, undefined, undefined, undefined, { ttl: 64 });
      this.closeSocket(socket);
      return { delivered: false, dropped: false, traceId: requestTrace.id, latencyMs };
    }
    const listener = this.socketRecords.find((candidate) => candidate.owner === service.id && candidate.state === 'LISTEN');
    if (listener) listener.rxBytes += payload.length;
    const message: DatagramMessage = {
      id: randomUUID(), computerId: service.computerId,
      source: { address: sourceAddress, port: localPort },
      destination: { address: routedDestination, port },
      payload,
    };
    const reply = await service.handle(message);
    if (reply !== undefined) {
      if (listener) listener.txBytes += reply.length;
      socket.rxBytes += reply.length;
      this.trace('udp', routedDestination, sourceAddress, `datagram ${reply.length} bytes`, reply.length, port, localPort, undefined, { latencyMs, ttl: 64 });
    }
    this.closeSocket(socket);
    return { delivered: true, dropped: false, reply, traceId: requestTrace.id, latencyMs };
  }

  /**
   * Binds a UDP DNS responder backed by this fabric's zone data, making
   * DNS-over-UDP expressible. Queries are `name` or `{"name":…,"type":…}`;
   * answers are JSON `DnsResolution` documents.
   */
  serveDns(computerId: string, options: { host?: string; port?: number; id?: string; pid?: number } = {}): NetworkBindingHandle {
    const spec = this.computers.get(computerId);
    if (!spec) throw new Error(`unknown computer: ${computerId}`);
    return this.bindDatagram({
      id: options.id ?? `seed-dnsd-${computerId}`,
      computerId,
      host: options.host ?? spec.ipv4,
      port: options.port ?? 53,
      pid: options.pid,
      handle: (message) => {
        let name = message.payload.trim();
        let family: AddressFamily | undefined;
        if (name.startsWith('{')) {
          try {
            const query = JSON.parse(name) as { name?: string; type?: string };
            name = query.name ?? '';
            if (query.type === 'AAAA') family = 6;
            if (query.type === 'A') family = 4;
          } catch { /* fall through to the plain-name form */ }
        }
        if (!name) return JSON.stringify({ name: '', status: 'nxdomain', source: 'none', chain: [], records: [], addresses: [] });
        return JSON.stringify(this.resolveDetailed(name, { computerId: message.computerId, family }));
      },
    });
  }

  /** Client half of DNS-over-UDP: queries the configured resolver address. */
  async resolveOverUdp(computerId: string, name: string, options: { server?: string; port?: number; family?: AddressFamily } = {}): Promise<DnsResolution> {
    const payload = options.family === undefined ? name : JSON.stringify({ name, type: options.family === 6 ? 'AAAA' : 'A' });
    const result = await this.sendDatagram(computerId, options.server ?? this.config.dns, options.port ?? 53, payload);
    if (!result.delivered || result.reply === undefined) {
      return { name: normalizeHost(name), status: 'nxdomain', source: 'none', chain: [], records: [], addresses: [] };
    }
    return JSON.parse(result.reply) as DnsResolution;
  }

  /* ---------------------------------------------------------------------- *
   * Raw TCP streams
   * ---------------------------------------------------------------------- */

  private findStreamService(sourceComputerId: string, destinationComputerId: string, requestedHost: string, port: number): StreamService | undefined {
    const candidates = [...this.streamServices.values()].filter((service) => service.computerId === destinationComputerId && service.port === port);
    if (isLoopbackHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      const requestedAddress = loopbackAddress(requestedHost);
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'loopback' && loopbackAddress(candidate.host) === requestedAddress)
        ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    if (isWildcardHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    return candidates.find((candidate) => serviceBinding(candidate.host) !== 'loopback' && normalizeHost(candidate.host) === requestedHost)
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard')
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'network');
  }

  /** Opens a socket-level TCP connection to a `bindStream` service. */
  async connect(computerId: string, host: string, port: number): Promise<StreamConnection> {
    const source = this.computers.get(computerId);
    if (!source) throw new Error(`unknown computer: ${computerId}`);
    const requestedHost = normalizeHost(host);
    const loopback = isLoopbackHost(requestedHost);
    const wildcard = isWildcardHost(requestedHost);
    const destination = wildcard ? source.ipv4 : this.resolve(requestedHost, computerId);
    if (!destination) throw new Error(`connect: cannot resolve ${host}`);
    const destinationComputer = loopback || wildcard ? source : this.computerByAddress(destination);
    const service = destinationComputer ? this.findStreamService(computerId, destinationComputer.id, requestedHost, port) : undefined;
    if (!service) throw new Error(`connection refused: ${host}:${port}`);
    const routedDestination = loopback ? loopbackAddress(requestedHost) : destination;
    const sourceAddress = loopback ? routedDestination : source.ipv4;
    const localPort = this.allocateEphemeralPort(computerId, 'tcp', sourceAddress);
    const socket: InternalSocket = {
      id: randomUUID(), protocol: 'tcp', computerId, localAddress: sourceAddress, localPort,
      remoteAddress: routedDestination, remotePort: port, state: 'SYN-SENT', rxBytes: 0, txBytes: 0, ephemeral: true,
    };
    this.socketRecords.push(socket);
    const profile = loopback ? this.linkProfile(computerId, computerId) : this.linkProfile(computerId, destinationComputer!.id);
    const flow = this.handshake(sourceAddress, routedDestination, localPort, port, profile);
    socket.state = 'ESTABLISHED';
    const info: StreamConnectionInfo = { id: socket.id, computerId, localAddress: sourceAddress, localPort, remoteAddress: routedDestination, remotePort: port };
    const listener = this.socketRecords.find((candidate) => candidate.owner === service.id && candidate.state === 'LISTEN');
    let greeting: string | undefined;
    try { greeting = await service.onConnect?.(info); }
    catch (error) { this.closeSocket(socket); throw error; }
    if (greeting !== undefined) {
      socket.rxBytes += greeting.length;
      if (listener) listener.txBytes += greeting.length;
      this.emitPayload('tcp', routedDestination, sourceAddress, port, localPort, `stream ${greeting.length} bytes`, greeting.length, flow, 'server', profile);
    }
    let closed = false;
    const connection: StreamConnection = {
      ...info,
      get closed() { return closed; },
      send: async (data: string) => {
        if (closed) throw new Error(`EPIPE: stream ${info.id} is closed`);
        socket.txBytes += data.length;
        if (listener) listener.rxBytes += data.length;
        this.emitPayload('tcp', sourceAddress, routedDestination, localPort, port, `stream ${data.length} bytes`, data.length, flow, 'client', profile);
        const reply = await service.handle(data, info);
        if (reply !== undefined) {
          socket.rxBytes += reply.length;
          if (listener) listener.txBytes += reply.length;
          this.emitPayload('tcp', routedDestination, sourceAddress, port, localPort, `stream ${reply.length} bytes`, reply.length, flow, 'server', profile);
        }
        return reply;
      },
      close: () => {
        if (closed) return;
        closed = true;
        this.trace('tcp', sourceAddress, routedDestination, 'FIN, ACK', 0, localPort, port, ['FIN', 'ACK'], { sequence: flow.clientSeq, ack: flow.serverSeq, window: flow.clientWindow });
        service.onClose?.(info);
        this.closeSocket(socket);
      },
    };
    return connection;
  }

  private handshake(sourceAddress: string, destinationAddress: string, sourcePort: number, destinationPort: number, profile: LinkProfile): TcpFlow {
    const flow = new TcpFlow(this.random.int(0xffffffff), this.random.int(0xffffffff));
    const mss = Math.max(536, profile.mtu - 40);
    const latencyMs = this.latencyFor(profile, 0);
    this.trace('tcp', sourceAddress, destinationAddress, 'SYN', 0, sourcePort, destinationPort, ['SYN'], { sequence: flow.clientSeq, window: flow.clientWindow, mss, latencyMs });
    this.trace('tcp', destinationAddress, sourceAddress, 'SYN, ACK', 0, destinationPort, sourcePort, ['SYN', 'ACK'], { sequence: flow.serverSeq, ack: (flow.clientSeq + 1) >>> 0, window: flow.serverWindow, mss, latencyMs });
    flow.clientSeq = (flow.clientSeq + 1) >>> 0;
    flow.serverSeq = (flow.serverSeq + 1) >>> 0;
    this.trace('tcp', sourceAddress, destinationAddress, 'ACK', 0, sourcePort, destinationPort, ['ACK'], { sequence: flow.clientSeq, ack: flow.serverSeq, window: flow.clientWindow, latencyMs });
    return flow;
  }

  /**
   * Emits the PSH/ACK rows for a payload. The first row keeps the caller's
   * summary verbatim (existing evidence matches on it); additional rows only
   * appear when the link enables fragmentation and the payload exceeds the MSS.
   */
  private emitPayload(
    protocol: Protocol, source: string, destination: string, sourcePort: number, destinationPort: number,
    summary: string, bytes: number, flow: TcpFlow, side: 'client' | 'server', profile: LinkProfile,
  ): NetworkPacketTrace {
    const segments = this.segmentsFor(profile, bytes);
    const mss = Math.max(536, profile.mtu - 40);
    const sequence = side === 'client' ? flow.clientSeq : flow.serverSeq;
    const ack = side === 'client' ? flow.serverSeq : flow.clientSeq;
    const window = side === 'client' ? flow.clientWindow : flow.serverWindow;
    let first: NetworkPacketTrace | undefined;
    for (let index = 0; index < segments; index += 1) {
      const offset = index * mss;
      const size = segments === 1 ? bytes : Math.min(mss, bytes - offset);
      const retransmit = this.drops(profile);
      const packet = this.trace(
        protocol, source, destination, index === 0 ? summary : `${summary} [segment ${index + 1}/${segments}]`,
        size, sourcePort, destinationPort, ['PSH', 'ACK'],
        {
          sequence: (sequence + offset) >>> 0, ack, window, mss, latencyMs: this.latencyFor(profile, size),
          ...(segments > 1 ? { fragment: { index, count: segments, offset, more: index < segments - 1 } } : {}),
          ...(retransmit ? { retransmit: true } : {}),
        },
      );
      if (retransmit) {
        // Loss on this link: the sender retransmits the same segment.
        this.trace(protocol, source, destination, `${summary} [retransmission]`, size, sourcePort, destinationPort, ['PSH', 'ACK'], {
          sequence: (sequence + offset) >>> 0, ack, window, mss, retransmit: true, latencyMs: this.latencyFor(profile, size),
        });
      }
      if (index === 0) first = packet;
    }
    if (side === 'client') flow.advanceClient(bytes); else flow.advanceServer(bytes);
    return first!;
  }

  /* ---------------------------------------------------------------------- *
   * HTTP
   * ---------------------------------------------------------------------- */

  async request(computerId: string, rawUrl: string, method = 'GET', body?: string, options: NetworkRequestOptions = {}): Promise<VirtualHttpResponse> {
    const source = this.computers.get(computerId);
    if (!source) throw new Error(`unknown computer: ${computerId}`);
    const url = new URL(rawUrl.includes('://') ? rawUrl : `http://${rawUrl}`);
    const port = Number(url.port || (url.protocol === 'https:' ? 443 : 80));
    const protocol = url.protocol === 'https:' ? 'https' : 'http';
    const requestedHost = normalizeHost(url.hostname);
    const loopbackDestination = isLoopbackHost(requestedHost);
    const wildcardDestination = isWildcardHost(requestedHost);
    const destination = this.resolve(requestedHost, computerId);
    const destinationComputer = loopbackDestination || wildcardDestination
      ? source
      : destination ? this.computerByAddress(destination) : undefined;
    const virtualDestination = Boolean(destinationComputer);
    if (!virtualDestination) return await this.egress(source, url, protocol, port, method, body, options);
    if (!destination || !destinationComputer) throw new Error(`virtual destination disappeared: ${url.hostname}`);
    const service = this.findVirtualService(computerId, destinationComputer.id, requestedHost, port);
    if (!service) throw new Error(`connection refused: ${url.hostname}:${port}`);
    const routedDestination = loopbackDestination ? loopbackAddress(requestedHost) : destination;
    const sourceAddress = loopbackDestination ? routedDestination : source.ipv4;
    if (!loopbackDestination) this.resolveNeighbor(computerId, routedDestination);
    const profile = loopbackDestination ? this.linkProfile(computerId, computerId) : this.linkProfile(computerId, destinationComputer.id);
    const localPort = this.allocateEphemeralPort(computerId, 'tcp', sourceAddress);
    const clientSocket: InternalSocket = {
      id: randomUUID(), protocol: 'tcp', computerId, localAddress: sourceAddress, localPort,
      remoteAddress: routedDestination, remotePort: port, state: 'SYN-SENT', rxBytes: 0, txBytes: 0, ephemeral: true,
    };
    this.socketRecords.push(clientSocket);
    const flow = this.handshake(sourceAddress, routedDestination, localPort, port, profile);
    clientSocket.state = 'ESTABLISHED';
    const requestBytes = body?.length ?? 0;
    const trace = this.emitPayload(protocol, sourceAddress, routedDestination, localPort, port, `${method} ${url.pathname}`, requestBytes, flow, 'client', profile);
    let response: Awaited<ReturnType<VirtualService['handle']>>;
    try { response = await service.handle(`${url.pathname}${url.search}`, method, body); }
    catch (error) {
      // A failing handler must still release the connection's ephemeral port.
      this.trace('tcp', routedDestination, sourceAddress, 'RST, ACK', 0, port, localPort, ['RST', 'ACK'], { sequence: flow.serverSeq, ack: flow.clientSeq });
      this.closeSocket(clientSocket);
      throw error;
    }
    const listenerAddress = this.listenerAddressForService(service);
    const listener = this.socketRecords.find((socket) => socket.computerId === service.computerId && socket.localAddress === listenerAddress && socket.localPort === port && socket.state === 'LISTEN');
    if (listener) { listener.rxBytes += requestBytes; listener.txBytes += response.body.length; }
    clientSocket.txBytes += requestBytes;
    clientSocket.rxBytes += response.body.length;
    this.emitPayload(protocol, routedDestination, sourceAddress, port, localPort, `${response.status} response`, response.body.length, flow, 'server', profile);
    this.trace('tcp', sourceAddress, routedDestination, 'FIN, ACK', 0, localPort, port, ['FIN', 'ACK'], { sequence: flow.clientSeq, ack: flow.serverSeq, window: flow.clientWindow });
    this.closeSocket(clientSocket);
    return { ...response, traceId: trace.id };
  }

  /**
   * Resolves a non-virtual hostname and returns the gateway rule plus the one
   * address the connection is allowed to use. Every answer must satisfy the
   * rule (unchanged policy); the returned address is then pinned so the
   * connection cannot be redirected to a different answer afterwards.
   */
  private async validateEgress(protocol: 'http' | 'https', hostname: string, port: number): Promise<{ rule: GatewayRule; address: string; answers: string[] }> {
    const normalized = normalizeHost(hostname);
    const fabricAnswer = this.resolve(normalized);
    const answers = fabricAnswer ? [fabricAnswer] : await this.externalResolver(normalized);
    const rule = this.canEgress(protocol, hostname, port, answers);
    if (!rule) throw new Error(`gateway denied: ${hostname}:${port}`);
    // Pick the first answer the rule actually covers. When the rule has no
    // CIDRs `canEgress` already accepted the hostname, so any answer is fine.
    const address = rule.cidrs.length === 0
      ? answers[0]
      : answers.find((candidate) => rule.cidrs.some((cidr) => cidrContains(cidr, candidate)));
    if (!address) throw new Error(`gateway denied: ${hostname}:${port} (no validated address)`);
    return { rule, address, answers };
  }

  private async egress(
    source: ComputerSpec, url: URL, protocol: 'http' | 'https', port: number,
    method: string, body: string | undefined, options: NetworkRequestOptions,
  ): Promise<VirtualHttpResponse> {
    const maxRedirects = options.maxRedirects ?? 0;
    let current = url;
    let currentProtocol = protocol;
    let currentPort = port;
    let firstTraceId: string | undefined;
    for (let hop = 0; ; hop += 1) {
      const { address } = await this.validateEgress(currentProtocol, current.hostname, currentPort);
      const localPort = this.allocateEphemeralPort(source.id, 'tcp', source.ipv4);
      const nat = this.translate(source, localPort, currentProtocol, address, currentPort);
      const socket: InternalSocket = {
        id: randomUUID(), protocol: 'tcp', computerId: source.id, localAddress: source.ipv4, localPort,
        remoteAddress: address, remotePort: currentPort, state: 'SYN-SENT', rxBytes: 0, txBytes: 0, ephemeral: true,
      };
      this.socketRecords.push(socket);
      this.resolveNeighbor(source.id, this.config.gateway);
      const profile = this.linkProfile(source.id, 'internet');
      const label = `gateway:${current.hostname}`;
      const flow = new TcpFlow(this.random.int(0xffffffff), this.random.int(0xffffffff));
      const natExtra = { translatedSource: nat.outsideAddress, translatedSourcePort: nat.outsidePort };
      this.trace('tcp', source.ipv4, label, 'SYN', 0, localPort, currentPort, ['SYN'], { sequence: flow.clientSeq, window: flow.clientWindow, ...natExtra });
      this.trace('tcp', label, source.ipv4, 'SYN, ACK', 0, currentPort, localPort, ['SYN', 'ACK'], { sequence: flow.serverSeq, ack: (flow.clientSeq + 1) >>> 0, window: flow.serverWindow });
      flow.clientSeq = (flow.clientSeq + 1) >>> 0;
      flow.serverSeq = (flow.serverSeq + 1) >>> 0;
      this.trace('tcp', source.ipv4, label, 'ACK', 0, localPort, currentPort, ['ACK'], { sequence: flow.clientSeq, ack: flow.serverSeq, window: flow.clientWindow, ...natExtra });
      socket.state = 'ESTABLISHED';
      const requestBytes = body?.length ?? 0;
      const requestTrace = this.emitPayload(currentProtocol, source.ipv4, label, localPort, currentPort, `${method} ${current.pathname}`, requestBytes, flow, 'client', profile);
      firstTraceId ??= requestTrace.id;
      let response: PinnedEgressResponse;
      try {
        response = await this.egressTransport({
          address,
          hostname: current.hostname,
          port: currentPort,
          protocol: currentProtocol,
          method,
          path: `${current.pathname}${current.search}`,
          body,
          headers: { 'user-agent': 'seed-gateway/1.0', ...options.headers },
          timeoutMs: options.timeoutMs ?? 10_000,
        });
      } catch (error) {
        this.trace('tcp', label, source.ipv4, 'RST, ACK', 0, currentPort, localPort, ['RST', 'ACK'], { sequence: flow.serverSeq, ack: flow.clientSeq });
        this.closeSocket(socket);
        throw error;
      }
      socket.txBytes += requestBytes;
      socket.rxBytes += response.body.length;
      this.emitPayload(currentProtocol, label, source.ipv4, currentPort, localPort, `${response.status} ${response.statusText}`, response.body.length, flow, 'server', profile);
      this.trace('tcp', source.ipv4, label, 'FIN, ACK', 0, localPort, currentPort, ['FIN', 'ACK'], { sequence: flow.clientSeq, ack: flow.serverSeq, window: flow.clientWindow, ...natExtra });
      this.closeSocket(socket);

      const location = response.headers.location;
      const redirecting = response.status >= 300 && response.status < 400 && location !== undefined;
      if (!redirecting || hop >= maxRedirects) {
        return { status: response.status, headers: response.headers, body: response.body, traceId: firstTraceId };
      }
      // A redirect is a brand-new destination: re-resolve and re-authorise it
      // through the same gateway policy before any follow-up connection.
      const next = new URL(location, current);
      if (next.protocol !== 'http:' && next.protocol !== 'https:') throw new Error(`gateway denied: unsupported redirect scheme ${next.protocol}`);
      currentProtocol = next.protocol === 'https:' ? 'https' : 'http';
      currentPort = Number(next.port || (currentProtocol === 'https' ? 443 : 80));
      current = next;
      body = undefined;
      method = response.status === 307 || response.status === 308 ? method : 'GET';
    }
  }

  /** Source NAT for outbound flows. Traces keep the inside address plus the map. */
  private translate(source: ComputerSpec, insidePort: number, protocol: Protocol, destination: string, destinationPort: number): NatBindingRecord {
    const existing = this.natBindings.find((binding) => binding.computerId === source.id && binding.insidePort === insidePort && binding.protocol === protocol);
    if (existing) return existing;
    const outsidePort = 32768 + (this.natBindings.length % 28000);
    const binding: NatBindingRecord = {
      id: randomUUID(), computerId: source.id, protocol, insideAddress: source.ipv4, insidePort,
      outsideAddress: this.config.publicAddress, outsidePort, destination, destinationPort,
      createdAt: new Date(this.now()).toISOString(),
    };
    this.natBindings.push(binding);
    if (this.natBindings.length > 512) this.natBindings.shift();
    return binding;
  }

  /* ---------------------------------------------------------------------- *
   * Gateway policy
   * ---------------------------------------------------------------------- */

  addGateway(rule: GatewayRule): void { this.gateways.push(structuredClone(rule)); }

  setGatewayEnabled(id: string, enabled: boolean): GatewayRule {
    const rule = this.gateways.find((candidate) => candidate.id === id);
    if (!rule) throw new Error(`unknown gateway rule: ${id}`);
    rule.enabled = enabled;
    return structuredClone(rule);
  }

  private findVirtualService(sourceComputerId: string, destinationComputerId: string, requestedHost: string, port: number): VirtualService | undefined {
    const candidates = [...this.services.values()].filter((service) => service.computerId === destinationComputerId && service.port === port);
    if (isLoopbackHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      const requestedAddress = loopbackAddress(requestedHost);
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'loopback' && loopbackAddress(candidate.host) === requestedAddress)
        ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    if (isWildcardHost(requestedHost)) {
      if (sourceComputerId !== destinationComputerId) return undefined;
      return candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard');
    }
    // Exact virtual hosts take precedence. The interface fallback preserves
    // direct-IP/computer-hostname access, but loopback listeners are
    // deliberately excluded so a remote peer cannot reach them via the NIC.
    return candidates.find((candidate) => serviceBinding(candidate.host) !== 'loopback' && normalizeHost(candidate.host) === requestedHost)
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'wildcard')
      ?? candidates.find((candidate) => serviceBinding(candidate.host) === 'network');
  }

  private closeUnusedListeners(computerId: string, port: number): void {
    const bindings = new Set([...this.portBindings.values()]
      .filter((binding) => binding.computerId === computerId && binding.port === port && binding.transport === 'tcp' && binding.kind !== 'ephemeral')
      .map((binding) => binding.address));
    for (const socket of this.socketRecords) {
      if (socket.computerId === computerId && socket.localPort === port && socket.state === 'LISTEN' && socket.protocol !== 'udp' && !bindings.has(socket.localAddress)) this.closeSocket(socket);
    }
  }

  private listenerAddressForService(service: VirtualService): string {
    const binding = serviceBinding(service.host);
    if (binding === 'loopback') return loopbackAddress(service.host);
    if (binding === 'wildcard') return '0.0.0.0';
    return this.computers.get(service.computerId)?.ipv4 ?? '0.0.0.0';
  }

  canEgress(protocol: Protocol, hostname: string, port: number, resolvedAddresses: string[] = []): GatewayRule | undefined {
    return this.gateways.find((rule) => {
      if (!rule.enabled || rule.direction !== 'egress' || !rule.protocols.includes(protocol)) return false;
      if (rule.ports !== '*' && !rule.ports.includes(port)) return false;
      const hostnameAllowed = rule.hostnames.length === 0 || rule.hostnames.some((pattern) => hostnameMatches(pattern, hostname));
      if (!hostnameAllowed) return false;
      if (rule.cidrs.length === 0) return rule.hostnames.length > 0;
      if (resolvedAddresses.length === 0) return false;
      // Every current DNS answer must remain inside the rule. This blocks a
      // mixed-answer hostname from bypassing a CIDR constraint.
      return resolvedAddresses.every((address) => rule.cidrs.some((cidr) => cidrContains(cidr, address)));
    });
  }

  listServices(): Array<Omit<VirtualService, 'handle'>> { return [...this.services.values()].map(({ handle: _, ...service }) => service); }

  listDatagramServices(): Array<Omit<DatagramService, 'handle'>> { return [...this.datagramServices.values()].map(({ handle: _, ...service }) => service); }

  listStreamServices(): Array<Pick<StreamService, 'id' | 'computerId' | 'host' | 'port' | 'pid'>> {
    return [...this.streamServices.values()].map(({ id, computerId, host, port, pid }) => ({ id, computerId, host, port, pid }));
  }
}
