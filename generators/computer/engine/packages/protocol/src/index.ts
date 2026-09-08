export type OSKind = 'macos' | 'windows' | 'ubuntu';
export type ShellKind = 'zsh' | 'powershell' | 'bash';
export type ProcessState = 'running' | 'sleeping' | 'stopped' | 'zombie';
export type Protocol = 'tcp' | 'udp' | 'icmp' | 'http' | 'https';
export type PackageManagerKind = 'brew' | 'mas' | 'apt' | 'dpkg' | 'snap' | 'flatpak' | 'winget' | 'choco' | 'scoop' | 'npm' | 'pnpm' | 'yarn' | 'bun' | 'pip' | 'pipx' | 'poetry' | 'uv' | 'cargo' | 'go' | 'gem' | 'composer' | 'dotnet' | 'nuget' | 'vcpkg' | 'conda';
export type CollaborationServiceId = 'slack' | 'teams';
export type AppServiceKind = 'app-registry' | 'collaboration' | 'git' | 'mail' | 'calendar' | 'identity' | 'cloud-data' | 'media-catalog' | 'model-api' | 'virtual-network-client';

export interface DiskSpec {
  id: string;
  label: string;
  mount: string;
  capacityBytes: number;
}

export interface DisplaySpec {
  id: string;
  name: string;
  width: number;
  height: number;
  scale: number;
}

export interface ComputerSpec {
  id: string;
  hostname: string;
  os: OSKind;
  shell: ShellKind;
  ipv4: string;
  memoryBytes: number;
  cpuCores: number;
  disks: DiskSpec[];
  displays: DisplaySpec[];
}

export interface ProcessRecord {
  pid: number;
  ppid: number;
  computerId: string;
  executable: string;
  argv: string[];
  cwd: string;
  env: Record<string, string>;
  state: ProcessState;
  startedAt: string;
  cpuTimeMs: number;
  memoryBytes: number;
  listeningPorts: number[];
}

export interface InodeRecord {
  id: string;
  diskId: string;
  kind: 'file' | 'directory' | 'symlink';
  mode: number;
  size: number;
  createdAt: string;
  modifiedAt: string;
  target?: string;
}

export interface DirectoryEntry {
  name: string;
  path: string;
  inode: InodeRecord;
}

/** Inode as the virtual filesystem stores it: POSIX ownership plus a hard-link count. */
export interface VfsInodeRecord extends InodeRecord {
  uid: number;
  gid: number;
  links: number;
}

/** Caller identity used for permission enforcement. `uid: 0` is root and bypasses every check. */
export interface VfsIdentity {
  uid: number;
  gid: number;
}

/** Derived, fully expanded view of a filesystem: every path, and every inode it points at. */
export interface VfsFileTable {
  version: 1;
  paths: Record<string, string>;
  inodes: Record<string, InodeRecord>;
}

export interface DnsRecord {
  name: string;
  type: 'A' | 'CNAME';
  value: string;
  ttl: number;
}

export interface SocketRecord {
  id: string;
  protocol: Protocol;
  computerId: string;
  localAddress: string;
  localPort: number;
  remoteAddress?: string;
  remotePort?: number;
  state: 'LISTEN' | 'SYN-SENT' | 'ESTABLISHED' | 'CLOSED';
  rxBytes: number;
  txBytes: number;
}

export interface PacketTrace {
  id: string;
  at: string;
  protocol: Protocol;
  source: string;
  destination: string;
  sourcePort?: number;
  destinationPort?: number;
  flags?: string[];
  bytes: number;
  summary: string;
}

export interface GatewayRule {
  id: string;
  name: string;
  enabled: boolean;
  direction: 'egress' | 'ingress';
  protocols: Protocol[];
  cidrs: string[];
  hostnames: string[];
  ports: number[] | '*';
  audit: boolean;
}

/**
 * Fully serializable input consumed by the simulation kernel. Ecosystem
 * packages own policy and composition; the kernel only owns execution.
 */
export interface SimulationComputerTemplate {
  spec: ComputerSpec;
  systemAppIds: readonly string[];
  thirdPartyAppIds: readonly string[];
  roles: readonly string[];
}

export type SimulationServiceKind = AppServiceKind | 'dns' | 'intranet';

export interface SimulationServiceSpec {
  id: string;
  host: string;
  ipv4: string;
  computerId: string;
  port: number;
  protocol: Protocol;
  kind: SimulationServiceKind;
  /** Required for app-registry services so installation resolves by OS. */
  targetOS?: OSKind;
  /** State with different isolation domains must never share a backing store. */
  isolationDomain: string;
}

export interface SimulationTopology {
  id: string;
  version: string;
  network: {
    cidr: string;
    dns: string;
    domain: string;
  };
  computers: readonly SimulationComputerTemplate[];
  services: readonly SimulationServiceSpec[];
  gateways: readonly GatewayRule[];
  /**
   * Serialized per-OS runtime behavior. The kernel reads this generically so it
   * can honor platform conventions without importing a concrete OS package; a
   * full `OperatingSystemProfile` structurally satisfies it.
   */
  operatingSystems?: Readonly<Record<OSKind, OSRuntimeProfile>>;
}

/**
 * The subset of an operating-system profile the kernel actually enforces.
 * Anything declared here MUST have a runtime consumer — fields that only
 * document intent belong in the OS package, not in this contract.
 */
export interface OSRuntimeProfile {
  shell: {
    default: ShellKind;
    executable: string;
    promptDialect: 'posix' | 'powershell';
    startupFiles: readonly string[];
  };
  filesystem: {
    root: string;
    home: string;
    applications: string;
    userData: string;
    temporary: string;
    caseSensitive: boolean;
    pathSeparator: '/' | '\\';
  };
  packageManagers: {
    native: readonly PackageManagerKind[];
    language: readonly PackageManagerKind[];
    receiptRoots: readonly string[];
  };
  bootServices: readonly {
    id: string;
    executable: string;
    role: string;
    parent: string | null;
    required: boolean;
  }[];
  conventions: {
    executableSuffix: string;
    sharedLibrarySuffix: string;
    environmentPathKey: string;
    localhostNames: readonly string[];
  };
}

export interface AppManifest {
  id: string;
  name: string;
  version: string;
  publisher: string;
  description: string;
  icon: string;
  supportedOS: OSKind[];
  entrypoint: string;
  packagePath: string;
  system?: boolean;
  defaultSize?: { width: number; height: number };
  fileAssociations?: string[];
  capabilities: Array<'filesystem' | 'network' | 'notifications' | 'microphone' | 'camera'>;
  /** User-observable operations exposed by this particular application surface. */
  operations: string[];
  /** Explicit backend dependencies. An empty array means the app is local-only. */
  serviceContracts: AppServiceContract[];
  runtime: AppRuntimeDescriptor;
}

export interface AppRuntimeDescriptor {
  kind: 'seed-js' | 'seed-wasm' | 'system-component';
  apiVersion: 1;
  entryFile: string;
  stateSchema: string;
}

export interface InstalledApp extends AppManifest {
  installedAt: string;
  installPath: string;
  dataPath: string;
  receiptPath: string;
  registryHost: string;
  installState: 'installed' | 'updating';
}

export interface AppServiceContract {
  id: string;
  kind: AppServiceKind;
  host: string;
  protocol: 'http' | 'https' | 'virtual';
  port: number;
  auth: 'none' | 'session' | 'oauth' | 'device';
  required: boolean;
  operations: string[];
}

export interface AppLaunchRequest {
  operation: string;
  payload?: Record<string, unknown>;
}

export interface AppExecutionRecord {
  id: string;
  computerId: string;
  appId: string;
  runtime: AppRuntimeDescriptor['kind'];
  operation: string;
  startedAt: string;
  completedAt: string;
  status: 'completed' | 'failed';
  result?: unknown;
  error?: string;
}

export interface HostExecutionRule {
  id: string;
  enabled: boolean;
  computerIds: string[] | '*';
  appIds: string[];
  executables: string[];
  cwdRoots: string[];
  timeoutMs: number;
  maxOutputBytes: number;
  audit: boolean;
}

export interface HostExecutionResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

export interface PackageRecord {
  id: string;
  name: string;
  version: string;
  manager: PackageManagerKind;
  scope: 'system' | 'user' | 'project';
  installPath: string;
  installedAt: string;
  files: string[];
  source: string;
  integrity: string;
  dependencies: string[];
  dependencyType: 'direct' | 'transitive';
}

export interface PackageTransactionRecord {
  id: string;
  manager: PackageManagerKind;
  operation: 'index-refresh' | 'install' | 'remove' | 'upgrade';
  packages: string[];
  startedAt: string;
  completedAt: string;
  status: 'committed' | 'rolled-back';
  receiptPaths: string[];
}

export interface GitCommitRecord {
  hash: string;
  message: string;
  author: string;
  at: string;
  /** Root tree object id of the commit. */
  treeDigest: string;
  /** Parent commit ids, oldest link first. Absent on records received from a metadata-only remote. */
  parents?: string[];
  committer?: string;
  committedAt?: string;
}

export interface GitRepositoryRecord {
  root: string;
  branch: string;
  head?: string;
  branches: Record<string, string | undefined>;
  remotes: Record<string, string>;
  remoteRefs: Record<string, string>;
  staged: string[];
  commits: GitCommitRecord[];
  /** Tag name → object id (annotated tags point at the tag object). */
  tags?: Record<string, string>;
  detached?: boolean;
}

/**
 * A loose Git object in transit. `data` is the serialized loose-object text
 * (`<type> <length>\0<payload>`), which keeps the whole record JSON-safe as it
 * crosses the virtual HTTPS fabric.
 */
export interface GitObjectPayload {
  hash: string;
  data: string;
}

export interface CollaborationMessage {
  id: string;
  serviceId: CollaborationServiceId;
  workspaceId: string;
  channelId: string;
  sequence: number;
  author: string;
  computerId: string;
  text: string;
  at: string;
  editedAt?: string;
  threadId?: string;
}

export interface CollaborationChannel {
  id: string;
  name: string;
  displayName: string;
  memberCount: number;
}

export interface CollaborationServiceSnapshot {
  id: CollaborationServiceId;
  productName: 'Slack' | 'Microsoft Teams';
  host: string;
  workspaceId: string;
  workspaceName: string;
  revision: number;
  channels: CollaborationChannel[];
  messages: CollaborationMessage[];
}

export interface CollaborationPollResult {
  serviceId: CollaborationServiceId;
  workspaceId: string;
  channelId: string;
  revision: number;
  messages: CollaborationMessage[];
}

export interface VirtualHttpResponse {
  status: number;
  headers: Record<string, string>;
  body: string;
  traceId: string;
}

/**
 * Metadata returned when a virtual HTTP response is promoted to a real browser
 * document. The body is intentionally delivered by `documentUrl`, so Chromium
 * parses the response with its native HTML/JavaScript engine instead of the UI
 * pretending to interpret it.
 */
export interface BrowserNavigationResponse {
  url: string;
  documentUrl: string;
  status: number;
  headers: Record<string, string>;
  traceId: string;
}

/* ------------------------------------------------------------------------ *
 * Virtual network fabric.
 *
 * These types are additive: `DnsRecord`, `PacketTrace`, `SocketRecord` and
 * `Protocol` keep their original shape so every existing consumer (snapshots,
 * the Wireshark surface, netstat) is unaffected. The richer records below are
 * what `@tcn-computer/kernel`'s InternetFabric produces internally and exposes through
 * its extended listing methods.
 * ------------------------------------------------------------------------ */

export type AddressFamily = 4 | 6;

/** Record types the fabric's zone/cache can hold. `DnsRecord` stays A|CNAME. */
export type DnsRecordType = 'A' | 'AAAA' | 'CNAME' | 'PTR' | 'TXT' | 'SRV';

export interface DnsResourceRecord {
  name: string;
  type: DnsRecordType;
  value: string;
  ttl: number;
  /**
   * Authoritative records are served by a zone the fabric owns; their TTL is
   * published to clients but never causes local expiry. Cached (non
   * authoritative) records expire at `expiresAt`.
   */
  authoritative: boolean;
  zone?: string;
  expiresAt?: number;
}

/** A cached proof that a name does not exist (RFC 2308 negative caching). */
export interface DnsNegativeRecord {
  name: string;
  type: DnsRecordType | 'ANY';
  ttl: number;
  expiresAt: number;
  reason: 'nxdomain' | 'nodata';
}

export type DnsResolutionStatus = 'ok' | 'nxdomain' | 'nodata' | 'loop';
export type DnsResolutionSource = 'literal' | 'hosts' | 'authoritative' | 'cache' | 'negative-cache' | 'none';

export interface DnsResolution {
  name: string;
  status: DnsResolutionStatus;
  source: DnsResolutionSource;
  /** CNAME chain walked while answering, starting at the queried name. */
  chain: string[];
  records: DnsResourceRecord[];
  addresses: string[];
  ttl?: number;
}

/** Extra layer-3/4 detail carried by fabric traces. Superset of `PacketTrace`. */
export interface NetworkPacketTrace extends PacketTrace {
  sequence?: number;
  ack?: number;
  window?: number;
  mss?: number;
  ttl?: number;
  retransmit?: boolean;
  dropped?: boolean;
  latencyMs?: number;
  linkId?: string;
  /** Present when the datagram was split to fit the path MTU. */
  fragment?: { index: number; count: number; offset: number; more: boolean };
  /** Present when a NAT binding rewrote the source of an egress packet. */
  translatedSource?: string;
  translatedSourcePort?: number;
}

export interface NetworkInterfaceAddress {
  address: string;
  prefix: number;
  family: AddressFamily;
  scope: 'host' | 'link' | 'global';
}

export interface NetworkInterfaceRecord {
  id: string;
  computerId: string;
  name: string;
  mac: string;
  mtu: number;
  up: boolean;
  loopback: boolean;
  addresses: NetworkInterfaceAddress[];
}

export interface RouteRecord {
  id: string;
  computerId: string;
  /** CIDR, e.g. `0.0.0.0/0` for the default route. */
  destination: string;
  via?: string;
  dev: string;
  metric: number;
  family: AddressFamily;
  source: 'kernel' | 'static' | 'dhcp';
}

export interface NeighborRecord {
  computerId: string;
  address: string;
  mac: string;
  dev: string;
  family: AddressFamily;
  state: 'INCOMPLETE' | 'REACHABLE' | 'STALE' | 'PERMANENT' | 'FAILED';
  resolvedBy: 'arp' | 'ndp' | 'static';
}

export interface NatBindingRecord {
  id: string;
  computerId: string;
  protocol: Protocol;
  insideAddress: string;
  insidePort: number;
  outsideAddress: string;
  outsidePort: number;
  destination: string;
  destinationPort: number;
  createdAt: string;
}

/** Per-link transport characteristics. Defaults reproduce the legacy values. */
export interface LinkProfile {
  /** Base round-trip time in milliseconds. */
  latencyMs: number;
  /** Maximum extra RTT added by the seeded jitter source. */
  jitterMs: number;
  /** 0..1 probability that a packet is dropped, drawn from the seeded PRNG. */
  lossRate: number;
  /** Bits per second; `0` means unmetered (no serialisation delay). */
  bandwidthBps: number;
  mtu: number;
  /** When false, oversized payloads are traced as one row instead of fragments. */
  fragment: boolean;
}

export interface LinkRule {
  id: string;
  from: string;
  to: string;
  profile: LinkProfile;
}

export interface FabricNetworkConfig {
  cidr: string;
  gateway: string;
  dns: string;
  domain: string;
  /** Address the NAT translates outbound flows onto. */
  publicAddress: string;
  /** ULA prefix used to derive a deterministic IPv6 address per computer. */
  ipv6Prefix: string;
}

export interface DatagramMessage {
  id: string;
  computerId: string;
  source: { address: string; port: number };
  destination: { address: string; port: number };
  payload: string;
}

export interface DatagramResult {
  delivered: boolean;
  dropped: boolean;
  reply?: string;
  traceId: string;
  latencyMs: number;
}

export interface StreamConnectionInfo {
  id: string;
  computerId: string;
  localAddress: string;
  localPort: number;
  remoteAddress: string;
  remotePort: number;
}

export interface PortBindingRecord {
  computerId: string;
  transport: 'tcp' | 'udp';
  address: string;
  port: number;
  host: string;
  owner: string;
  pid?: number;
  kind: 'service' | 'stream' | 'datagram' | 'ephemeral';
}

export interface TrajectoryEvent {
  sequence: number;
  at: string;
  runId: string;
  computerId?: string;
  displayId?: string;
  actor: 'human' | 'agent' | 'system';
  kind: 'pointer' | 'keyboard' | 'window' | 'process' | 'filesystem' | 'network' | 'app' | 'snapshot';
  action: string;
  target?: string;
  data?: Record<string, unknown>;
  stateHash?: string;
}

export interface ComputerSnapshot {
  spec: ComputerSpec;
  bootedAt: string;
  uptimeMs: number;
  processes: ProcessRecord[];
  sockets: SocketRecord[];
  installedApps: InstalledApp[];
  packages: PackageRecord[];
  packageTransactions: PackageTransactionRecord[];
  repositories: GitRepositoryRecord[];
}

export interface SimulationSnapshot {
  runId: string;
  topology: { id: string; version: string };
  now: string;
  computers: ComputerSnapshot[];
  dns: DnsRecord[];
  packets: PacketTrace[];
  gateways: GatewayRule[];
  appCatalog: AppManifest[];
  collaborationServices: CollaborationServiceSnapshot[];
  appExecutions: AppExecutionRecord[];
  hostExecutionRules: HostExecutionRule[];
  trajectoryLength: number;
}
