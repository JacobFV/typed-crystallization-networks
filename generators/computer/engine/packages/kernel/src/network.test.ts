import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { describe, expect, it } from 'vitest';
import type { ComputerSpec, GatewayRule, NetworkPacketTrace } from '@tcn-computer/protocol';
import {
  addressFamily, cidrContains, createPinnedLookup, formatIpv6, InternetFabric, ipv6Number,
  normalizeAddress, SeededRandom,
  type EgressTransport, type InternetFabricOptions, type PinnedEgressRequest, type PinnedEgressResponse,
} from './network.js';

function spec(id: string, ipv4: string, os: ComputerSpec['os'] = 'ubuntu'): ComputerSpec {
  return {
    id, hostname: id, os, shell: os === 'windows' ? 'powershell' : 'bash', ipv4,
    memoryBytes: 1024, cpuCores: 2, disks: [], displays: [],
  };
}

function gateway(overrides: Partial<GatewayRule> = {}): GatewayRule {
  return {
    id: 'egress', name: 'egress', enabled: true, direction: 'egress',
    protocols: ['http', 'https'], cidrs: [], hostnames: ['*'], ports: '*', audit: true,
    ...overrides,
  };
}

function okTransport(body = 'ok'): { transport: EgressTransport; seen: PinnedEgressRequest[] } {
  const seen: PinnedEgressRequest[] = [];
  const transport: EgressTransport = async (request) => {
    seen.push(request);
    return { status: 200, statusText: 'OK', headers: { 'content-type': 'text/plain' }, body };
  };
  return { transport, seen };
}

function fabricWith(options: InternetFabricOptions = {}): InternetFabric {
  const fabric = new InternetFabric(options);
  fabric.attach(spec('alpha', '10.42.0.10'));
  fabric.attach(spec('beta', '10.42.0.20'));
  return fabric;
}

/* -------------------------------------------------------------------------- */

describe('address arithmetic', () => {
  it('keeps the original IPv4 CIDR semantics', () => {
    expect(cidrContains('203.0.113.0/24', '203.0.113.42')).toBe(true);
    expect(cidrContains('203.0.113.0/24', '203.0.114.42')).toBe(false);
    expect(cidrContains('0.0.0.0/0', '198.51.100.1')).toBe(true);
    expect(cidrContains('10.42.0.10', '10.42.0.10')).toBe(true);
    expect(cidrContains('10.42.0.10', '10.42.0.11')).toBe(false);
    expect(cidrContains('203.0.113.0/33', '203.0.113.1')).toBe(false);
    expect(cidrContains('not-an-address/24', '203.0.113.1')).toBe(false);
    expect(cidrContains('203.0.113.0/24', 'not-an-address')).toBe(false);
    expect(cidrContains('203.0.113.0/24', '203.0.113.256')).toBe(false);
  });

  it('parses and normalises IPv6 text forms', () => {
    expect(ipv6Number('::')).toBe(0n);
    expect(ipv6Number('::1')).toBe(1n);
    expect(ipv6Number('2001:db8::1')).toBe(ipv6Number('2001:0db8:0000:0000:0000:0000:0000:0001'));
    expect(ipv6Number('::ffff:203.0.113.9')).toBe(0xffffn << 32n | 0xcb007109n);
    expect(ipv6Number('fe80::1%eth0')).toBe(ipv6Number('fe80::1'));
    expect(ipv6Number('[2001:db8::1]')).toBe(ipv6Number('2001:db8::1'));
    expect(ipv6Number('1:2:3:4:5:6:7')).toBeUndefined();
    expect(ipv6Number('1::2::3')).toBeUndefined();
    expect(ipv6Number('gggg::1')).toBeUndefined();
    expect(ipv6Number('10.42.0.1')).toBeUndefined();
    expect(normalizeAddress('2001:0DB8:0000:0000:0000:0000:0000:0001')).toBe('2001:db8::1');
    expect(normalizeAddress('0:0:0:0:0:0:0:0')).toBe('::');
    expect(normalizeAddress('010.042.000.001')).toBe('10.42.0.1');
    expect(formatIpv6(0n)).toBe('::');
    expect(formatIpv6(1n)).toBe('::1');
    expect(addressFamily('10.0.0.1')).toBe(4);
    expect(addressFamily('fd42::a')).toBe(6);
    expect(addressFamily('example.test')).toBeUndefined();
  });

  it('does real 128-bit containment for IPv6 CIDRs', () => {
    expect(cidrContains('2001:db8::/32', '2001:db8:dead:beef::1')).toBe(true);
    expect(cidrContains('2001:db8::/32', '2001:db9::1')).toBe(false);
    expect(cidrContains('fd42::/64', 'fd42::a2a')).toBe(true);
    expect(cidrContains('fd42::/64', 'fd43::a2a')).toBe(false);
    expect(cidrContains('::/0', '2001:db8::1')).toBe(true);
    expect(cidrContains('::1/128', '::1')).toBe(true);
    expect(cidrContains('::1/128', '::2')).toBe(false);
    expect(cidrContains('2001:db8::/129', '2001:db8::1')).toBe(false);
    // Families do not silently cross.
    expect(cidrContains('2001:db8::/32', '203.0.113.1')).toBe(false);
    expect(cidrContains('203.0.113.0/24', '2001:db8::1')).toBe(false);
  });

  it('unwraps IPv4-mapped addresses so a v4 rule still governs them', () => {
    expect(cidrContains('203.0.113.0/24', '::ffff:203.0.113.9')).toBe(true);
    expect(cidrContains('203.0.113.0/24', '::ffff:169.254.169.254')).toBe(false);
    expect(cidrContains('::ffff:0:0/96', '203.0.113.9')).toBe(true);
  });
});

describe('seeded randomness', () => {
  it('reproduces the same stream for the same seed and differs across seeds', () => {
    const draw = (seed: number) => { const stream = new SeededRandom(seed); return Array.from({ length: 8 }, () => stream.next()); };
    expect(draw(7)).toEqual(draw(7));
    expect(draw(7)).not.toEqual(draw(8));
    expect(draw(7).every((value) => value >= 0 && value < 1)).toBe(true);
    expect(new Set(draw(7)).size).toBe(8);
    expect(new SeededRandom(0).next()).toBe(new SeededRandom(0).next());
    const bounded = new SeededRandom(5);
    expect(Array.from({ length: 20 }, () => bounded.int(10)).every((value) => Number.isInteger(value) && value >= 0 && value < 10)).toBe(true);
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 1 — DNS rebinding / TOCTOU
 * -------------------------------------------------------------------------- */

describe('egress is pinned to the validated address (DNS rebinding)', () => {
  const cidrRule = gateway({ id: 'research', cidrs: ['203.0.113.0/24'], hostnames: ['*.example.test'], ports: [443], protocols: ['https'] });

  it('connects to the address that was validated, never re-resolving', async () => {
    let resolutions = 0;
    const { transport, seen } = okTransport('pinned');
    const fabric = fabricWith({
      externalResolver: async () => {
        resolutions += 1;
        // The attacker flips the answer the instant the check is done.
        return resolutions === 1 ? ['203.0.113.10'] : ['169.254.169.254'];
      },
      egressTransport: transport,
    });
    fabric.addGateway(cidrRule);
    const response = await fabric.request('alpha', 'https://api.example.test/metadata');
    expect(response.status).toBe(200);
    expect(resolutions).toBe(1);
    expect(seen).toHaveLength(1);
    expect(seen[0]!.address).toBe('203.0.113.10');
    expect(seen[0]!.hostname).toBe('api.example.test');
  });

  it('denies the request once the rebound answer is what the resolver reports', async () => {
    const { transport, seen } = okTransport();
    const fabric = fabricWith({ externalResolver: async () => ['169.254.169.254'], egressTransport: transport });
    fabric.addGateway(cidrRule);
    await expect(fabric.request('alpha', 'https://api.example.test/metadata')).rejects.toThrow('gateway denied');
    expect(seen).toHaveLength(0);
  });

  it('rejects a mixed answer where only some addresses satisfy the rule', async () => {
    const { transport, seen } = okTransport();
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10', '169.254.169.254'], egressTransport: transport });
    fabric.addGateway(cidrRule);
    await expect(fabric.request('alpha', 'https://api.example.test/x')).rejects.toThrow('gateway denied');
    expect(seen).toHaveLength(0);
  });

  it('refuses to connect when no answer can be pinned', async () => {
    const { transport, seen } = okTransport();
    const fabric = fabricWith({ externalResolver: async () => [], egressTransport: transport });
    fabric.addGateway(gateway({ id: 'open', cidrs: [], hostnames: ['*'] }));
    await expect(fabric.request('alpha', 'http://nowhere.test/')).rejects.toThrow('no validated address');
    expect(seen).toHaveLength(0);
  });

  it('pins an IPv4-mapped answer through the v4 rule', async () => {
    const { transport, seen } = okTransport();
    const fabric = fabricWith({ externalResolver: async () => ['::ffff:203.0.113.10'], egressTransport: transport });
    fabric.addGateway(cidrRule);
    await fabric.request('alpha', 'https://api.example.test/x');
    expect(seen[0]!.address).toBe('::ffff:203.0.113.10');
  });

  it('createPinnedLookup ignores the hostname entirely', () => {
    const lookupFn = createPinnedLookup('203.0.113.10');
    let single: unknown;
    lookupFn('rebound.example.test', {}, (_error, address) => { single = address; });
    expect(single).toBe('203.0.113.10');
    let all: unknown;
    lookupFn('rebound.example.test', { all: true }, (_error, address) => { all = address; });
    expect(all).toEqual([{ address: '203.0.113.10', family: 4 }]);
    let v6: unknown;
    createPinnedLookup('2001:db8::1')('anything', { all: true }, (_error, address) => { v6 = address; });
    expect(v6).toEqual([{ address: '2001:db8::1', family: 6 }]);
  });
});

describe('the default transport really pins the socket', () => {
  it('reaches a server that only the validated address can reach, with the Host header intact', async () => {
    // `rebound.invalid` has no DNS answer at all: the .invalid TLD is
    // guaranteed never to resolve. If the request arrives, the connection can
    // only have used the pinned address — no second resolution took place.
    const server = createServer((request, response) => {
      response.writeHead(200, { 'content-type': 'text/plain' });
      response.end(`host=${request.headers.host} path=${request.url}`);
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      let resolutions = 0;
      const fabric = fabricWith({
        externalResolver: async () => { resolutions += 1; return ['127.0.0.1']; },
      });
      fabric.addGateway(gateway({ id: 'lab', cidrs: ['127.0.0.0/8'], hostnames: ['rebound.invalid'], protocols: ['http'], ports: [port] }));
      const response = await fabric.request('alpha', `http://rebound.invalid:${port}/probe?x=1`);
      expect(response.status).toBe(200);
      expect(response.body).toBe(`host=rebound.invalid:${port} path=/probe?x=1`);
      expect(resolutions).toBe(1);
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('denies the same host when the validated answer falls outside the rule', async () => {
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10'] });
    fabric.addGateway(gateway({ id: 'lab', cidrs: ['127.0.0.0/8'], hostnames: ['rebound.invalid'], protocols: ['http'], ports: [80] }));
    await expect(fabric.request('alpha', 'http://rebound.invalid/probe')).rejects.toThrow('gateway denied');
  });
});

describe('redirects', () => {
  function redirectTransport(location: string): { transport: EgressTransport; seen: PinnedEgressRequest[] } {
    const seen: PinnedEgressRequest[] = [];
    const transport: EgressTransport = async (request): Promise<PinnedEgressResponse> => {
      seen.push(request);
      if (seen.length === 1) return { status: 302, statusText: 'Found', headers: { location }, body: '' };
      return { status: 200, statusText: 'OK', headers: {}, body: 'followed' };
    };
    return { transport, seen };
  }

  it('returns the 3xx untouched by default (no follow-up connection)', async () => {
    const { transport, seen } = redirectTransport('https://evil.test/');
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10'], egressTransport: transport });
    fabric.addGateway(gateway({ hostnames: ['docs.test'], protocols: ['https'] }));
    const response = await fabric.request('alpha', 'https://docs.test/a');
    expect(response.status).toBe(302);
    expect(seen).toHaveLength(1);
  });

  it('re-validates the redirect target through canEgress before following', async () => {
    const { transport, seen } = redirectTransport('https://evil.test/');
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10'], egressTransport: transport });
    fabric.addGateway(gateway({ hostnames: ['docs.test'], protocols: ['https'] }));
    await expect(fabric.request('alpha', 'https://docs.test/a', 'GET', undefined, { maxRedirects: 3 })).rejects.toThrow('gateway denied');
    expect(seen).toHaveLength(1);
  });

  it('follows and re-pins an allowed redirect target', async () => {
    const { transport, seen } = redirectTransport('https://mirror.test/b');
    const fabric = fabricWith({
      externalResolver: async (hostname) => (hostname === 'docs.test' ? ['203.0.113.10'] : ['203.0.113.77']),
      egressTransport: transport,
    });
    fabric.addGateway(gateway({ hostnames: ['docs.test', 'mirror.test'], protocols: ['https'], cidrs: ['203.0.113.0/24'] }));
    const response = await fabric.request('alpha', 'https://docs.test/a', 'GET', undefined, { maxRedirects: 3 });
    expect(response.body).toBe('followed');
    expect(seen.map((request) => request.address)).toEqual(['203.0.113.10', '203.0.113.77']);
    expect(seen[1]!.hostname).toBe('mirror.test');
  });

  it('refuses to follow a non-http redirect scheme', async () => {
    const { transport } = redirectTransport('file:///etc/passwd');
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10'], egressTransport: transport });
    fabric.addGateway(gateway({ hostnames: ['docs.test'], protocols: ['https'] }));
    await expect(fabric.request('alpha', 'https://docs.test/a', 'GET', undefined, { maxRedirects: 2 })).rejects.toThrow('unsupported redirect scheme');
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 2 — DNS depth
 * -------------------------------------------------------------------------- */

describe('dns', () => {
  it('keeps loopback names out of the shared zone and answers them per computer', () => {
    const fabric = fabricWith();
    fabric.addDns('localhost', '10.42.0.99');
    fabric.addDns('127.0.0.5', '10.42.0.99');
    fabric.addDns('0.0.0.0', '10.42.0.99');
    expect(fabric.listDns().some((record) => record.name === 'localhost')).toBe(false);
    expect(fabric.resolve('localhost', 'alpha')).toBe('127.0.0.1');
    expect(fabric.resolve('127.0.0.5', 'beta')).toBe('127.0.0.5');
    expect(fabric.resolve('::1')).toBe('::1');
    expect(() => fabric.resolve('localhost', 'ghost')).toThrow('unknown computer');
  });

  it('expires cached records at their TTL while authoritative records persist', () => {
    let clock = 1_000_000;
    const fabric = new InternetFabric({ clock: () => clock });
    fabric.attach(spec('alpha', '10.42.0.10'));
    fabric.addDns('always.seed.local', '10.42.0.50', 60);
    fabric.cacheRecord({ name: 'cached.seed.local', type: 'A', value: '203.0.113.5', ttl: 30 });
    expect(fabric.resolve('cached.seed.local')).toBe('203.0.113.5');
    clock += 29_000;
    expect(fabric.resolve('cached.seed.local')).toBe('203.0.113.5');
    clock += 2_000;
    expect(fabric.resolve('cached.seed.local')).toBeUndefined();
    expect(fabric.listDnsRecords().some((record) => record.name === 'cached.seed.local')).toBe(false);
    // The fabric is the zone authority for seed.local, so its own data never ages out.
    clock += 10 * 365 * 24 * 3_600_000;
    expect(fabric.resolve('always.seed.local')).toBe('10.42.0.50');
    expect(fabric.resolve('alpha.seed.local')).toBe('10.42.0.10');
  });

  it('follows CNAME chains and reports the walked chain', () => {
    const fabric = fabricWith();
    fabric.addRecord({ name: 'www.seed.local', type: 'CNAME', value: 'edge.seed.local' });
    fabric.addRecord({ name: 'edge.seed.local', type: 'CNAME', value: 'alpha.seed.local' });
    expect(fabric.resolve('www.seed.local')).toBe('10.42.0.10');
    const resolution = fabric.resolveDetailed('www.seed.local');
    expect(resolution.status).toBe('ok');
    expect(resolution.chain).toEqual(['www.seed.local', 'edge.seed.local', 'alpha.seed.local']);
  });

  it('detects CNAME loops instead of spinning', () => {
    const fabric = fabricWith();
    fabric.addRecord({ name: 'a.seed.local', type: 'CNAME', value: 'b.seed.local' });
    fabric.addRecord({ name: 'b.seed.local', type: 'CNAME', value: 'c.seed.local' });
    fabric.addRecord({ name: 'c.seed.local', type: 'CNAME', value: 'a.seed.local' });
    const resolution = fabric.resolveDetailed('a.seed.local');
    expect(resolution.status).toBe('loop');
    expect(resolution.chain).toEqual(['a.seed.local', 'b.seed.local', 'c.seed.local', 'a.seed.local']);
    expect(fabric.resolve('a.seed.local')).toBeUndefined();
    fabric.addRecord({ name: 'self.seed.local', type: 'CNAME', value: 'self.seed.local' });
    expect(fabric.resolveDetailed('self.seed.local').status).toBe('loop');
  });

  it('caches NXDOMAIN answers and invalidates them when the name appears', () => {
    let clock = 5_000_000;
    const fabric = new InternetFabric({ clock: () => clock, negativeTtlSeconds: 10 });
    fabric.attach(spec('alpha', '10.42.0.10'));
    expect(fabric.resolveDetailed('ghost.seed.local').source).toBe('none');
    expect(fabric.resolveDetailed('ghost.seed.local').source).toBe('negative-cache');
    expect(fabric.listNegativeCache().map((record) => record.name)).toContain('ghost.seed.local');
    clock += 11_000;
    expect(fabric.listNegativeCache()).toHaveLength(0);
    fabric.addDns('ghost.seed.local', '10.42.0.60');
    expect(fabric.resolve('ghost.seed.local')).toBe('10.42.0.60');
    expect(fabric.listNegativeCache()).toHaveLength(0);
  });

  it('serves coherent dual-stack answers with A preferred', () => {
    const fabric = fabricWith();
    expect(fabric.resolve('alpha.seed.local')).toBe('10.42.0.10');
    expect(fabric.resolveAll('alpha.seed.local', { family: 6 })).toEqual(['fd42::a2a:a']);
    expect(fabric.resolveAll('alpha.seed.local')).toEqual(['10.42.0.10', 'fd42::a2a:a']);
    expect(fabric.resolveDetailed('alpha.seed.local', { family: 6 }).records.map((record) => record.type)).toEqual(['AAAA']);
    fabric.addRecord({ name: 'v6only.seed.local', type: 'AAAA', value: '2001:db8::5' });
    expect(fabric.resolveDetailed('v6only.seed.local', { family: 4 }).status).toBe('nodata');
    expect(fabric.resolve('v6only.seed.local')).toBe('2001:db8::5');
    // listDns() stays the legacy A/CNAME view so existing snapshots are unchanged.
    expect(fabric.listDns().every((record) => record.type === 'A' || record.type === 'CNAME')).toBe(true);
    expect(fabric.listDnsRecords().some((record) => record.type === 'AAAA')).toBe(true);
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 3 — ports
 * -------------------------------------------------------------------------- */

describe('port allocation', () => {
  it('never hands out a port that is already in use', () => {
    const fabric = fabricWith();
    const ports = Array.from({ length: 200 }, () => fabric.allocateEphemeralPort('alpha'));
    expect(new Set(ports).size).toBe(200);
    expect(ports.every((port) => port >= 49152 && port <= 65535)).toBe(true);
    // Another computer has an independent port space.
    expect(fabric.allocateEphemeralPort('beta')).toBe(49152);
  });

  it('releases ephemeral ports on close and reuses them only afterwards', async () => {
    const fabric = fabricWith({ ephemeralPortRange: [50000, 50002] as [number, number] });
    fabric.bindStream({ id: 'echo', computerId: 'beta', host: 'beta.seed.local', port: 7, handle: (data) => data });
    const first = await fabric.connect('alpha', 'beta.seed.local', 7);
    const second = await fabric.connect('alpha', 'beta.seed.local', 7);
    expect(first.localPort).not.toBe(second.localPort);
    await expect(fabric.connect('alpha', 'beta.seed.local', 7)).resolves.toBeDefined();
    await expect(fabric.connect('alpha', 'beta.seed.local', 7)).rejects.toThrow('EADDRNOTAVAIL');
    first.close();
    await expect(fabric.connect('alpha', 'beta.seed.local', 7)).resolves.toBeDefined();
  });

  it('will not allocate an ephemeral port over a bound listener', () => {
    const fabric = fabricWith({ ephemeralPortRange: [49152, 49153] as [number, number] });
    fabric.registerService({
      id: 'listener', computerId: 'alpha', host: 'alpha.seed.local', port: 49152, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    });
    expect(fabric.allocateEphemeralPort('alpha', 'tcp', '10.42.0.10')).toBe(49153);
    expect(() => fabric.allocateEphemeralPort('alpha', 'tcp', '10.42.0.10')).toThrow('EADDRNOTAVAIL');
  });

  it('rejects a second bind of the same address and port (EADDRINUSE)', () => {
    const fabric = fabricWith();
    const service = (id: string, host: string, port: number) => ({
      id, computerId: 'alpha', host, port, protocol: 'http' as const, pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    });
    fabric.registerService(service('one', 'alpha.seed.local', 8080));
    expect(() => fabric.registerService(service('two', 'alpha.seed.local', 8080))).toThrow('EADDRINUSE');
    // Name-based virtual hosts on the same listening socket remain legal.
    expect(() => fabric.registerService(service('vhost', 'shop.seed.local', 8080))).not.toThrow();
    // Re-registering the same owner is idempotent.
    expect(() => fabric.registerService(service('one', 'alpha.seed.local', 8080))).not.toThrow();
    // A wildcard bind collides with everything on the port.
    expect(() => fabric.registerService(service('wild', '0.0.0.0', 8080))).toThrow('EADDRINUSE');
    // Loopback is a different address, and a different computer is unaffected.
    expect(() => fabric.registerService(service('loop', '127.0.0.1', 8080))).not.toThrow();
    expect(() => fabric.registerService({ ...service('remote', 'beta.seed.local', 8080), computerId: 'beta' })).not.toThrow();
  });

  it('keeps the tcp and udp port spaces separate and frees them on unbind', () => {
    const fabric = fabricWith();
    fabric.registerService({
      id: 'http', computerId: 'alpha', host: 'alpha.seed.local', port: 53, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    });
    const handle = fabric.bindDatagram({ id: 'dns', computerId: 'alpha', host: 'alpha.seed.local', port: 53, handle: () => undefined });
    expect(() => fabric.bindDatagram({ id: 'dns2', computerId: 'alpha', host: 'alpha.seed.local', port: 53, handle: () => undefined })).toThrow('EADDRINUSE');
    handle.close();
    expect(() => fabric.bindDatagram({ id: 'dns2', computerId: 'alpha', host: 'alpha.seed.local', port: 53, handle: () => undefined })).not.toThrow();
    expect(fabric.listPortBindings('alpha').filter((binding) => binding.transport === 'udp')).toHaveLength(1);
  });

  it('frees every binding a terminated process owned', () => {
    const fabric = fabricWith();
    fabric.registerService({
      id: 'httpd', computerId: 'alpha', host: 'alpha.seed.local', port: 9000, protocol: 'http', pid: 42,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    });
    fabric.bindDatagram({ id: 'syslog', computerId: 'alpha', host: 'alpha.seed.local', port: 514, pid: 42, handle: () => undefined });
    fabric.bindStream({ id: 'echo', computerId: 'alpha', host: 'alpha.seed.local', port: 7, pid: 42, handle: (data) => data });
    expect(fabric.unregisterServicesForProcess('alpha', 42).sort()).toEqual(['echo', 'httpd', 'syslog']);
    expect(fabric.listPortBindings('alpha')).toHaveLength(0);
    expect(() => fabric.registerService({
      id: 'httpd-2', computerId: 'alpha', host: 'alpha.seed.local', port: 9000, protocol: 'http', pid: 43,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    })).not.toThrow();
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 4 — socket retention
 * -------------------------------------------------------------------------- */

describe('socket retention', () => {
  it('prunes closed sockets beyond the retention cap', async () => {
    const fabric = new InternetFabric({ maxClosedSockets: 5 });
    fabric.attach(spec('alpha', '10.42.0.10'));
    fabric.attach(spec('beta', '10.42.0.20'));
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'hi' }),
    });
    for (let index = 0; index < 40; index += 1) await fabric.request('alpha', 'http://beta.seed.local/');
    const sockets = fabric.listSockets();
    expect(sockets.filter((socket) => socket.state === 'CLOSED')).toHaveLength(5);
    expect(sockets.filter((socket) => socket.state === 'LISTEN')).toHaveLength(1);
    expect(sockets.length).toBeLessThanOrEqual(6);
  });

  it('ages closed sockets out by time as well', async () => {
    let clock = 0;
    const fabric = new InternetFabric({ clock: () => clock, closedSocketTtlMs: 1_000 });
    fabric.attach(spec('alpha', '10.42.0.10'));
    fabric.attach(spec('beta', '10.42.0.20'));
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'hi' }),
    });
    await fabric.request('alpha', 'http://beta.seed.local/');
    expect(fabric.listSockets('alpha')).toHaveLength(1);
    clock += 5_000;
    await fabric.request('alpha', 'http://beta.seed.local/');
    expect(fabric.listSockets('alpha')).toHaveLength(1);
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 5 — transport realism
 * -------------------------------------------------------------------------- */

describe('transport model', () => {
  it('defaults to the historical deterministic ping output', () => {
    const fabric = fabricWith();
    expect(fabric.ping('alpha', 'beta.seed.local')).toBe(
      'PING beta.seed.local (10.42.0.20): 56 data bytes\n'
      + '64 bytes from 10.42.0.20: icmp_seq=0 ttl=64 time=0.42 ms\n'
      + '--- beta.seed.local ping statistics ---\n'
      + '1 packets transmitted, 1 received, 0.0% packet loss',
    );
    expect(fabric.ping('alpha', 'nowhere.seed.local')).toBe('ping: cannot resolve nowhere.seed.local: unknown host');
    expect(() => fabric.ping('ghost', 'beta.seed.local')).toThrow('unknown computer');
  });

  it('applies per-link latency, loss and bandwidth from the topology', () => {
    const fabric = fabricWith({ seed: 99 });
    fabric.configureLink({ id: 'wan', from: 'alpha', to: 'beta', profile: { latencyMs: 12, lossRate: 0.5, bandwidthBps: 1_000_000 } });
    expect(fabric.linkProfile('alpha', 'beta').latencyMs).toBe(12);
    expect(fabric.linkProfile('alpha', 'alpha').latencyMs).toBe(0.42);
    const output = fabric.ping('alpha', 'beta.seed.local', 10);
    expect(output).toMatch(/10 packets transmitted, \d+ received, \d+\.\d% packet loss/);
    expect(output).toContain('Request timeout for icmp_seq');
    // Same seed and same topology ⇒ byte-identical output.
    const replica = fabricWith({ seed: 99 });
    replica.configureLink({ id: 'wan', from: 'alpha', to: 'beta', profile: { latencyMs: 12, lossRate: 0.5, bandwidthBps: 1_000_000 } });
    expect(replica.ping('alpha', 'beta.seed.local', 10)).toBe(output);
  });

  it('carries monotonically increasing sequence, ack and window values', async () => {
    const fabric = fabricWith();
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'response-body' }),
    });
    await fabric.request('alpha', 'http://beta.seed.local/api', 'POST', 'request-body');
    const packets = fabric.listPackets().filter((packet) => packet.sourcePort !== undefined) as NetworkPacketTrace[];
    const client = packets.filter((packet) => packet.source === '10.42.0.10');
    const syn = client.find((packet) => packet.flags?.includes('SYN'))!;
    const push = client.find((packet) => packet.summary === 'POST /api')!;
    const fin = client.find((packet) => packet.flags?.includes('FIN'))!;
    expect(syn.sequence).toBeGreaterThanOrEqual(0);
    expect(syn.window).toBe(65535);
    expect(syn.mss).toBe(1460);
    expect(push.sequence).toBe((syn.sequence! + 1) >>> 0);
    expect(push.ack).toBeGreaterThan(0);
    expect(fin.sequence).toBe((push.sequence! + 'request-body'.length) >>> 0);
    const serverPush = packets.find((packet) => packet.summary === '200 response')!;
    expect(serverPush.ack).toBe((push.sequence! + 'request-body'.length) >>> 0);
    expect(serverPush.window).toBeGreaterThan(0);
  });

  it('fragments oversized payloads when the link enables it', async () => {
    const fabric = fabricWith();
    fabric.configureLink({ id: 'lan', from: 'alpha', to: 'beta', profile: { mtu: 600, fragment: true } });
    const body = 'x'.repeat(2000);
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'ok' }),
    });
    await fabric.request('alpha', 'http://beta.seed.local/upload', 'POST', body);
    const segments = fabric.listPackets().filter((packet) => packet.summary.startsWith('POST /upload'));
    expect(segments.length).toBeGreaterThan(1);
    expect(segments[0]!.summary).toBe('POST /upload');
    expect(segments[0]!.fragment).toEqual({ index: 0, count: segments.length, offset: 0, more: true });
    expect(segments.at(-1)!.fragment?.more).toBe(false);
    expect(segments.reduce((sum, packet) => sum + packet.bytes, 0)).toBe(body.length);
    // Sequence numbers advance by the segment offset.
    expect(segments[1]!.sequence).toBe((segments[0]!.sequence! + 560) >>> 0);
  });

  it('emits retransmissions on a lossy link', async () => {
    const fabric = fabricWith({ seed: 3 });
    fabric.configureLink({ id: 'lossy', from: 'alpha', to: 'beta', profile: { lossRate: 1 } });
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'ok' }),
    });
    await fabric.request('alpha', 'http://beta.seed.local/');
    expect(fabric.listPackets().some((packet) => packet.retransmit === true)).toBe(true);
    expect(fabric.listPackets().some((packet) => packet.summary.includes('[retransmission]'))).toBe(true);
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 6 — UDP and raw sockets
 * -------------------------------------------------------------------------- */

describe('udp', () => {
  it('delivers datagrams and returns the reply', async () => {
    const fabric = fabricWith();
    const received: string[] = [];
    fabric.bindDatagram({
      id: 'echo', computerId: 'beta', host: 'beta.seed.local', port: 9999,
      handle: (message) => { received.push(message.payload); return `echo:${message.payload}`; },
    });
    const result = await fabric.sendDatagram('alpha', 'beta.seed.local', 9999, 'ping');
    expect(result.delivered).toBe(true);
    expect(result.reply).toBe('echo:ping');
    expect(received).toEqual(['ping']);
    const udpPackets = fabric.listPackets().filter((packet) => packet.protocol === 'udp');
    expect(udpPackets).toHaveLength(2);
    expect(udpPackets[0]!.destinationPort).toBe(9999);
  });

  it('reports port-unreachable when nothing is bound', async () => {
    const fabric = fabricWith();
    const result = await fabric.sendDatagram('alpha', 'beta.seed.local', 9999, 'ping');
    expect(result.delivered).toBe(false);
    expect(result.dropped).toBe(false);
    expect(fabric.listPackets().some((packet) => packet.summary.includes('destination port unreachable'))).toBe(true);
  });

  it('drops datagrams on a lossy link without delivering them', async () => {
    const fabric = fabricWith({ seed: 11 });
    fabric.configureLink({ id: 'lossy', from: 'alpha', to: 'beta', profile: { lossRate: 1 } });
    let delivered = 0;
    fabric.bindDatagram({ id: 'sink', computerId: 'beta', host: 'beta.seed.local', port: 9999, handle: () => { delivered += 1; return undefined; } });
    const result = await fabric.sendDatagram('alpha', 'beta.seed.local', 9999, 'ping');
    expect(result.dropped).toBe(true);
    expect(delivered).toBe(0);
  });

  it('keeps loopback UDP listeners isolated per computer', async () => {
    const fabric = fabricWith();
    fabric.bindDatagram({ id: 'local', computerId: 'alpha', host: 'localhost', port: 5300, handle: () => 'alpha-only' });
    expect((await fabric.sendDatagram('alpha', 'localhost', 5300, 'q')).reply).toBe('alpha-only');
    expect((await fabric.sendDatagram('beta', 'localhost', 5300, 'q')).delivered).toBe(false);
    expect((await fabric.sendDatagram('beta', 'alpha.seed.local', 5300, 'q')).delivered).toBe(false);
  });

  it('answers DNS over UDP from the zone', async () => {
    const fabric = fabricWith();
    fabric.attach(spec('resolver', '10.42.0.2'));
    fabric.serveDns('resolver');
    const answer = await fabric.resolveOverUdp('alpha', 'beta.seed.local');
    expect(answer.status).toBe('ok');
    expect(answer.addresses[0]).toBe('10.42.0.20');
    const v6 = await fabric.resolveOverUdp('alpha', 'beta.seed.local', { family: 6 });
    expect(v6.addresses[0]).toBe('fd42::a2a:14');
    const missing = await fabric.resolveOverUdp('alpha', 'nope.seed.local');
    expect(missing.status).toBe('nxdomain');
    expect(fabric.listDatagramServices().map((service) => service.id)).toEqual(['seed-dnsd-resolver']);
  });
});

describe('raw tcp streams', () => {
  it('carries arbitrary bytes over a bound stream service', async () => {
    const fabric = fabricWith();
    const log: string[] = [];
    fabric.bindStream({
      id: 'echo', computerId: 'beta', host: 'beta.seed.local', port: 7,
      onConnect: () => '220 seed-echo ready',
      handle: (data) => { log.push(data); return data.toUpperCase(); },
      onClose: () => log.push('<closed>'),
    });
    const connection = await fabric.connect('alpha', 'beta.seed.local', 7);
    expect(connection.remoteAddress).toBe('10.42.0.20');
    expect(await connection.send('hello')).toBe('HELLO');
    expect(await connection.send('world')).toBe('WORLD');
    connection.close();
    expect(connection.closed).toBe(true);
    expect(log).toEqual(['hello', 'world', '<closed>']);
    await expect(connection.send('again')).rejects.toThrow('EPIPE');
    expect(fabric.listPackets().some((packet) => packet.summary === 'stream 5 bytes')).toBe(true);
    expect(fabric.listStreamServices().map((service) => service.id)).toEqual(['echo']);
  });

  it('refuses a connection to a loopback stream from another computer', async () => {
    const fabric = fabricWith();
    fabric.bindStream({ id: 'local', computerId: 'alpha', host: 'localhost', port: 7000, handle: (data) => data });
    await expect(fabric.connect('alpha', 'localhost', 7000)).resolves.toBeDefined();
    await expect(fabric.connect('beta', 'localhost', 7000)).rejects.toThrow('connection refused');
    await expect(fabric.connect('beta', 'alpha.seed.local', 7000)).rejects.toThrow('connection refused');
  });
});

/* -------------------------------------------------------------------------- *
 * DEFECT 7 + routing / ARP / NAT
 * -------------------------------------------------------------------------- */

describe('interfaces, routing, neighbours and NAT', () => {
  it('provisions a dual-stack interface, loopback and a default route per computer', () => {
    const fabric = fabricWith();
    const interfaces = fabric.listInterfaces('alpha');
    expect(interfaces.map((entry) => entry.name).sort()).toEqual(['lo', 'seed0']);
    const primary = interfaces.find((entry) => entry.name === 'seed0')!;
    expect(primary.addresses.map((address) => address.address)).toEqual(['10.42.0.10', 'fd42::a2a:a']);
    expect(primary.mac).toBe('02:42:0a:2a:00:0a');
    const loopback = interfaces.find((entry) => entry.name === 'lo')!;
    expect(loopback.addresses.map((address) => address.address)).toEqual(['127.0.0.1', '::1']);
    const routes = fabric.listRoutes('alpha');
    expect(routes.find((route) => route.destination === '0.0.0.0/0')?.via).toBe('10.42.0.1');
    expect(routes.some((route) => route.destination === 'fd42::/64')).toBe(true);
  });

  it('picks the longest matching prefix and honours static routes', () => {
    const fabric = fabricWith();
    expect(fabric.routeFor('alpha', '10.42.0.20')?.destination).toBe('10.42.0.0/24');
    expect(fabric.routeFor('alpha', '203.0.113.9')?.destination).toBe('0.0.0.0/0');
    expect(fabric.routeFor('alpha', '127.0.0.1')?.dev).toBe('lo');
    const route = fabric.addRoute({ computerId: 'alpha', destination: '203.0.113.0/24', via: '10.42.0.254', dev: 'seed0', metric: 50, family: 4 });
    expect(fabric.routeFor('alpha', '203.0.113.9')?.via).toBe('10.42.0.254');
    expect(fabric.removeRoute(route.id)).toBe(true);
    expect(fabric.routeFor('alpha', '203.0.113.9')?.destination).toBe('0.0.0.0/0');
    fabric.setInterfaceUp('alpha', 'seed0', false);
    expect(fabric.routeFor('alpha', '203.0.113.9')).toBeUndefined();
    expect(() => fabric.setInterfaceUp('alpha', 'wlan9', true)).toThrow('unknown interface');
  });

  it('reconfigures the network from topology values', () => {
    const fabric = new InternetFabric();
    fabric.attach(spec('alpha', '10.42.0.10'));
    fabric.configureNetwork({ cidr: '192.168.7.0/24', gateway: '192.168.7.1', dns: '192.168.7.2' });
    expect(fabric.networkConfig().gateway).toBe('192.168.7.1');
    expect(fabric.listRoutes('alpha').find((route) => route.destination === '0.0.0.0/0')?.via).toBe('192.168.7.1');
    expect(fabric.listInterfaces('alpha').find((entry) => entry.name === 'seed0')?.addresses[0]?.prefix).toBe(24);
  });

  it('resolves on-link peers with ARP and off-link destinations via the gateway', () => {
    const fabric = fabricWith();
    const peer = fabric.resolveNeighbor('alpha', '10.42.0.20')!;
    expect(peer.address).toBe('10.42.0.20');
    expect(peer.mac).toBe('02:42:0a:2a:00:14');
    expect(peer.resolvedBy).toBe('arp');
    const remote = fabric.resolveNeighbor('alpha', '203.0.113.9')!;
    expect(remote.address).toBe('10.42.0.1');
    expect(fabric.resolveNeighbor('alpha', 'fd42::a2a:14')?.resolvedBy).toBe('ndp');
    expect(fabric.listNeighbors('alpha').length).toBeGreaterThanOrEqual(2);
  });

  it('records a NAT binding for each egress flow', async () => {
    const { transport } = okTransport();
    const fabric = fabricWith({ externalResolver: async () => ['203.0.113.10'], egressTransport: transport });
    fabric.addGateway(gateway());
    await fabric.request('alpha', 'http://docs.test/');
    const bindings = fabric.listNatBindings();
    expect(bindings).toHaveLength(1);
    expect(bindings[0]!.insideAddress).toBe('10.42.0.10');
    expect(bindings[0]!.outsideAddress).toBe('198.51.100.1');
    expect(bindings[0]!.destination).toBe('203.0.113.10');
    const trace = fabric.listPackets().find((packet) => packet.flags?.includes('SYN') && !packet.flags.includes('ACK'))!;
    expect(trace.source).toBe('10.42.0.10');
    expect(trace.translatedSource).toBe('198.51.100.1');
  });

  it('routes an HTTP request addressed by IPv6 literal to the owning computer', async () => {
    const fabric = fabricWith();
    fabric.registerService({
      id: 'web', computerId: 'beta', host: 'beta.seed.local', port: 8080, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'v6' }),
    });
    const response = await fabric.request('alpha', 'http://[fd42::a2a:14]:8080/');
    expect(response.body).toBe('v6');
  });
});

/* -------------------------------------------------------------------------- *
 * Preserved behaviour
 * -------------------------------------------------------------------------- */

describe('preserved gateway and virtual-host behaviour', () => {
  it('keeps the CIDR-every-answer rule intact', () => {
    const fabric = new InternetFabric();
    fabric.addGateway({
      id: 'research-egress', name: 'research egress', enabled: true, direction: 'egress',
      protocols: ['https'], cidrs: ['203.0.113.0/24'], hostnames: ['*.example.test'], ports: [443], audit: true,
    });
    expect(fabric.canEgress('https', 'api.example.test', 443, ['203.0.113.10', '203.0.113.11'])?.id).toBe('research-egress');
    expect(fabric.canEgress('https', 'api.example.test', 443, ['203.0.113.10', '198.51.100.9'])).toBeUndefined();
    expect(fabric.canEgress('https', 'unrelated.test', 443, ['203.0.113.10'])).toBeUndefined();
    expect(fabric.canEgress('https', 'example.test', 443, ['203.0.113.10'])).toBeUndefined();
    expect(fabric.canEgress('http', 'api.example.test', 443, ['203.0.113.10'])).toBeUndefined();
    expect(fabric.canEgress('https', 'api.example.test', 8443, ['203.0.113.10'])).toBeUndefined();
    expect(fabric.canEgress('https', 'api.example.test', 443, [])).toBeUndefined();
    expect(fabric.setGatewayEnabled('research-egress', false).enabled).toBe(false);
    expect(fabric.canEgress('https', 'api.example.test', 443, ['203.0.113.10'])).toBeUndefined();
    expect(() => fabric.setGatewayEnabled('missing', true)).toThrow('unknown gateway rule');
  });

  it('gives exact virtual hosts precedence and hides loopback listeners from the NIC', async () => {
    const fabric = fabricWith();
    const service = (id: string, host: string, body: string) => ({
      id, computerId: 'beta', host, port: 8080, protocol: 'http' as const, pid: 1,
      handle: async () => ({ status: 200, headers: {}, body }),
    });
    fabric.registerService(service('nic', 'beta.seed.local', 'nic'));
    fabric.registerService(service('shop', 'shop.seed.local', 'shop'));
    fabric.registerService(service('loop', 'localhost', 'loop'));
    expect((await fabric.request('alpha', 'http://shop.seed.local:8080/')).body).toBe('shop');
    expect((await fabric.request('alpha', 'http://beta.seed.local:8080/')).body).toBe('nic');
    expect((await fabric.request('alpha', 'http://10.42.0.20:8080/')).body).toBe('nic');
    expect((await fabric.request('beta', 'http://localhost:8080/')).body).toBe('loop');
    expect((await fabric.request('beta', 'http://127.0.0.1:8080/')).body).toBe('loop');
    await expect(fabric.request('alpha', 'http://localhost:8080/')).rejects.toThrow('connection refused');
    fabric.unregisterService('shop.seed.local', 8080, 'beta');
    expect((await fabric.request('alpha', 'http://shop.seed.local:8080/')).body).toBe('nic');
  });

  it('releases the connection when a virtual handler throws', async () => {
    const fabric = fabricWith();
    fabric.registerService({
      id: 'broken', computerId: 'beta', host: 'beta.seed.local', port: 8080, protocol: 'http', pid: 1,
      handle: async () => { throw new Error('handler exploded'); },
    });
    await expect(fabric.request('alpha', 'http://beta.seed.local:8080/')).rejects.toThrow('handler exploded');
    expect(fabric.listSockets('alpha').every((socket) => socket.state === 'CLOSED')).toBe(true);
    expect(fabric.listPortBindings('alpha')).toHaveLength(0);
    expect(fabric.listPackets().some((packet) => packet.flags?.includes('RST'))).toBe(true);
    // The port is free again, so the next request still works.
    fabric.registerService({
      id: 'fixed', computerId: 'beta', host: 'other.seed.local', port: 8080, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: 'fine' }),
    });
    expect((await fabric.request('alpha', 'http://other.seed.local:8080/')).body).toBe('fine');
  });

  it('renders an interface summary from the real interface and route model', () => {
    const fabric = fabricWith();
    expect(fabric.interfaceSummary('alpha')).toBe(
      'seed0\n  ether 02:42:0a:2a:00:0a\n  inet 10.42.0.10/24\n  inet6 fd42::a2a:a/64\n  gateway 10.42.0.1\n  dns 10.42.0.2\n  mtu 1500\n  state UP',
    );
    fabric.configureNetwork({ gateway: '10.42.0.254', dns: '10.42.0.253' });
    expect(fabric.interfaceSummary('alpha')).toContain('gateway 10.42.0.254');
    expect(fabric.interfaceSummary('alpha')).toContain('dns 10.42.0.253');
  });

  it('rejects services on unknown computers', () => {
    const fabric = fabricWith();
    expect(() => fabric.registerService({
      id: 'ghost', computerId: 'nowhere', host: 'ghost.seed.local', port: 80, protocol: 'http', pid: 1,
      handle: async () => ({ status: 200, headers: {}, body: '' }),
    })).toThrow('unknown computer');
    expect(() => fabric.bindDatagram({ id: 'ghost', computerId: 'nowhere', host: 'x', port: 80, handle: () => undefined })).toThrow('unknown computer');
    expect(() => fabric.bindStream({ id: 'ghost', computerId: 'nowhere', host: 'x', port: 80, handle: () => undefined })).toThrow('unknown computer');
  });
});
