import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { seed2026Blueprint } from '@tcn-computer/ecosystem-seed-2026';
import { cidrContains, InternetFabric, SimulationRuntime } from '@tcn-computer/kernel';
import type { SimulationTopology } from '@tcn-computer/protocol';

describe('seed ecosystem kernel', () => {
  it('boots all three display computers plus the registry service node', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-kernel-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-test' });
    await runtime.initialize();
    const snapshot = runtime.snapshot();
    expect(snapshot.computers.filter((computer) => computer.spec.displays.length)).toHaveLength(3);
    expect(snapshot.computers.map((computer) => computer.spec.os)).toEqual(expect.arrayContaining(['macos', 'windows', 'ubuntu']));
    expect(snapshot.dns.find((record) => record.name === 'appstore.seed.local')?.value).toBe('10.42.0.2');
  });

  it('boots the supplied topology exactly without hidden computer, app, or gateway defaults', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-topology-'));
    const topology: SimulationTopology = structuredClone(seed2026Blueprint);
    topology.id = 'research-fixture';
    topology.computers = topology.computers.map((computer) => computer.spec.id === 'mac-studio'
      ? { ...computer, spec: { ...computer.spec, hostname: 'research-mac', ipv4: '10.42.0.11' }, thirdPartyAppIds: ['chromium'] }
      : computer);
    topology.gateways = [];
    const runtime = new SimulationRuntime({ topology, stateRoot, runId: 'run-topology-contract' });
    await runtime.initialize();
    const snapshot = runtime.snapshot();
    const mac = snapshot.computers.find((computer) => computer.spec.id === 'mac-studio')!;
    expect(snapshot.topology).toEqual({ id: 'research-fixture', version: topology.version });
    expect(mac.spec.hostname).toBe('research-mac');
    expect(mac.installedApps.some((app) => app.id === 'chromium')).toBe(true);
    expect(mac.installedApps.some((app) => app.id === 'chatgpt')).toBe(false);
    expect(snapshot.gateways).toEqual([]);
  });

  it('routes http between computers and records packet evidence', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-network-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-network' });
    await runtime.initialize();
    const response = await runtime.http('win-workstation', 'http://intranet.seed.local:8080/');
    expect(response.status).toBe(200);
    expect(response.body).toContain('factory control plane');
    expect(response.headers['content-type']).toBe('text/html; charset=utf-8');
    expect(response.headers['content-security-policy']).toContain("script-src 'unsafe-inline'");
    expect(response.body).toContain('id="js-runtime-status" data-state="pending"');
    expect(response.body).toContain("status.dataset.state = 'executed'");
    expect(response.body).toContain('id="execution-canvas"');
    expect(runtime.snapshot().packets.some((packet) => packet.source === '10.42.0.20' && packet.protocol === 'http')).toBe(true);
  });

  it('isolates localhost listeners per computer while preserving network-visible services', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-loopback-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-loopback' });
    await runtime.initialize();

    const macServe = await runtime.execute('mac-studio', 'echo mac-loopback > ~/Documents/loopback.txt; serve 9100 ~/Documents/loopback.txt localhost');
    const macServerPid = Number(macServe.stdout.match(/pid (\d+)/)?.[1]);
    expect(macServerPid).toBeGreaterThan(1);
    expect((await runtime.execute('mac-studio', 'curl http://localhost:9100/')).stdout).toBe('mac-loopback');
    expect((await runtime.execute('mac-studio', 'curl http://127.0.0.1:9100/')).stdout).toBe('mac-loopback');

    const remoteLocalhostBeforeListen = await runtime.execute('win-workstation', 'curl http://localhost:9100/');
    expect(remoteLocalhostBeforeListen.exitCode).toBe(1);
    expect(remoteLocalhostBeforeListen.stderr).toContain('connection refused: localhost:9100');
    await expect(runtime.http('win-workstation', 'http://mac-studio.seed.local:9100/')).rejects.toThrow('connection refused');
    await expect(runtime.http('ubuntu-dev', 'http://10.42.0.10:9100/')).rejects.toThrow('connection refused');

    await runtime.execute('win-workstation', 'echo windows-loopback > ~/Documents/loopback.txt; serve 9100 ~/Documents/loopback.txt 127.0.0.1');
    expect((await runtime.execute('win-workstation', 'curl http://localhost:9100/')).stdout).toBe('windows-loopback');
    expect((await runtime.execute('mac-studio', 'curl http://localhost:9100/')).stdout).toBe('mac-loopback');

    await runtime.execute('ubuntu-dev', 'echo network-visible > ~/Documents/visible.txt; serve 9200 ~/Documents/visible.txt ubuntu-dev.seed.local');
    expect((await runtime.execute('win-workstation', 'curl http://ubuntu-dev.seed.local:9200/')).stdout).toBe('network-visible');
    const localAliasDoesNotReachNicOnlyListener = await runtime.execute('ubuntu-dev', 'curl http://localhost:9200/');
    expect(localAliasDoesNotReachNicOnlyListener.exitCode).toBe(1);
    expect(localAliasDoesNotReachNicOnlyListener.stderr).toContain('connection refused: localhost:9200');

    const snapshot = runtime.snapshot();
    expect(snapshot.dns.some((record) => record.name === 'localhost')).toBe(false);
    for (const computerId of ['mac-studio', 'win-workstation', 'ubuntu-dev']) {
      expect((await runtime.execute(computerId, 'nslookup localhost')).stdout).toContain('Address: 127.0.0.1');
    }
    expect(snapshot.computers.find((computer) => computer.spec.id === 'mac-studio')?.sockets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ localAddress: '127.0.0.1', localPort: 9100, state: 'LISTEN' })]));
    expect(snapshot.computers.find((computer) => computer.spec.id === 'win-workstation')?.sockets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ localAddress: '127.0.0.1', localPort: 9100, state: 'LISTEN' })]));
    expect(snapshot.computers.find((computer) => computer.spec.id === 'ubuntu-dev')?.sockets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ localAddress: '10.42.0.30', localPort: 9200, state: 'LISTEN' })]));

    expect(runtime.terminateProcess('mac-studio', macServerPid).servicesStopped).toEqual([expect.stringMatching(/^httpd-/)]);
    expect((await runtime.execute('mac-studio', 'curl http://localhost:9100/')).exitCode).toBe(1);
    expect((await runtime.execute('win-workstation', 'curl http://localhost:9100/')).stdout).toBe('windows-loopback');
    expect(runtime.snapshot().computers.find((computer) => computer.spec.id === 'mac-studio')?.sockets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ localAddress: '127.0.0.1', localPort: 9100, state: 'CLOSED' })]));
  });

  it('persists virtual files as inode blobs with a path table', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-vfs-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-vfs' });
    await runtime.initialize();
    const result = await runtime.execute('mac-studio', 'echo trajectory-data > ~/Desktop/capture.txt');
    expect(result.exitCode).toBe(0);
    const vfs = runtime.getVfs('mac-studio');
    const inode = vfs.statSync('/Users/agent/Desktop/capture.txt');
    expect(inode?.kind).toBe('file');
    expect(await readFile(path.join(stateRoot, 'run-vfs', 'mac-studio', inode!.diskId, inode!.id), 'utf8')).toBe('trajectory-data');
    expect(vfs.hostLayout().paths['/Users/agent/Desktop/capture.txt']).toBe(inode!.id);
    expect(await vfs.verifyContent()).toEqual([]);
  });

  it('uses separate shell dialects over one typed kernel', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-shell-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-shell' });
    await runtime.initialize();
    expect((await runtime.execute('mac-studio', 'ps | grep WindowServer')).stdout).toContain('WindowServer');
    expect((await runtime.execute('win-workstation', 'Get-Process | findstr explorer')).stdout).toContain('explorer.exe');
    expect((await runtime.execute('ubuntu-dev', 'curl http://intranet.seed.local:8080/ | grep nominal')).stdout).toContain('nominal');
  });

  it('installs software through native and language package managers into the vfs', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-packages-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-packages' });
    await runtime.initialize();
    // Transcripts are the managers' authentic output, so these assert on real
    // manager phrasing; installation itself is proven by observable state below.
    expect((await runtime.execute('mac-studio', 'brew install ripgrep')).stdout).toContain('ripgrep');
    expect((await runtime.execute('win-workstation', 'winget install Docker.DockerDesktop')).stdout).toContain('Docker.DockerDesktop');
    expect((await runtime.execute('ubuntu-dev', 'apt install nginx')).stdout).toContain('Setting up nginx');
    expect((await runtime.execute('ubuntu-dev', 'pnpm add vite')).stdout).toContain('+ vite');
    const snapshot = runtime.snapshot();
    expect(snapshot.computers.find((computer) => computer.spec.id === 'mac-studio')?.packages.some((item) => item.manager === 'brew' && item.name === 'ripgrep')).toBe(true);
    expect(runtime.getVfs('win-workstation').statSync('/C/Program Files/Docker.DockerDesktop/seed-package.json')?.kind).toBe('file');
    // The point of installing is reachability: the binary must land on PATH and run.
    expect((await runtime.execute('mac-studio', 'which rg')).stdout.trim()).toBe('/opt/homebrew/bin/rg');
    expect((await runtime.execute('mac-studio', 'rg --version')).exitCode).toBe(0);
  });

  it('models git object storage and keeps Slack and Teams as independent services', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-git-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-git' });
    await runtime.initialize();
    const result = await runtime.execute('ubuntu-dev', 'mkdir ~/code; cd ~/code; git init; echo hello > README.md; git add README.md; git commit -m "initial commit"; git log --oneline');
    expect(result.stdout).toContain('initial commit');
    const repository = runtime.snapshot().computers.find((computer) => computer.spec.id === 'ubuntu-dev')?.repositories.find((item) => item.root === '/home/agent/code');
    expect(repository?.commits).toHaveLength(1);
    expect(runtime.getVfs('ubuntu-dev').statSync(`/home/agent/code/.git/objects/${repository!.head!.slice(0, 2)}/${repository!.head!.slice(2)}`)?.kind).toBe('file');
    const slackBefore = runtime.snapshot().collaborationServices.find((service) => service.id === 'slack')!;
    const teamsBefore = runtime.snapshot().collaborationServices.find((service) => service.id === 'teams')!;
    expect(slackBefore.host).toBe('slack.seed.local');
    expect(teamsBefore.host).toBe('teams.seed.local');
    expect(slackBefore.messages.map((message) => message.text)).not.toEqual(teamsBefore.messages.map((message) => message.text));
    const sent = await runtime.postCollaborationMessage('mac-studio', 'slack', 'agent-runs', 'Jacob', 'Slack-only release note');
    const slackPoll = await runtime.pollCollaboration('ubuntu-dev', 'slack', 'agent-runs', slackBefore.revision);
    const teamsPoll = await runtime.pollCollaboration('win-workstation', 'teams', 'agent-runs', teamsBefore.revision);
    expect(sent.serviceId).toBe('slack');
    expect(slackPoll.messages.map((message) => message.text)).toContain('Slack-only release note');
    expect(teamsPoll.messages).toHaveLength(0);
    expect(runtime.snapshot().packets.some((packet) => packet.destination === '10.42.0.2' && packet.summary === 'POST /api/channels/agent-runs/messages')).toBe(true);
  });

  it('executes installed JavaScript app bundles from the VFS with scoped filesystem and network APIs', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-app-runtime-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-app-runtime' });
    await runtime.initialize();
    const vscode = runtime.snapshot().computers.find((computer) => computer.spec.id === 'mac-studio')!.installedApps.find((app) => app.id === 'vscode')!;
    expect(runtime.getVfs('mac-studio').statSync(`${vscode.installPath}/main.seed.js`)?.kind).toBe('file');
    expect(runtime.getVfs('mac-studio').statSync(vscode.receiptPath)?.kind).toBe('file');
    const edited = await runtime.launchApp('mac-studio', 'vscode', { operation: 'edit', payload: { path: '/Users/agent/Documents/runtime-proof.md', content: '# written by VFS app code\n' } });
    expect(edited.status).toBe('completed');
    expect(await runtime.getVfs('mac-studio').readFile('/Users/agent/Documents/runtime-proof.md')).toContain('written by VFS app code');
    const browser = await runtime.launchApp('win-workstation', 'chromium', { operation: 'navigate', payload: { url: 'http://intranet.seed.local:8080/' } });
    expect(browser.status).toBe('completed');
    expect(JSON.stringify(browser.result)).toContain('factory control plane');
    expect(runtime.snapshot().appExecutions.map((execution) => execution.id)).toEqual(expect.arrayContaining([edited.id, browser.id]));
  });

  it('writes dependency metadata, lockfiles, receipts, and committed package transactions', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-package-transactions-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-package-transactions' });
    await runtime.initialize();
    expect((await runtime.execute('ubuntu-dev', 'apt update')).stdout).toContain('Reading package lists');
    await runtime.execute('ubuntu-dev', 'mkdir ~/runtime-project; cd ~/runtime-project; pnpm add vite');
    const projectManifest = JSON.parse(await runtime.getVfs('ubuntu-dev').readFile('/home/agent/runtime-project/package.json')) as { dependencies: Record<string, string> };
    expect(projectManifest.dependencies.vite).toMatch(/^\^/);
    expect(await runtime.getVfs('ubuntu-dev').readFile('/home/agent/runtime-project/pnpm-lock.yaml')).toContain('lockfileVersion');
    const computer = runtime.snapshot().computers.find((item) => item.spec.id === 'ubuntu-dev')!;
    const vite = computer.packages.find((item) => item.manager === 'pnpm' && item.name === 'vite')!;
    expect(vite.dependencies).toEqual(expect.arrayContaining(['esbuild', 'rollup']));
    expect(vite.integrity).toHaveLength(64);
    expect(computer.packageTransactions.map((transaction) => transaction.operation)).toEqual(expect.arrayContaining(['index-refresh', 'install']));
    await runtime.execute('ubuntu-dev', 'dpkg -i seed-agent_1.0_amd64.deb');
    expect(runtime.getVfs('ubuntu-dev').statSync('/var/lib/dpkg/info/seed-agent_1.0_amd64.deb.list')?.kind).toBe('file');
    await runtime.execute('ubuntu-dev', 'dpkg remove seed-agent_1.0_amd64.deb');
    expect(runtime.getVfs('ubuntu-dev').statSync('/var/lib/dpkg/info/seed-agent_1.0_amd64.deb.list')).toBeUndefined();
  });

  it('pushes and fetches commits through the shared Git smart-HTTP service without sharing local worktrees', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-git-remote-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-git-remote' });
    await runtime.initialize();
    const push = await runtime.execute('ubuntu-dev', 'mkdir ~/shared-src; cd ~/shared-src; git init; echo remote-proof > proof.txt; git add proof.txt; git commit -m "remote proof"; git remote add origin https://git.seed.local/seed/example.git; git push origin main');
    expect(push.exitCode).toBe(0);
    expect(push.stdout).toContain('main -> main');
    // Only committed content crosses the fabric. This untracked file is written
    // after the push and must never reach another computer's worktree.
    await runtime.execute('ubuntu-dev', 'cd ~/shared-src; echo never-shared > local-only.txt');
    const clone = await runtime.execute('win-workstation', 'mkdir ~/clones; cd ~/clones; git clone https://git.seed.local/seed/example.git shared; cd shared; git log --oneline');
    expect(clone.stdout).toContain('remote proof');
    expect(runtime.getVfs('ubuntu-dev').statSync('/home/agent/shared-src/proof.txt')?.kind).toBe('file');
    // Committed files materialize in the clone's working tree.
    expect(runtime.getVfs('win-workstation').statSync('/C/Users/agent/clones/shared/proof.txt')?.kind).toBe('file');
    expect(await runtime.getVfs('win-workstation').readFile('/C/Users/agent/clones/shared/proof.txt')).toContain('remote-proof');
    // Uncommitted local worktree state does not.
    expect(runtime.getVfs('win-workstation').statSync('/C/Users/agent/clones/shared/local-only.txt')).toBeUndefined();
    expect(runtime.snapshot().packets.some((packet) => packet.summary === 'POST /api/repos/seed/example.git/push')).toBe(true);
  });

  it('keeps host execution default-deny and preserves app data on uninstall', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-host-gateway-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-host-gateway' });
    await runtime.initialize();
    await expect(runtime.executeHost('mac-studio', 'vscode', process.execPath, ['--version'], process.cwd())).rejects.toThrow('host execution denied');
    const installed = runtime.snapshot().computers.find((computer) => computer.spec.id === 'mac-studio')!.installedApps.find((app) => app.id === 'vscode')!;
    await runtime.getVfs('mac-studio').writeFile(`${installed.dataPath}/preferences.json`, '{"theme":"dark"}');
    await runtime.uninstallApp('mac-studio', 'vscode');
    expect(runtime.getVfs('mac-studio').statSync(installed.installPath)).toBeUndefined();
    expect(runtime.getVfs('mac-studio').statSync(`${installed.dataPath}/preferences.json`)?.kind).toBe('file');
  });

  it('mutates gateway policy and process state through server-authoritative kernel methods', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-controls-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-controls' });
    await runtime.initialize();
    expect(runtime.snapshot().gateways.find((rule) => rule.id === 'docs-egress')?.enabled).toBe(true);
    runtime.setGatewayEnabled('win-workstation', 'docs-egress', false);
    expect(runtime.snapshot().gateways.find((rule) => rule.id === 'docs-egress')?.enabled).toBe(false);
    const explorer = runtime.snapshot().computers.find((computer) => computer.spec.id === 'win-workstation')!.processes.find((process) => process.executable === 'explorer.exe')!;
    expect(runtime.terminateProcess('win-workstation', explorer.pid).terminated).toBe(true);
    expect(runtime.snapshot().computers.find((computer) => computer.spec.id === 'win-workstation')!.processes.some((process) => process.pid === explorer.pid)).toBe(false);
    expect(runtime.trajectory.jsonl()).toContain('gateway.policy.set');
    expect(runtime.trajectory.jsonl()).toContain('process.terminate');
  });

  it('routes cloud-backed app operations through isolated services and persists local app state in the VFS', async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-app-state-'));
    const runtime = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'run-app-state' });
    await runtime.initialize();
    const chat = await runtime.launchApp('mac-studio', 'chatgpt', { operation: 'send-message', payload: { text: 'inspect the gateway policy', mode: 'work' } });
    expect(chat.status).toBe('completed');
    const chatgpt = runtime.snapshot().computers.find((computer) => computer.spec.id === 'mac-studio')!.installedApps.find((app) => app.id === 'chatgpt')!;
    const chatState = JSON.parse(await runtime.getVfs('mac-studio').readFile(`${chatgpt.dataPath}/state.json`)) as { lastServiceOperation: string; lastServiceResult: unknown };
    expect(chatState.lastServiceOperation).toBe('send-message');
    expect(JSON.stringify(chatState.lastServiceResult)).toContain('chatgpt-backend');

    const zoom = await runtime.launchApp('ubuntu-dev', 'zoom', { operation: 'join-call', payload: { meeting: 'simulator-review' } });
    expect(zoom.status).toBe('completed');
    expect(runtime.snapshot().packets.some((packet) => packet.summary === 'POST /api/apps/zoom/operations/join-call')).toBe(true);

    const blender = await runtime.launchApp('ubuntu-dev', 'blender', { operation: 'edit-properties', payload: { property: 'mode', value: 'Edit Mode' } });
    expect(blender.status).toBe('completed');
    const blenderApp = runtime.snapshot().computers.find((computer) => computer.spec.id === 'ubuntu-dev')!.installedApps.find((app) => app.id === 'blender')!;
    const blenderState = JSON.parse(await runtime.getVfs('ubuntu-dev').readFile(`${blenderApp.dataPath}/state.json`)) as { lastOperation: string; events: Array<{ payload: Record<string, unknown> }> };
    expect(blenderState.lastOperation).toBe('edit-properties');
    expect(blenderState.events.at(-1)?.payload).toMatchObject({ property: 'mode', value: 'Edit Mode' });
  });

  it('enforces CIDR-constrained gateway rules against every resolved address', () => {
    expect(cidrContains('203.0.113.0/24', '203.0.113.42')).toBe(true);
    expect(cidrContains('203.0.113.0/24', '203.0.114.42')).toBe(false);
    const fabric = new InternetFabric();
    fabric.addGateway({
      id: 'research-egress', name: 'research egress', enabled: true, direction: 'egress',
      protocols: ['https'], cidrs: ['203.0.113.0/24'], hostnames: ['*.example.test'], ports: [443], audit: true,
    });
    expect(fabric.canEgress('https', 'api.example.test', 443, ['203.0.113.10', '203.0.113.11'])?.id).toBe('research-egress');
    expect(fabric.canEgress('https', 'api.example.test', 443, ['203.0.113.10', '198.51.100.9'])).toBeUndefined();
    expect(fabric.canEgress('https', 'unrelated.test', 443, ['203.0.113.10'])).toBeUndefined();
  });
});
