/** Long-lived sibling of `bridge.ts`: one process, one deterministic replay, applied incrementally.
 *
 * `bridge.ts` boots a fresh kernel and replays the whole event log for every
 * transition, so an episode pays the boot cost once per acting step. This
 * process reads newline-delimited requests of exactly the same shape on stdin
 * and writes one newline-delimited response per request. When a request's event
 * log extends the log already applied (same seed, same prefix) only the new
 * suffix is applied to the live runtime; otherwise the runtime is torn down and
 * rebuilt, which is byte-for-byte what `bridge.ts` does on every call.
 *
 * That is the whole difference: the same runtime object receives the same
 * sequence of calls in the same order with the same logical clock. Nothing here
 * is reachable unless `configuration['session']` is set, and
 * `research/credit-assignment/session_equivalence.py` checks the two transports
 * agree byte-for-byte on the emitted payload.
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import readline from 'node:readline';

let logicalTime=946684800000;
const NativeDate=Date;
class LogicalDate extends NativeDate {
  constructor(...args: any[]) { super(args.length ? args[0] : logicalTime); }
  static now() { return logicalTime; }
}
(globalThis as any).Date=LogicalDate;
const { setIdentitySeed }=await import('./packages/kernel/src/determinism.js');
const { SimulationRuntime }=await import('@tcn-computer/kernel');
const { seed2026Blueprint }=await import('@tcn-computer/ecosystem-seed-2026');

type Live={seed:any,runtime:any,computer:string,root:string,applied:string[],output:any};
let live:Live|null=null;

async function teardown() {
  if(!live) return;
  const root=live.root; live=null;
  // Deleting the episode's scratch directory is cleanup, not semantics. Under heavy
  // parallel load `rm` intermittently raises ENOTEMPTY here and in `bridge.ts`, and
  // in `bridge.ts` -- where it sits in a `finally` -- that kills the whole
  // transition. A failed cleanup must not end an episode, so retry and then give up.
  for(let attempt=0;attempt<3;attempt++) {
    try { await rm(root,{recursive:true,force:true}); return; }
    catch { await new Promise(resolve=>setTimeout(resolve,20)); }
  }
}

async function boot(seed:any):Promise<Live> {
  // `bridge.ts` boots in a fresh process, so the logical clock is always at the
  // epoch when the runtime is constructed and every boot-time stamp -- inode
  // times, process start times, the first trajectory entry -- is t = 0. A rebuild
  // inside a live session inherits the clock from the previous request, which
  // silently moved all of them and made a replayed episode diverge. Reset it.
  logicalTime=946684800000;
  setIdentitySeed(String(seed));
  const root=await mkdtemp(path.join(tmpdir(),'tcn-computer-'));
  const runtime=new SimulationRuntime({topology:seed2026Blueprint,stateRoot:root,runId:`episode-${seed}`});
  await runtime.initialize();
  const computer=runtime.snapshot().computers.find((c:any)=>c.spec.os==='ubuntu')!.spec.id;
  return {seed,runtime,computer,root,applied:[],output:{stdout:'',stderr:'',exitCode:0}};
}

async function apply(session:Live,event:any) {
  logicalTime=946684800000+Math.round(event.time*1000);
  const runtime=session.runtime, computer=session.computer;
  if(event.kind==='command') session.output=await runtime.execute(computer,event.command);
  else if(event.kind==='write') session.output=await runtime.writeTextFile(computer,event.path,event.text);
  else if(event.kind==='read') session.output=await runtime.readTextFile(computer,event.path);
  else if(event.kind==='launch') session.output=await runtime.launchApp(computer,event.app,event.request);
  else throw new Error(`unknown transition ${event.kind}`);
}

async function handle(request:any) {
  const events=(request.events??[]).map((e:any)=>JSON.stringify(e));
  const reusable=live!==null && String(live.seed)===String(request.seed)
    && live.applied.length<=events.length
    && live.applied.every((e,i)=>e===events[i]);
  if(!reusable) { await teardown(); live=await boot(request.seed); }
  const session=live!;
  for(let i=session.applied.length;i<events.length;i++) {
    await apply(session,request.events[i]);
    session.applied.push(events[i]);
  }
  logicalTime=946684800000+Math.round((request.time??0)*1000);
  const runtime=session.runtime, computer=session.computer;
  const files=runtime.listFiles(computer);
  const snapshot=runtime.snapshot();
  const result:any={output:session.output,files,prompt:runtime.getPrompt(computer),snapshot,trajectory:runtime.trajectory.snapshot()};
  const wanted=request.probe;
  if(wanted) {
    const vfs=runtime.getVfs(computer);
    const entries:any[]=[];
    const walk=(dir:string,depth:number)=>{
      let listing:any[];
      try { listing=vfs.list(dir); } catch { return; }
      for(const entry of listing) {
        entries.push({path:entry.path,name:entry.name,kind:entry.inode.kind,size:entry.inode.size,mode:entry.inode.mode});
        if(entry.inode.kind==='directory'&&depth>0) walk(entry.path,depth-1);
      }
    };
    walk(wanted.root??'/home/agent',wanted.depth??1);
    entries.sort((a,b)=>a.path<b.path?-1:a.path>b.path?1:0);
    const processes=((snapshot.computers.find((c:any)=>c.spec.id===computer)?.processes)??[])
      .map((p:any)=>({pid:p.pid,ppid:p.ppid,state:p.state,executable:p.executable}))
      .sort((a:any,b:any)=>a.pid-b.pid);
    const contents:any[]=[];
    for(const target of wanted.contents??[]) {
      try { contents.push({path:target,content:await vfs.readFile(target)}); }
      catch { contents.push({path:target,content:null}); }
    }
    result.probe={files:entries,processes,contents};
  }
  return JSON.stringify(result).split(session.root).join('<episode-storage>');
}

const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
process.stdout.write('{"ready":true}\n');
try {
  for await (const line of lines) {
    if(!line.trim()) continue;
    let payload:string;
    try { payload=await handle(JSON.parse(line)); }
    catch(error:any) { await teardown(); payload=JSON.stringify({error:String(error?.stack??error)}); }
    process.stdout.write(payload+'\n');
  }
} finally { await teardown(); }
