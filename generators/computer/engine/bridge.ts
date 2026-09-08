/** One isolated process owns one deterministic replay; never reads host time. */
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
const request=JSON.parse(readFileSync(0,'utf8'));
let logicalTime=946684800000;
const NativeDate=Date;
class LogicalDate extends NativeDate {
  constructor(...args: any[]) { super(args.length ? args[0] : logicalTime); }
  static now() { return logicalTime; }
}
(globalThis as any).Date=LogicalDate;
const { setIdentitySeed }=await import('./packages/kernel/src/determinism.js');
setIdentitySeed(String(request.seed));
const { SimulationRuntime }=await import('@tcn-computer/kernel');
const { seed2026Blueprint }=await import('@tcn-computer/ecosystem-seed-2026');
const root=await mkdtemp(path.join(tmpdir(),'tcn-computer-'));
try {
  const runtime=new SimulationRuntime({topology:seed2026Blueprint,stateRoot:root,runId:`episode-${request.seed}`});
  await runtime.initialize();
  const computer=runtime.snapshot().computers.find((c:any)=>c.spec.os==='ubuntu')!.spec.id;
  let output:any={stdout:'',stderr:'',exitCode:0};
  for(const event of request.events??[]) {
    logicalTime=946684800000+Math.round(event.time*1000);
    if(event.kind==='command') output=await runtime.execute(computer,event.command);
    else if(event.kind==='write') output=await runtime.writeTextFile(computer,event.path,event.text);
    else if(event.kind==='read') output=await runtime.readTextFile(computer,event.path);
    else if(event.kind==='launch') output=await runtime.launchApp(computer,event.app,event.request);
    else throw new Error(`unknown transition ${event.kind}`);
  }
  logicalTime=946684800000+Math.round((request.time??0)*1000);
  const files=runtime.listFiles(computer);
  const snapshot=runtime.snapshot();
  // Host storage paths are transport details, never semantic state or observations.
  const result={output,files,prompt:runtime.getPrompt(computer),snapshot,trajectory:runtime.trajectory.snapshot()};
  process.stdout.write(JSON.stringify(result).split(root).join('<episode-storage>'));
} finally { await rm(root,{recursive:true,force:true}); }
