import {spawnSync} from 'node:child_process';
// Some enclosing development hosts enable Node's private import-watcher IPC.
// It is not part of the test protocol and must not enter Vitest worker channels.
const env={...process.env};delete env.WATCH_REPORT_DEPENDENCIES;
const result=spawnSync(process.execPath,['node_modules/vitest/vitest.mjs','run',...process.argv.slice(2)],{stdio:'inherit',env});
process.exit(result.status??1);
