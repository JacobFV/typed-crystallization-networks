/** Episode-scoped reproducible identity stream for simulated kernel objects. */
import { createHash } from 'node:crypto';
let seed='0';
let counter=0;
export function setIdentitySeed(value:string) { seed=value; counter=0; }
export function randomUUID(): `${string}-${string}-${string}-${string}-${string}` {
  const hex=createHash('sha256').update(`${seed}:${counter++}`).digest('hex');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-4${hex.slice(13,16)}-8${hex.slice(17,20)}-${hex.slice(20,32)}`;
}
