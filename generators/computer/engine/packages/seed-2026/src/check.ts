import { appCatalog } from '@tcn-computer/catalog';
import { applicationSurfaces } from '@tcn-computer/ui-surfaces';
import { assertSeed2026Blueprint, seed2026Blueprint } from './index.js';

assertSeed2026Blueprint();
console.log(JSON.stringify({
  ecosystem: seed2026Blueprint.id,
  computers: seed2026Blueprint.computers.length,
  operatingSystems: Object.keys(seed2026Blueprint.operatingSystems),
  applications: appCatalog.length,
  surfaces: applicationSurfaces.length,
  services: seed2026Blueprint.services.length,
  status: 'valid',
}, null, 2));
