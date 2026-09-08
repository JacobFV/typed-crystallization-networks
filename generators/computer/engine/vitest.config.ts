import { defineConfig } from 'vitest/config';
export default defineConfig({test:{pool:'threads',maxWorkers:2,minWorkers:1,include:['packages/**/*.test.ts','tests/**/*.test.ts'],testTimeout:30000}});
