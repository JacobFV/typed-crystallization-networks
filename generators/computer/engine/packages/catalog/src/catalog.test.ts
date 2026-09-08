import { describe, expect, it } from 'vitest';
import { appCatalog, appsForOS, defaultFileHandler, fileExtension, fileHandlersFor } from './index.js';

describe('file associations are a behavioral contract', () => {
  it('extracts extensions without treating dotfiles as extensions', () => {
    expect(fileExtension('/Users/agent/notes.MD')).toBe('md');
    expect(fileExtension('/Users/agent/archive.tar.gz')).toBe('gz');
    expect(fileExtension('/Users/agent/.zshrc')).toBe('');
    expect(fileExtension('/C/Users/agent/report.PDF')).toBe('pdf');
    expect(fileExtension('/Users/agent/Makefile')).toBe('');
  });

  it('routes a text file to the platform-native editor', () => {
    expect(defaultFileHandler('/Users/agent/a.txt', 'macos')?.id).toBe('textedit');
    expect(defaultFileHandler('/C/Users/agent/a.txt', 'windows')?.id).toBe('notepad');
    expect(defaultFileHandler('/home/agent/a.txt', 'ubuntu')?.id).toBe('gedit');
  });

  it('never returns a handler that does not run on that OS', () => {
    for (const os of ['macos', 'windows', 'ubuntu'] as const) {
      for (const candidate of fileHandlersFor('/tmp/x.png', os)) {
        expect(candidate.supportedOS).toContain(os);
      }
    }
  });

  it('restricts handlers to what is installed when an install set is given', () => {
    const handler = defaultFileHandler('/Users/agent/shot.png', 'macos', ['textedit']);
    expect(handler).toBeUndefined();
    expect(defaultFileHandler('/Users/agent/shot.png', 'macos', ['textedit', 'preview'])?.id).toBe('preview');
  });

  it('prefers a system default over a third-party claimant for the same type', () => {
    const handlers = fileHandlersFor('/Users/agent/photo.jpg', 'macos');
    expect(handlers.length).toBeGreaterThan(1);
    expect(handlers[0]!.system).toBe(true);
  });

  it('returns nothing for a type no application claims', () => {
    expect(defaultFileHandler('/Users/agent/unknown.zzzz', 'macos')).toBeUndefined();
    expect(defaultFileHandler('/Users/agent/noextension', 'macos')).toBeUndefined();
  });

  it('declares associations in lowercase so lookup is case-insensitive', () => {
    for (const manifest of appCatalog) {
      for (const association of manifest.fileAssociations ?? []) {
        expect(association).toBe(association.toLowerCase());
        expect(association.startsWith('.')).toBe(false);
      }
    }
  });

  it('gives every OS a handler for the common document and image types', () => {
    for (const os of ['macos', 'windows', 'ubuntu'] as const) {
      const installed = appsForOS(os).map((manifest) => manifest.id);
      for (const sample of ['notes.txt', 'notes.md']) {
        expect(defaultFileHandler(`/x/${sample}`, os, installed), `${os} has no handler for ${sample}`).toBeDefined();
      }
    }
  });
});
