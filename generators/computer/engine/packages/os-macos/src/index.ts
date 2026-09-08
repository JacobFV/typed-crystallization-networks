import { defineOperatingSystem } from '@tcn-computer/os-core';

export const macOSProfile = defineOperatingSystem({
  id: 'macos',
  productName: 'macOS',
  release: '26',
  kernel: { family: 'xnu', version: '25.0', init: 'launchd', serviceManager: 'launchd' },
  desktop: {
    shell: 'Finder', windowManager: 'WindowServer', compositor: 'Quartz Compositor',
    displayServer: 'CoreGraphics', launcher: 'Dock + Launchpad', settingsApp: 'System Settings',
  },
  shell: { default: 'zsh', executable: '/bin/zsh', promptDialect: 'posix', startupFiles: ['/etc/zprofile', '~/.zprofile', '~/.zshrc'] },
  filesystem: {
    root: '/', home: '/Users/agent', applications: '/Applications', userData: '/Users/agent/Library/Application Support',
    temporary: '/private/tmp', caseSensitive: false, pathSeparator: '/', nativeFormats: ['APFS', 'HFS+'],
  },
  packageManagers: {
    native: ['mas', 'brew'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'gem', 'composer', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/var/db/receipts', '/opt/homebrew/Cellar', '/Applications'],
  },
  bootServices: [
    { id: 'launchd', executable: '/sbin/launchd', role: 'init', parent: null, required: true },
    { id: 'WindowServer', executable: '/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer', role: 'display', parent: 'launchd', required: true },
    { id: 'loginwindow', executable: '/System/Library/CoreServices/loginwindow.app', role: 'session', parent: 'launchd', required: true },
    { id: 'configd', executable: '/usr/libexec/configd', role: 'network', parent: 'launchd', required: true },
    { id: 'coreaudiod', executable: '/usr/sbin/coreaudiod', role: 'audio', parent: 'launchd', required: false },
    { id: 'tccd', executable: '/System/Library/PrivateFrameworks/TCC.framework/Support/tccd', role: 'security', parent: 'launchd', required: true },
    { id: 'mds', executable: '/System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Support/mds', role: 'indexing', parent: 'launchd', required: false },
  ],
  peripherals: [
    { kind: 'display', driver: 'CoreDisplay', hotPluggable: true }, { kind: 'keyboard', driver: 'IOHIDFamily', hotPluggable: true },
    { kind: 'pointer', driver: 'IOHIDFamily', hotPluggable: true }, { kind: 'camera', driver: 'CoreMediaIO', hotPluggable: true },
    { kind: 'microphone', driver: 'CoreAudio', hotPluggable: true }, { kind: 'speaker', driver: 'CoreAudio', hotPluggable: true },
    { kind: 'storage', driver: 'IOStorageFamily', hotPluggable: true }, { kind: 'network', driver: 'IONetworkingFamily', hotPluggable: true },
  ],
  systemAppIds: ['finder', 'terminal', 'settings', 'textedit', 'preview', 'photos', 'calculator', 'calendar', 'mail', 'app-store', 'safari', 'messages', 'facetime', 'music', 'maps', 'reminders'],
  conventions: { executableSuffix: '', sharedLibrarySuffix: '.dylib', environmentPathKey: 'PATH', localhostNames: ['localhost', 'localhost.local'] },
});
