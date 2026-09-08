import { defineOperatingSystem } from '@tcn-computer/os-core';

export const ubuntuProfile = defineOperatingSystem({
  id: 'ubuntu', productName: 'Ubuntu', release: '26.04 LTS',
  kernel: { family: 'linux', version: '6.18', init: 'systemd', serviceManager: 'systemd' },
  desktop: {
    shell: 'GNOME Shell', windowManager: 'Mutter', compositor: 'Mutter',
    displayServer: 'Wayland', launcher: 'Activities Overview', settingsApp: 'Settings',
  },
  shell: { default: 'bash', executable: '/usr/bin/bash', promptDialect: 'posix', startupFiles: ['/etc/profile', '~/.profile', '~/.bashrc'] },
  filesystem: {
    root: '/', home: '/home/agent', applications: '/opt', userData: '/home/agent/.config', temporary: '/tmp',
    caseSensitive: true, pathSeparator: '/', nativeFormats: ['ext4', 'btrfs', 'xfs'],
  },
  packageManagers: {
    native: ['apt', 'dpkg', 'snap', 'flatpak'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'gem', 'composer', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/var/lib/dpkg', '/var/lib/snapd', '/var/lib/flatpak', '/var/lib/seed/apps'],
  },
  bootServices: [
    { id: 'systemd', executable: '/sbin/init', role: 'init', parent: null, required: true },
    { id: 'gdm', executable: '/usr/sbin/gdm3', role: 'session', parent: 'systemd', required: true },
    { id: 'gnome-shell', executable: '/usr/bin/gnome-shell', role: 'display', parent: 'gdm', required: true },
    { id: 'NetworkManager', executable: '/usr/sbin/NetworkManager', role: 'network', parent: 'systemd', required: true },
    { id: 'pipewire', executable: '/usr/bin/pipewire', role: 'audio', parent: 'systemd', required: false },
    { id: 'udisksd', executable: '/usr/libexec/udisks2/udisksd', role: 'device', parent: 'systemd', required: false },
    { id: 'fwupd', executable: '/usr/libexec/fwupd/fwupd', role: 'updates', parent: 'systemd', required: false },
  ],
  peripherals: [
    { kind: 'display', driver: 'DRM/KMS', hotPluggable: true }, { kind: 'keyboard', driver: 'evdev/libinput', hotPluggable: true },
    { kind: 'pointer', driver: 'libinput', hotPluggable: true }, { kind: 'camera', driver: 'V4L2', hotPluggable: true },
    { kind: 'microphone', driver: 'PipeWire', hotPluggable: true }, { kind: 'speaker', driver: 'PipeWire', hotPluggable: true },
    { kind: 'storage', driver: 'udev/udisks2', hotPluggable: true }, { kind: 'network', driver: 'netlink/NetworkManager', hotPluggable: true },
  ],
  systemAppIds: ['nautilus', 'terminal', 'settings', 'gedit', 'calculator', 'calendar', 'mail', 'app-center', 'system-monitor', 'software-updater', 'document-viewer', 'rhythmbox'],
  conventions: { executableSuffix: '', sharedLibrarySuffix: '.so', environmentPathKey: 'PATH', localhostNames: ['localhost', 'localhost.localdomain'] },
});
