import { defineOperatingSystem } from '@tcn-computer/os-core';

export const windowsProfile = defineOperatingSystem({
  id: 'windows', productName: 'Windows 11', release: '26H2',
  kernel: { family: 'nt', version: '10.0.28000', init: 'smss.exe', serviceManager: 'services.exe' },
  desktop: {
    shell: 'explorer.exe', windowManager: 'Desktop Window Manager', compositor: 'dwm.exe',
    displayServer: 'Win32k / DirectComposition', launcher: 'Start', settingsApp: 'Settings',
  },
  shell: { default: 'powershell', executable: 'C:\\Program Files\\PowerShell\\7\\pwsh.exe', promptDialect: 'powershell', startupFiles: ['$PROFILE.AllUsersAllHosts', '$PROFILE.CurrentUserAllHosts'] },
  filesystem: {
    root: '/C', home: '/C/Users/agent', applications: '/C/Program Files', userData: '/C/Users/agent/AppData/Roaming',
    temporary: '/C/Users/agent/AppData/Local/Temp', caseSensitive: false, pathSeparator: '\\', nativeFormats: ['NTFS', 'ReFS', 'FAT32'],
  },
  packageManagers: {
    native: ['winget', 'choco', 'scoop'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'gem', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/C/ProgramData/Seed/AppRepository', '/C/ProgramData/chocolatey', '/C/Users/agent/scoop/apps'],
  },
  bootServices: [
    { id: 'smss', executable: 'C:\\Windows\\System32\\smss.exe', role: 'init', parent: null, required: true },
    { id: 'services', executable: 'C:\\Windows\\System32\\services.exe', role: 'init', parent: 'smss', required: true },
    { id: 'winlogon', executable: 'C:\\Windows\\System32\\winlogon.exe', role: 'session', parent: 'smss', required: true },
    { id: 'dwm', executable: 'C:\\Windows\\System32\\dwm.exe', role: 'display', parent: 'winlogon', required: true },
    { id: 'explorer', executable: 'C:\\Windows\\explorer.exe', role: 'session', parent: 'winlogon', required: true },
    { id: 'NlaSvc', executable: 'C:\\Windows\\System32\\svchost.exe -k NetworkService', role: 'network', parent: 'services', required: true },
    { id: 'Audiosrv', executable: 'C:\\Windows\\System32\\svchost.exe -k LocalServiceNetworkRestricted', role: 'audio', parent: 'services', required: false },
    { id: 'WinDefend', executable: 'C:\\ProgramData\\Microsoft\\Windows Defender\\MsMpEng.exe', role: 'security', parent: 'services', required: true },
  ],
  peripherals: [
    { kind: 'display', driver: 'WDDM', hotPluggable: true }, { kind: 'keyboard', driver: 'kbdclass.sys', hotPluggable: true },
    { kind: 'pointer', driver: 'mouclass.sys', hotPluggable: true }, { kind: 'camera', driver: 'AVStream', hotPluggable: true },
    { kind: 'microphone', driver: 'WASAPI', hotPluggable: true }, { kind: 'speaker', driver: 'WASAPI', hotPluggable: true },
    { kind: 'storage', driver: 'storport.sys', hotPluggable: true }, { kind: 'network', driver: 'NDIS', hotPluggable: true },
  ],
  systemAppIds: ['explorer', 'terminal', 'settings', 'notepad', 'photos', 'calculator', 'calendar', 'store', 'edge', 'paint', 'snipping-tool', 'task-manager', 'outlook'],
  conventions: { executableSuffix: '.exe', sharedLibrarySuffix: '.dll', environmentPathKey: 'Path', localhostNames: ['localhost'] },
});
