; Inno Setup script for Dollar Detective (Windows installer).
; Packages the PyInstaller onedir (dist\DollarDetective\) into a setup .exe
; with Start-menu + optional desktop shortcuts and an uninstaller.
; Build (in CI): ISCC /DMyAppVersion=1.4.0 installer\DollarDetective.iss

#define MyAppExeName "DollarDetective.exe"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
; Edition tags let a second build (e.g. CUDA) install alongside the default one.
; Defaults are empty, so a plain build is byte-for-byte the same as before.
;   EditionTag  -> filename/dir suffix, e.g. "-cuda"
;   EditionName -> display suffix, e.g. " (CUDA)"
#ifndef EditionTag
  #define EditionTag ""
#endif
#ifndef EditionName
  #define EditionName ""
#endif
#define MyAppName "Dollar Detective" + EditionName

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Paul Barros
DefaultDirName={autopf}\DollarDetective{#EditionTag}
DefaultGroupName={#MyAppName}
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
; Installer's own icon (the app exe already carries the icon via PyInstaller).
SetupIconFile=..\assets\icon.ico
; Paths are relative to this .iss file (installer/), so reach up to the repo root.
OutputDir=..\dist
OutputBaseFilename=DollarDetective-{#MyAppVersion}{#EditionTag}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user install: no admin/UAC prompt, and the install dir is writable.
PrivilegesRequired=lowest
WizardStyle=modern
DisableProgramGroupPage=yes
; Show the "Select Destination Location" page so users can choose the install dir.
DisableDirPage=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[InstallDelete]
; Remove the bundled pattern library BEFORE copying the new one, so pattern
; folders dropped between versions don't linger as orphaned libraries. This is
; what left a stale "Essentials" group on updated installs, and it matters more
; for the single-core flatten (which removes the Nicks and Green Guide folders).
; Bundled patterns are read-only and fully replaced each build; USER patterns
; live in the per-user data dir and are NOT touched by this.
Type: filesandordirs; Name: "{app}\_internal\patterns"
Type: filesandordirs; Name: "{app}\patterns"
; Remove the previously-shipped Green Guide bundle from updated installs (it is no
; longer distributed). Already-imported user copies live in the per-user data dir
; and are not touched.
Type: files; Name: "{app}\Green Guide Library.ddpat"

[Files]
Source: "..\dist\DollarDetective\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
