; Inno Setup script for Dollar Bill Processor (Windows installer).
; Packages the PyInstaller onedir (dist\DollarBillProcessor\) into a setup .exe
; with Start-menu + optional desktop shortcuts and an uninstaller.
; Build (in CI): ISCC /DMyAppVersion=1.4.0 installer\DollarBillProcessor.iss

#define MyAppExeName "DollarBillProcessor.exe"
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
#define MyAppName "Dollar Bill Processor" + EditionName

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Paul Barros
DefaultDirName={autopf}\DollarBillProcessor{#EditionTag}
DefaultGroupName={#MyAppName}
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
; Installer's own icon (the app exe already carries the icon via PyInstaller).
SetupIconFile=..\assets\icon.ico
; Paths are relative to this .iss file (installer/), so reach up to the repo root.
OutputDir=..\dist
OutputBaseFilename=DollarBillProcessor-{#MyAppVersion}{#EditionTag}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user install: no admin/UAC prompt, and the install dir is writable.
PrivilegesRequired=lowest
WizardStyle=modern
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\DollarBillProcessor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
