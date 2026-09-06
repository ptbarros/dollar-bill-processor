; Inno Setup script for the standalone Dollar Detective Crop tool (Windows installer).
; Packages the PyInstaller onedir (dist\DollarDetectiveCrop\) into a setup .exe
; with Start-menu + optional desktop shortcuts and an uninstaller.
; Build (in CI): ISCC /DMyAppVersion=1.5.1 installer\DollarDetectiveCrop.iss

#define MyAppExeName "DollarDetectiveCrop.exe"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Dollar Detective Crop"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Paul Barros
DefaultDirName={autopf}\DollarDetectiveCrop
DefaultGroupName={#MyAppName}
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\assets\DD-Crop.ico
; Paths are relative to this .iss file (installer/), so reach up to the repo root.
OutputDir=..\dist
OutputBaseFilename=DollarDetectiveCrop-{#MyAppVersion}-setup
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
Source: "..\dist\DollarDetectiveCrop\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
