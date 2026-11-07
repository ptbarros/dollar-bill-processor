// dd_twain_helper -- Dollar Detective's TWAIN bridge.
//
// TWAIN data sources load in-process and Epson ships only a 32-bit one, while
// the app is 64-bit, so the app runs this small helper instead. It drives the
// scanner with no driver UI, writes each finished page to a folder, and prints
// one JSON object per line on stdout (protocol: dollar_detective/scanner/
// twain_backend.py).
//
//   dd_twain_helper list
//   dd_twain_helper caps --source "EPSON FF-640"
//   dd_twain_helper scan --source NAME --out DIR [--dpi 300] [--duplex 1]
//                        [--color 1] [--crop 1] [--deskew 1] [--quality 92]
//   Stop a scan by creating DIR\STOP.
//
// Built from the probe rig used on Paul's laptop (2026-11-07, Epson Scan 2
// 6.4.8 + 6.7.81). Lessons baked in:
//  * GET before SET: setting ICAP_AUTOMATICBORDERDETECTION to its current
//    value fails on 6.7.81 and then ENABLEDS refuses until the DS is reopened.
//  * Never call DAT_IMAGELAYOUT GET (hangs on 6.7.81 once caps are set).
//  * Don't sit in state 5 waiting for paper when CAP_FEEDERLOADED works: poll
//    it in state 4 and only ENABLEDS when paper is there.
//  * Always DISABLEDS/CLOSEDS cleanly; a killed process can wedge the DS.
//
// Build (no install needed on Windows):
//   %WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe /platform:x86
//       /target:exe /r:System.Windows.Forms.dll /out:dd_twain_helper.exe TwainHelper.cs

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[StructLayout(LayoutKind.Sequential, Pack = 2, CharSet = CharSet.Ansi)]
public struct TwVersion {
    public ushort MajorNum, MinorNum, Language, Country;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 34)] public string Info;
}

// On Windows Id stays 32-bit even in 64-bit processes (TW_IDENTITY_LEGACY).
[StructLayout(LayoutKind.Sequential, Pack = 2, CharSet = CharSet.Ansi)]
public class TwIdentity {
    public uint Id;
    public TwVersion Version;
    public ushort ProtocolMajor, ProtocolMinor;
    public uint SupportedGroups;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 34)] public string Manufacturer;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 34)] public string ProductFamily;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 34)] public string ProductName;
}

[StructLayout(LayoutKind.Sequential, Pack = 2)]
public struct TwCapability { public ushort Cap; public ushort ConType; public IntPtr hContainer; }

[StructLayout(LayoutKind.Sequential, Pack = 2)]
public struct TwStatus { public ushort ConditionCode; public ushort Data; }

static class Native {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibrary(string name);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
    public static extern IntPtr GetProcAddress(IntPtr mod, string name);
    [DllImport("kernel32.dll")] public static extern IntPtr GlobalAlloc(uint flags, UIntPtr bytes);
    [DllImport("kernel32.dll")] public static extern IntPtr GlobalLock(IntPtr h);
    [DllImport("kernel32.dll")] public static extern bool GlobalUnlock(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr GlobalFree(IntPtr h);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern uint GetShortPathName(string longPath, StringBuilder shortPath, uint size);
    [DllImport("user32.dll")] public static extern bool PeekMessage(IntPtr msg, IntPtr hwnd, uint min, uint max, uint remove);
    [DllImport("user32.dll")] public static extern bool TranslateMessage(IntPtr msg);
    [DllImport("user32.dll")] public static extern IntPtr DispatchMessage(IntPtr msg);
}

// DSM_Entry through a function pointer, so we can fall back from TWAINDSM.dll
// (TWAIN 2.x) to the old twain_32.dll (1.x) when the former isn't installed.
[UnmanagedFunctionPointer(CallingConvention.Winapi)]
delegate ushort EntryIdent([In, Out] TwIdentity origin, IntPtr dest, uint dg, ushort dat, ushort msg, [In, Out] TwIdentity data);
[UnmanagedFunctionPointer(CallingConvention.Winapi)]
delegate ushort EntryParent([In, Out] TwIdentity origin, IntPtr dest, uint dg, ushort dat, ushort msg, ref IntPtr hwnd);
[UnmanagedFunctionPointer(CallingConvention.Winapi)]
delegate ushort EntryCap([In, Out] TwIdentity origin, [In, Out] TwIdentity dest, uint dg, ushort dat, ushort msg, ref TwCapability cap);
[UnmanagedFunctionPointer(CallingConvention.Winapi)]
delegate ushort EntryStatus([In, Out] TwIdentity origin, [In, Out] TwIdentity dest, uint dg, ushort dat, ushort msg, ref TwStatus st);
[UnmanagedFunctionPointer(CallingConvention.Winapi)]
delegate ushort EntryPtr([In, Out] TwIdentity origin, [In, Out] TwIdentity dest, uint dg, ushort dat, ushort msg, IntPtr data);

static class Tw {
    // Return codes
    public const ushort RC_SUCCESS = 0, RC_FAILURE = 1, RC_CHECKSTATUS = 2, RC_DSEVENT = 4,
                        RC_NOTDSEVENT = 5, RC_XFERDONE = 6, RC_ENDOFLIST = 7;
    // Messages
    const ushort MSG_GET = 0x0001, MSG_GETCURRENT = 0x0002, MSG_SET = 0x0006,
                 MSG_QUERYSUPPORT = 0x0008, MSG_OPENDSM = 0x0301, MSG_CLOSEDSM = 0x0302,
                 MSG_OPENDS = 0x0401, MSG_CLOSEDS = 0x0402, MSG_GETFIRST = 0x0004,
                 MSG_GETNEXT = 0x0005, MSG_DISABLEDS = 0x0501, MSG_ENABLEDS = 0x0502,
                 MSG_PROCESSEVENT = 0x0601, MSG_ENDXFER = 0x0701, MSG_RESET = 0x0007;
    public const ushort MSG_XFERREADY = 0x0101, MSG_CLOSEDSREQ = 0x0102, MSG_CLOSEDSOK = 0x0103;
    // Item types
    public const ushort TWTY_INT16 = 1, TWTY_UINT16 = 4, TWTY_BOOL = 6, TWTY_FIX32 = 7;

    static EntryIdent dIdent; static EntryParent dParent; static EntryCap dCap;
    static EntryStatus dStatus; static EntryPtr dPtr;

    public static TwIdentity App;
    public static TwIdentity Src;
    public static IntPtr Hwnd;
    public static string DsmName;

    public static void LoadDsm() {
        IntPtr mod = Native.LoadLibrary("TWAINDSM.dll");
        DsmName = "TWAINDSM.dll";
        if (mod == IntPtr.Zero && IntPtr.Size == 4) { mod = Native.LoadLibrary("twain_32.dll"); DsmName = "twain_32.dll"; }
        if (mod == IntPtr.Zero) throw new Exception("No TWAIN data source manager (TWAINDSM.dll) found");
        IntPtr f = Native.GetProcAddress(mod, "DSM_Entry");
        if (f == IntPtr.Zero) throw new Exception("DSM_Entry not found in " + DsmName);
        dIdent = (EntryIdent)Marshal.GetDelegateForFunctionPointer(f, typeof(EntryIdent));
        dParent = (EntryParent)Marshal.GetDelegateForFunctionPointer(f, typeof(EntryParent));
        dCap = (EntryCap)Marshal.GetDelegateForFunctionPointer(f, typeof(EntryCap));
        dStatus = (EntryStatus)Marshal.GetDelegateForFunctionPointer(f, typeof(EntryStatus));
        dPtr = (EntryPtr)Marshal.GetDelegateForFunctionPointer(f, typeof(EntryPtr));
    }

    public static void OpenDsm(IntPtr hwnd) {
        bool app2 = DsmName == "TWAINDSM.dll";
        Hwnd = hwnd;
        App = new TwIdentity();
        App.Version.MajorNum = 1; App.Version.Info = "Dollar Detective"; App.Version.Language = 13; App.Version.Country = 1;
        App.ProtocolMajor = (ushort)(app2 ? 2 : 1); App.ProtocolMinor = (ushort)(app2 ? 4 : 9);
        App.SupportedGroups = 0x3u | (app2 ? 0x10000000u : 0u);   // DG_CONTROL|DG_IMAGE (+DF_APP2)
        App.Manufacturer = "Dollar Detective"; App.ProductFamily = "Dollar Detective";
        App.ProductName = "dd_twain_helper";
        ushort rc = dParent(App, IntPtr.Zero, 1, 4, MSG_OPENDSM, ref Hwnd);
        if (rc != 0) throw new Exception("MSG_OPENDSM rc=" + rc);
    }

    public static List<TwIdentity> Sources() {
        var l = new List<TwIdentity>();
        var id = new TwIdentity();
        ushort rc = dIdent(App, IntPtr.Zero, 1, 3, MSG_GETFIRST, id);
        while (rc == 0) { l.Add(id); id = new TwIdentity(); rc = dIdent(App, IntPtr.Zero, 1, 3, MSG_GETNEXT, id); }
        return l;
    }

    public static ushort OpenDs(TwIdentity src) { Src = src; return dIdent(App, IntPtr.Zero, 1, 3, MSG_OPENDS, src); }
    public static ushort CloseDs() { return dIdent(App, IntPtr.Zero, 1, 3, MSG_CLOSEDS, Src); }
    public static ushort CloseDsm() { return dParent(App, IntPtr.Zero, 1, 4, MSG_CLOSEDSM, ref Hwnd); }
    public static ushort CondCode() { var s = new TwStatus(); dStatus(App, Src, 1, 8, 0x0001, ref s); return s.ConditionCode; }

    static readonly int[] Sizes = { 1, 2, 4, 1, 2, 4, 2, 4, 16, 34, 66, 130, 256, 0, 0, 4 };
    public static int ItemSize(int t) { return (t >= 0 && t < Sizes.Length) ? Sizes[t] : 4; }
    static double Fix(IntPtr p) { return Marshal.ReadInt16(p) + (ushort)Marshal.ReadInt16(p + 2) / 65536.0; }

    static double Num(IntPtr p, int t) {
        switch (t) {
            case 0: return (sbyte)Marshal.ReadByte(p);
            case 1: return Marshal.ReadInt16(p);
            case 2: return Marshal.ReadInt32(p);
            case 3: return Marshal.ReadByte(p);
            case 4: return (ushort)Marshal.ReadInt16(p);
            case 5: return (uint)Marshal.ReadInt32(p);
            case 6: return Marshal.ReadInt16(p) != 0 ? 1 : 0;
            case 7: return Fix(p);
            default: return double.NaN;
        }
    }

    static string Item(IntPtr p, int t) {
        if (t >= 9 && t <= 12) return Json.Str(Marshal.PtrToStringAnsi(p));
        if (t == 8) return string.Format("[{0},{1},{2},{3}]", Json.Num(Fix(p)), Json.Num(Fix(p + 4)), Json.Num(Fix(p + 8)), Json.Num(Fix(p + 12)));
        double v = Num(p, t);
        if (t == 6) return v != 0 ? "true" : "false";
        return double.IsNaN(v) ? "null" : Json.Num(v);
    }

    // Current value of a capability as a number (bool = 0/1), or null.
    public static double? GetCurrent(ushort cap) {
        var c = new TwCapability(); c.Cap = cap; c.ConType = 0xFFFF;
        ushort rc = dCap(App, Src, 1, 1, MSG_GETCURRENT, ref c);
        if (rc != 0) { if (c.hContainer != IntPtr.Zero) Native.GlobalFree(c.hContainer); return null; }
        double? r = null;
        IntPtr p = Native.GlobalLock(c.hContainer);
        try {
            int t = (ushort)Marshal.ReadInt16(p);
            if (c.ConType == 5) r = Num(p + 2, t);                                  // ONEVALUE
            else if (c.ConType == 4) {                                               // ENUMERATION
                int cur = Marshal.ReadInt32(p + 6);
                r = Num(p + 14 + cur * ItemSize(t), t);
            } else if (c.ConType == 6) r = Num(p + 18, t);                          // RANGE
        } finally { Native.GlobalUnlock(c.hContainer); Native.GlobalFree(c.hContainer); }
        if (r.HasValue && double.IsNaN(r.Value)) return null;
        return r;
    }

    // Full GET as JSON (for the caps dump).
    public static string GetJson(ushort cap) {
        var c = new TwCapability(); c.Cap = cap; c.ConType = 0xFFFF;
        ushort rc = dCap(App, Src, 1, 1, MSG_GET, ref c);
        if (rc != 0) {
            ushort cc = CondCode();
            if (c.hContainer != IntPtr.Zero) Native.GlobalFree(c.hContainer);
            return "{\"rc\":" + rc + ",\"cc\":" + cc + "}";
        }
        var sb = new StringBuilder();
        IntPtr p = Native.GlobalLock(c.hContainer);
        try {
            int t = (ushort)Marshal.ReadInt16(p);
            int sz = ItemSize(t);
            switch (c.ConType) {
                case 5: sb.AppendFormat("{{\"type\":{0},\"one\":{1}}}", t, Item(p + 2, t)); break;
                case 6: sb.AppendFormat("{{\"type\":{0},\"min\":{1},\"max\":{2},\"step\":{3},\"default\":{4},\"current\":{5}}}",
                                        t, Item(p + 2, t), Item(p + 6, t), Item(p + 10, t), Item(p + 14, t), Item(p + 18, t)); break;
                case 3: {
                    int n = Marshal.ReadInt32(p + 2);
                    sb.AppendFormat("{{\"type\":{0},\"array\":[", t);
                    for (int i = 0; i < n && i < 300; i++) { if (i > 0) sb.Append(','); sb.Append(Item(p + 6 + i * sz, t)); }
                    sb.Append("]}"); break; }
                case 4: {
                    int n = Marshal.ReadInt32(p + 2), cur = Marshal.ReadInt32(p + 6), def = Marshal.ReadInt32(p + 10);
                    sb.AppendFormat("{{\"type\":{0},\"current_index\":{1},\"default_index\":{2},\"enum\":[", t, cur, def);
                    for (int i = 0; i < n && i < 300; i++) { if (i > 0) sb.Append(','); sb.Append(Item(p + 14 + i * sz, t)); }
                    sb.Append("]}"); break; }
                default: sb.AppendFormat("{{\"contype\":{0}}}", c.ConType); break;
            }
        } finally { Native.GlobalUnlock(c.hContainer); Native.GlobalFree(c.hContainer); }
        return sb.ToString();
    }

    public static List<int> SupportedCaps() {
        var r = new List<int>();
        var c = new TwCapability(); c.Cap = 0x1005; c.ConType = 0xFFFF;   // CAP_SUPPORTEDCAPS
        if (dCap(App, Src, 1, 1, MSG_GET, ref c) != 0) return r;
        IntPtr p = Native.GlobalLock(c.hContainer);
        try {
            int t = (ushort)Marshal.ReadInt16(p);
            int off = c.ConType == 3 ? 6 : (c.ConType == 4 ? 14 : -1);
            if (off > 0) {
                int n = Marshal.ReadInt32(p + 2);
                for (int i = 0; i < n; i++) r.Add((ushort)Marshal.ReadInt16(p + off + i * ItemSize(t)));
            }
        } finally { Native.GlobalUnlock(c.hContainer); Native.GlobalFree(c.hContainer); }
        return r;
    }

    static string SetOne(ushort cap, ushort itemType, double value) {
        IntPtr h = Native.GlobalAlloc(0x42, (UIntPtr)8);
        IntPtr p = Native.GlobalLock(h);
        Marshal.WriteInt16(p, (short)itemType);
        if (itemType == TWTY_FIX32) {
            int whole = (int)Math.Floor(value); int frac = (int)Math.Round((value - whole) * 65536.0);
            Marshal.WriteInt16(p + 2, (short)whole); Marshal.WriteInt16(p + 4, (short)(ushort)frac);
        } else if (ItemSize(itemType) == 2) Marshal.WriteInt16(p + 2, (short)(int)value);
        else Marshal.WriteInt32(p + 2, (int)value);
        Native.GlobalUnlock(h);
        var c = new TwCapability(); c.Cap = cap; c.ConType = 5; c.hContainer = h;
        ushort rc = dCap(App, Src, 1, 1, MSG_SET, ref c);
        string r = rc == 0 ? "ok" : (rc == RC_CHECKSTATUS ? "checkstatus" : "rc=" + rc + " cc=" + CondCode());
        Native.GlobalFree(h);
        return r;
    }

    // GET before SET: only set when the value differs (see header).
    public static string Ensure(ushort cap, ushort itemType, double value) {
        double? cur = GetCurrent(cap);
        if (cur.HasValue && Math.Abs(cur.Value - value) < 1e-6) return "unchanged";
        return SetOne(cap, itemType, value);
    }

    // TW_USERINTERFACE {BOOL ShowUI; BOOL ModalUI; TW_HANDLE hParent} -- pack 2.
    static IntPtr NewUi() {
        IntPtr ui = Marshal.AllocHGlobal(4 + IntPtr.Size);
        Marshal.WriteInt16(ui, 0); Marshal.WriteInt16(ui + 2, 0);
        Marshal.WriteIntPtr(ui + 4, Hwnd);
        return ui;
    }

    public static ushort Enable(out IntPtr ui) {
        ui = NewUi();
        return dPtr(App, Src, 1, 0x0009, MSG_ENABLEDS, ui);
    }
    public static ushort Disable(IntPtr ui) {
        ushort rc = dPtr(App, Src, 1, 0x0009, MSG_DISABLEDS, ui);
        Marshal.FreeHGlobal(ui);
        return rc;
    }

    // TW_EVENT {TW_MEMREF pEvent; TW_UINT16 TWMessage} -- pack 2.
    public static ushort ProcessEvent(IntPtr winMsg, IntPtr ev, out ushort twMsg) {
        Marshal.WriteIntPtr(ev, winMsg);
        Marshal.WriteInt16(ev + IntPtr.Size, 0);
        ushort rc = dPtr(App, Src, 1, 0x0002, MSG_PROCESSEVENT, ev);
        twMsg = (ushort)Marshal.ReadInt16(ev + IntPtr.Size);
        return rc;
    }

    // One file transfer: TW_SETUPFILEXFER {char FileName[256]; u16 Format; i16 VRefNum}.
    public static ushort FileXfer(string path, ushort fmt) {
        IntPtr sf = Marshal.AllocHGlobal(260);
        for (int i = 0; i < 260; i++) Marshal.WriteByte(sf + i, 0);
        byte[] nb = Encoding.Default.GetBytes(path);
        Marshal.Copy(nb, 0, sf, Math.Min(nb.Length, 255));
        Marshal.WriteInt16(sf + 256, (short)fmt);
        dPtr(App, Src, 1, 0x0007, MSG_SET, sf);
        Marshal.FreeHGlobal(sf);
        return dPtr(App, Src, 2, 0x0105, MSG_GET, IntPtr.Zero);       // DAT_IMAGEFILEXFER
    }

    // TW_PENDINGXFERS {u16 Count; u32 EOJ}. Returns remaining count.
    public static int EndXfer() {
        IntPtr pend = Marshal.AllocHGlobal(8);
        dPtr(App, Src, 1, 0x0005, MSG_ENDXFER, pend);
        int count = Marshal.ReadInt16(pend);
        Marshal.FreeHGlobal(pend);
        return count;
    }
    public static void ResetXfers() {
        IntPtr pend = Marshal.AllocHGlobal(8);
        dPtr(App, Src, 1, 0x0005, MSG_RESET, pend);
        Marshal.FreeHGlobal(pend);
    }
}

static class Json {
    public static string Str(string s) {
        if (s == null) return "null";
        var sb = new StringBuilder("\"");
        foreach (char ch in s) {
            if (ch == '"' || ch == '\\') sb.Append('\\').Append(ch);
            else if (ch < 0x20) sb.AppendFormat("\\u{0:x4}", (int)ch);
            else sb.Append(ch);
        }
        return sb.Append('"').ToString();
    }
    public static string Num(double v) {
        return v.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
    }
    public static void Emit(string body) {
        Console.Out.WriteLine("{" + body + "}");
        Console.Out.Flush();
    }
}

static class Program {
    // Capability ids
    const ushort CAP_XFERCOUNT = 0x0001, ICAP_PIXELTYPE = 0x0101, ICAP_XFERMECH = 0x0103,
                 CAP_FEEDERENABLED = 0x1002, CAP_FEEDERLOADED = 0x1003, CAP_DUPLEXENABLED = 0x1013,
                 ICAP_IMAGEFILEFORMAT = 0x110C, ICAP_XRESOLUTION = 0x1118, ICAP_YRESOLUTION = 0x1119,
                 ICAP_AUTOMATICBORDERDETECTION = 0x1150, ICAP_AUTOMATICDESKEW = 0x1151,
                 ICAP_JPEGQUALITY = 0x1153;

    static Stopwatch Clock = Stopwatch.StartNew();
    static string T() { return Json.Num(Clock.Elapsed.TotalSeconds); }

    static Dictionary<string, string> Args(string[] a) {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 1; i < a.Length; i++)
            if (a[i].StartsWith("--") && i + 1 < a.Length) { d[a[i].Substring(2)] = a[i + 1]; i++; }
        return d;
    }
    static string Opt(Dictionary<string, string> d, string k, string def) { string v; return d.TryGetValue(k, out v) ? v : def; }

    [STAThread]
    static int Main(string[] argv) {
        Console.OutputEncoding = Encoding.UTF8;
        if (argv.Length == 0) { Console.Error.WriteLine("usage: dd_twain_helper list|caps|scan ..."); return 2; }
        var form = new Form();                 // hidden window: TWAIN parent + message queue
        IntPtr hwnd = form.Handle;
        try {
            Tw.LoadDsm();
            Tw.OpenDsm(hwnd);
        } catch (Exception e) {
            Json.Emit("\"event\":\"error\",\"rc\":-1,\"cc\":0,\"msg\":" + Json.Str(e.Message));
            Json.Emit("\"event\":\"end\"");
            return 3;
        }
        int code = 0;
        try {
            var a = Args(argv);
            switch (argv[0].ToLowerInvariant()) {
                case "list": code = List(); break;
                case "caps": code = Caps(Opt(a, "source", "")); break;
                case "scan": code = Scan(a); break;
                default: Console.Error.WriteLine("unknown command " + argv[0]); code = 2; break;
            }
        } catch (Exception e) {
            Json.Emit("\"event\":\"error\",\"rc\":-1,\"cc\":0,\"msg\":" + Json.Str(e.ToString()));
            code = 4;
        } finally {
            Tw.CloseDsm();
            Json.Emit("\"event\":\"end\"");
        }
        return code;
    }

    static TwIdentity FindSource(string name) {
        foreach (var s in Tw.Sources())
            if (string.Equals(s.ProductName, name, StringComparison.OrdinalIgnoreCase)) return s;
        return null;
    }

    static int List() {
        foreach (var s in Tw.Sources())
            Json.Emit(string.Format("\"event\":\"source\",\"name\":{0},\"manufacturer\":{1},\"family\":{2},\"protocol\":\"{3}.{4}\",\"dsm\":{5},\"bits\":{6}",
                Json.Str(s.ProductName), Json.Str(s.Manufacturer), Json.Str(s.ProductFamily),
                s.ProtocolMajor, s.ProtocolMinor, Json.Str(Tw.DsmName), IntPtr.Size * 8));
        return 0;
    }

    static int Caps(string name) {
        var src = FindSource(name);
        if (src == null) { Json.Emit("\"event\":\"error\",\"rc\":-1,\"cc\":0,\"msg\":" + Json.Str("source not found: " + name)); return 5; }
        ushort rc = Tw.OpenDs(src);
        if (rc != 0) { Json.Emit("\"event\":\"error\",\"rc\":" + rc + ",\"cc\":" + Tw.CondCode() + ",\"msg\":\"MSG_OPENDS failed\""); return 6; }
        try {
            var sb = new StringBuilder();
            bool first = true;
            foreach (int cap in Tw.SupportedCaps()) {
                if (cap == 0x1005) continue;
                if (!first) sb.Append(',');
                first = false;
                sb.AppendFormat("\"0x{0:X4}\":{1}", cap, Tw.GetJson((ushort)cap));
            }
            Json.Emit("\"event\":\"caps\",\"source\":" + Json.Str(name) + ",\"caps\":{" + sb + "}");
        } finally { Tw.CloseDs(); }
        return 0;
    }

    static bool StopRequested(string outDir) { return File.Exists(Path.Combine(outDir, "STOP")); }

    static int Scan(Dictionary<string, string> a) {
        string name = Opt(a, "source", "");
        string outDir = Opt(a, "out", "");
        if (outDir == "" || !Directory.Exists(outDir)) throw new Exception("--out folder missing");
        // The driver takes the file name as an ANSI char[256]; a user name with
        // non-ANSI characters in the temp path would break it. The 8.3 short
        // form of the folder is plain ASCII.
        var shortBuf = new StringBuilder(1024);
        if (Native.GetShortPathName(outDir, shortBuf, (uint)shortBuf.Capacity) > 0) outDir = shortBuf.ToString();
        int dpi = int.Parse(Opt(a, "dpi", "300"));
        bool duplex = Opt(a, "duplex", "1") == "1";
        bool color = Opt(a, "color", "1") == "1";
        bool crop = Opt(a, "crop", "1") == "1";
        bool deskew = Opt(a, "deskew", "1") == "1";
        int quality = int.Parse(Opt(a, "quality", "92"));

        var src = FindSource(name);
        if (src == null) { Json.Emit("\"event\":\"error\",\"rc\":-1,\"cc\":0,\"msg\":" + Json.Str("source not found: " + name)); return 5; }
        ushort orc = Tw.OpenDs(src);
        if (orc != 0) { Json.Emit("\"event\":\"error\",\"rc\":" + orc + ",\"cc\":" + Tw.CondCode() + ",\"msg\":\"MSG_OPENDS failed (scanner busy or in use by another program?)\""); return 6; }
        try {
            var supported = new HashSet<int>(Tw.SupportedCaps());
            Func<ushort, bool> has = c => supported.Count == 0 || supported.Contains(c);
            var cfg = new StringBuilder();
            Action<string, ushort, ushort, double> ensure = (label, cap, type, val) => {
                string r = has(cap) ? Tw.Ensure(cap, type, val) : "unsupported";
                cfg.AppendFormat(",\"{0}\":{1}", label, Json.Str(r));
            };
            ensure("xfermech", ICAP_XFERMECH, Tw.TWTY_UINT16, 1);                 // TWSX_FILE
            ensure("fileformat", ICAP_IMAGEFILEFORMAT, Tw.TWTY_UINT16, 4);        // TWFF_JFIF
            ensure("pixeltype", ICAP_PIXELTYPE, Tw.TWTY_UINT16, color ? 2 : 1);   // RGB / GRAY
            ensure("xres", ICAP_XRESOLUTION, Tw.TWTY_FIX32, dpi);
            ensure("yres", ICAP_YRESOLUTION, Tw.TWTY_FIX32, dpi);
            ensure("feeder", CAP_FEEDERENABLED, Tw.TWTY_BOOL, 1);
            ensure("duplex", CAP_DUPLEXENABLED, Tw.TWTY_BOOL, duplex ? 1 : 0);
            ensure("xfercount", CAP_XFERCOUNT, Tw.TWTY_INT16, -1);
            ensure("jpegquality", ICAP_JPEGQUALITY, Tw.TWTY_UINT16, quality);
            ensure("border_detect_set", ICAP_AUTOMATICBORDERDETECTION, Tw.TWTY_BOOL, crop ? 1 : 0);
            double? bd = has(ICAP_AUTOMATICBORDERDETECTION) ? Tw.GetCurrent(ICAP_AUTOMATICBORDERDETECTION) : null;
            ensure("deskew", ICAP_AUTOMATICDESKEW, Tw.TWTY_BOOL, deskew ? 1 : 0);
            bool feederSensor = has(CAP_FEEDERLOADED) && Tw.GetCurrent(CAP_FEEDERLOADED).HasValue;
            Json.Emit("\"event\":\"config\",\"source\":" + Json.Str(name) + ",\"dsm\":" + Json.Str(Tw.DsmName) +
                      ",\"border_detect\":" + (bd.HasValue && bd.Value != 0 ? "true" : "false") +
                      ",\"feeder_sensor\":" + (feederSensor ? "true" : "false") + cfg);

            int total = 0;
            bool waitingSaid = false;
            IntPtr msg = Marshal.AllocHGlobal(64);
            IntPtr ev = Marshal.AllocHGlobal(IntPtr.Size + 4);
            try {
                while (!StopRequested(outDir)) {
                    // State 4: wait for paper using the feeder sensor when there is one.
                    if (feederSensor) {
                        double? loaded = Tw.GetCurrent(CAP_FEEDERLOADED);
                        if (!(loaded.HasValue && loaded.Value != 0)) {
                            if (!waitingSaid) { Json.Emit("\"event\":\"waiting\",\"t\":" + T()); waitingSaid = true; }
                            Pump(msg, 250);
                            continue;
                        }
                    }
                    waitingSaid = false;
                    int got = Batch(outDir, ref total, duplex, msg, ev);
                    if (got < 0) return 7;                        // couldn't enable -- fatal
                    if (got == 0 && !feederSensor) {
                        if (!waitingSaid) { Json.Emit("\"event\":\"waiting\",\"t\":" + T()); waitingSaid = true; }
                        Pump(msg, 1000);                          // empty feeder: retry shortly
                    }
                }
            } finally { Marshal.FreeHGlobal(msg); Marshal.FreeHGlobal(ev); }
        } finally {
            Tw.CloseDs();
        }
        return 0;
    }

    // Keep the window's message queue serviced for `ms` milliseconds.
    static void Pump(IntPtr msg, int ms) {
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < ms) {
            while (Native.PeekMessage(msg, IntPtr.Zero, 0, 0, 1)) { Native.TranslateMessage(msg); Native.DispatchMessage(msg); }
            Thread.Sleep(10);
        }
    }

    // One stack: ENABLEDS, transfer every page, DISABLEDS (back to state 4).
    // Returns pages transferred, or -1 if the source wouldn't enable.
    static int Batch(string outDir, ref int total, bool duplex, IntPtr msg, IntPtr ev) {
        IntPtr ui;
        ushort rc = Tw.Enable(out ui);
        if (rc != Tw.RC_SUCCESS && rc != Tw.RC_CHECKSTATUS) {
            ushort cc = Tw.CondCode();
            Marshal.FreeHGlobal(ui);
            Json.Emit("\"event\":\"error\",\"rc\":" + rc + ",\"cc\":" + cc + ",\"msg\":\"MSG_ENABLEDS failed\"");
            return -1;
        }
        int pages = 0;
        bool done = false;
        try {
            while (!done) {
                if (StopRequested(outDir)) break;
                if (!Native.PeekMessage(msg, IntPtr.Zero, 0, 0, 1)) { Thread.Sleep(5); continue; }
                ushort twMsg;
                ushort erc = Tw.ProcessEvent(msg, ev, out twMsg);
                if (erc != Tw.RC_DSEVENT) { Native.TranslateMessage(msg); Native.DispatchMessage(msg); continue; }
                if (twMsg == Tw.MSG_CLOSEDSREQ) {
                    Json.Emit("\"event\":\"error\",\"rc\":0,\"cc\":0,\"recoverable\":true,\"msg\":\"the scanner stopped the batch (jam, double feed or cover open?)\"");
                    done = true;
                } else if (twMsg == Tw.MSG_CLOSEDSOK) {
                    done = true;
                } else if (twMsg == Tw.MSG_XFERREADY) {
                    int failures = 0;
                    while (true) {
                        int n = total + 1;
                        string final = Path.Combine(outDir, string.Format("page_{0:D5}.jpg", n));
                        string part = final + ".part";
                        ushort xrc = Tw.FileXfer(part, 4);
                        if (xrc == Tw.RC_XFERDONE) {
                            if (File.Exists(final)) File.Delete(final);
                            File.Move(part, final);
                            total = n;
                            pages++;
                            string side = (duplex && pages % 2 == 0) ? "B" : "A";
                            Json.Emit(string.Format("\"event\":\"page\",\"n\":{0},\"side\":\"{1}\",\"file\":{2},\"t\":{3}",
                                n, side, Json.Str(final), T()));
                        } else {
                            failures++;
                            Json.Emit("\"event\":\"error\",\"rc\":" + xrc + ",\"cc\":" + Tw.CondCode() + ",\"recoverable\":true,\"msg\":\"page transfer failed\"");
                        }
                        int pending = Tw.EndXfer();
                        if (pending == 0) { done = true; break; }
                        if (failures > 3) { Tw.ResetXfers(); done = true; break; }
                        if (StopRequested(outDir)) { Tw.ResetXfers(); done = true; break; }
                    }
                }
            }
        } finally {
            Tw.Disable(ui);
        }
        if (duplex && pages % 2 == 1)
            Json.Emit("\"event\":\"warning\",\"msg\":\"odd page count in a duplex batch\",\"pages\":" + pages);
        if (pages > 0)
            Json.Emit("\"event\":\"batch_end\",\"pages\":" + pages + ",\"t\":" + T());
        return pages;
    }
}
