"""Assemble the interactive Seal Shift Loupe page from the rendered frames.

Reads <frames>/y and <frames>/x (PNG + manifest.json), re-encodes each frame to
JPEG for size, and writes a single self-contained HTML file with the sweep data
embedded. Vanilla JS, no libraries.
"""
import base64, json, sys
from pathlib import Path
import cv2

FRAMES = Path(sys.argv[1])
OUTHTML = Path(sys.argv[2])
Q = int(sys.argv[3]) if len(sys.argv) > 3 else 82
MAXW = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 0 = no downscale


def load_axis(sub):
    d = FRAMES / sub
    man = json.loads((d / 'manifest.json').read_text())
    out = []
    for m in man:
        img = cv2.imread(str(d / m['file']))
        if MAXW and img.shape[1] > MAXW:
            h = int(round(img.shape[0] * MAXW / img.shape[1]))
            img = cv2.resize(img, (MAXW, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, Q])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        out.append(dict(src='data:image/jpeg;base64,' + b64,
                        x=m['seal_x'], y=m['seal_y'], c=m['containment'],
                        serial=m['serial'], strap=m['strap']))
    return out


data = {'y': load_axis('y'), 'x': load_axis('x')}
payload = json.dumps(data, separators=(',', ':'))
print(f"y frames {len(data['y'])}  x frames {len(data['x'])}  "
      f"payload {len(payload)/1e6:.1f} MB")

HTML = r'''<title>Seal Shift Loupe</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,600&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap">
<style>
/* layout: masthead; loupe stage with live readout rail; sweep-plot scrubber; transport; legend + metric notes */
:root{
  --bg:#eceae2; --panel:#f8f6f0; --sunk:#e4e1d6; --ink:#1e231f; --muted:#6c716a;
  --line:#d6d2c6; --seal:#2f8f4e; --amber:#c6811b; --crit:#c2502c;
  --radius:14px;
  --disp:"Fraunces",Georgia,serif; --body:"Public Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#13161a; --panel:#1b1f24; --sunk:#0f1215; --ink:#e9e7e0; --muted:#9aa09a;
  --line:#2a2f35; --seal:#54b974; --amber:#e3a33e; --crit:#e2714a; color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#13161a; --panel:#1b1f24; --sunk:#0f1215; --ink:#e9e7e0; --muted:#9aa09a;
  --line:#2a2f35; --seal:#54b974; --amber:#e3a33e; --crit:#e2714a; color-scheme:dark;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:900px;margin:0 auto;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:28px 56px}
.mast{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
  border-bottom:1px solid var(--line);padding-bottom:16px}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--seal);font-weight:600}
h1{font-family:var(--disp);font-weight:600;font-size:clamp(1.9rem,5vw,2.9rem);
  line-height:1.02;margin:.25rem 0 .35rem;text-wrap:balance;letter-spacing:-.01em}
.sub{color:var(--muted);max-width:60ch;font-size:.98rem;margin:0}
.themebtn{flex:none;background:var(--panel);color:var(--muted);border:1px solid var(--line);
  border-radius:999px;width:38px;height:38px;cursor:pointer;font-size:1rem;line-height:1}
.themebtn:hover{color:var(--ink)}

.axis-toggle{display:inline-flex;gap:4px;background:var(--sunk);border:1px solid var(--line);
  border-radius:999px;padding:4px;margin:22px 0 14px}
.axis-toggle button{border:0;background:transparent;color:var(--muted);font-family:var(--mono);
  font-size:.82rem;font-weight:600;letter-spacing:.03em;padding:7px 15px;border-radius:999px;cursor:pointer}
.axis-toggle button[aria-pressed="true"]{background:var(--panel);color:var(--ink);
  box-shadow:0 1px 2px rgba(0,0,0,.12)}

.stage{display:grid;grid-template-columns:1fr 210px;gap:18px;align-items:start}
.framebox{position:relative;border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden;background:var(--sunk);min-width:0;box-shadow:0 10px 30px -18px rgba(0,0,0,.5)}
.framebox img{display:block;width:100%;height:auto}
.rail{display:flex;flex-direction:column;gap:12px;min-width:0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:12px 14px}
.tile .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:var(--mono);font-weight:600;font-size:1.7rem;font-variant-numeric:tabular-nums;
  line-height:1.1;margin-top:3px}
.tile .d{font-size:.78rem;color:var(--muted);margin-top:2px}
.v.seal{color:var(--seal)} .v.amb{color:var(--amber)}
.tile.warn .v{color:var(--crit)}
.meta{font-family:var(--mono);font-size:.74rem;color:var(--muted);letter-spacing:.02em}

.plotwrap{margin-top:18px;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:14px 14px 10px}
.plotlabel{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.7rem;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
svg.plot{display:block;width:100%;height:118px;touch-action:none;cursor:ew-resize}

.transport{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:14px}
.play{display:inline-flex;align-items:center;gap:8px;background:var(--seal);color:#fff;border:0;
  border-radius:999px;padding:9px 18px;font-weight:600;font-size:.92rem;cursor:pointer}
.play:hover{filter:brightness(1.06)}
.seg{display:inline-flex;gap:2px;background:var(--sunk);border:1px solid var(--line);
  border-radius:999px;padding:3px}
.seg button{border:0;background:transparent;color:var(--muted);font-family:var(--mono);
  font-size:.76rem;font-weight:600;padding:6px 11px;border-radius:999px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--panel);color:var(--ink)}
.toggle-dir{font-family:var(--body);font-size:.85rem;color:var(--muted);display:inline-flex;
  align-items:center;gap:7px;cursor:pointer;user-select:none}
.toggle-dir input{accent-color:var(--seal);width:16px;height:16px}

.notes{margin-top:30px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.note{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:15px 16px}
.note h3{margin:0 0 5px;font-family:var(--mono);font-size:.74rem;letter-spacing:.08em;
  text-transform:uppercase}
.note.seal h3{color:var(--seal)} .note.amb h3{color:var(--amber)} .note.crit h3{color:var(--crit)}
.note p{margin:0;font-size:.86rem;color:var(--muted)}
.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:baseline}
.legend{margin-top:18px;font-size:.84rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:16px}
.legend span{display:inline-flex;align-items:center}
.foot{margin-top:26px;font-size:.78rem;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
@media (max-width:620px){
  .stage{grid-template-columns:1fr}
  .rail{flex-direction:row;flex-wrap:wrap}
  .tile{flex:1 1 140px}
  .notes{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>

<div class="wrap">
  <header class="mast">
    <div>
      <div class="eyebrow">Overprint diagnostics &middot; $1 FRN</div>
      <h1>Seal Shift Loupe</h1>
      <p class="sub">The green Treasury seal is overprinted on top of the engraved
        &ldquo;ONE&rdquo;. When the overprint plate drifts, the seal slides off that
        word. Here the &ldquo;ONE&rdquo; box is pinned in place so you can watch the
        seal move across real bills &mdash; and read the exact numbers the Seal&nbsp;X,
        Seal&nbsp;Y and Containment columns report.</p>
    </div>
    <button class="themebtn" id="theme" title="Toggle light / dark" aria-label="Toggle theme">&#9680;</button>
  </header>

  <div class="axis-toggle" role="group" aria-label="Which shift to sweep">
    <button id="axY" aria-pressed="true">Vertical&nbsp;&middot;&nbsp;Seal&nbsp;Y</button>
    <button id="axX" aria-pressed="false">Horizontal&nbsp;&middot;&nbsp;Seal&nbsp;X</button>
  </div>

  <div class="stage">
    <div class="framebox"><img id="frame" alt="Zoomed treasury seal over the ONE box"></div>
    <div class="rail">
      <div class="tile" id="tY"><div class="k">Seal&nbsp;Y &middot; vertical</div>
        <div class="v seal" id="vY">&mdash;</div><div class="d">+ up &middot; &minus; down</div></div>
      <div class="tile" id="tX"><div class="k">Seal&nbsp;X &middot; horizontal</div>
        <div class="v" id="vX">&mdash;</div><div class="d">+ right &middot; &minus; left</div></div>
      <div class="tile" id="tC"><div class="k">Containment</div>
        <div class="v" id="vC">&mdash;</div><div class="d">seal area inside the box</div></div>
      <div class="meta" id="meta">&nbsp;</div>
    </div>
  </div>

  <div class="plotwrap">
    <div class="plotlabel"><span id="plotTitle">Seal Y across the sweep</span>
      <span id="plotRange"></span></div>
    <svg class="plot" id="plot" preserveAspectRatio="none" aria-hidden="true"></svg>
  </div>

  <div class="transport">
    <button class="play" id="play"><span id="playIcon">&#9658;</span><span id="playTxt">Play</span></button>
    <div class="seg" role="group" aria-label="Speed">
      <button data-ms="420" id="sp0">Slow</button>
      <button data-ms="200" id="sp1" aria-pressed="true">Normal</button>
      <button data-ms="90" id="sp2">Fast</button>
    </div>
    <label class="toggle-dir"><input type="checkbox" id="pingpong" checked>
      Sweep back &amp; forth</label>
  </div>

  <div class="legend">
    <span><span class="swatch" style="background:var(--seal)"></span>&ldquo;ONE&rdquo; box &mdash; the pinned reference</span>
    <span><span class="swatch" style="background:var(--crit)"></span>Seal pushed outside the box</span>
    <span><span class="swatch" style="background:var(--amber)"></span>Shift vector &amp; distance</span>
  </div>

  <div class="notes">
    <div class="note seal"><h3>Seal X / Seal Y</h3><p>Center-to-center offset of the
      seal from the &ldquo;ONE&rdquo; box, as a percent of that box&rsquo;s width and
      height. Zero means dead-centered.</p></div>
    <div class="note crit"><h3>Containment</h3><p>Share of the seal still sitting inside
      the box. 100% is a clean strike; it falls as the seal drifts off &mdash; the red
      area is what has escaped.</p></div>
    <div class="note amb"><h3>Why it matters</h3><p>A large, consistent shift is a
      collectible overprint-registration error. These columns flag candidates so you can
      pull them for a closer look.</p></div>
  </div>

  <p class="foot" id="foot">Each frame is a real scanned $1, aligned and zoomed on the
    detected &ldquo;ONE&rdquo; region, ordered by shift.</p>
</div>

<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function(){
  "use strict";
  var DATA, cur="y", idx=0, playing=false, timer=null, stepMs=200, pingpong=true, dir=1;
  try{ DATA=JSON.parse(document.getElementById('data').textContent); }catch(e){ DATA={y:[],x:[]}; }

  var $=function(id){return document.getElementById(id);};
  var frame=$('frame'), plot=$('plot');
  var vY=$('vY'),vX=$('vX'),vC=$('vC'),tC=$('tC'),tY=$('tY'),tX=$('tX'),meta=$('meta');

  function seq(){ return DATA[cur]||[]; }
  function sign(n){ return (n>0?'+':'')+n.toFixed(1); }

  function paint(){
    var s=seq(); if(!s.length) return;
    if(idx<0)idx=0; if(idx>=s.length)idx=s.length-1;
    var f=s[idx];
    frame.src=f.src;
    vY.textContent=sign(f.y)+'%'; vX.textContent=sign(f.x)+'%'; vC.textContent=f.c.toFixed(0)+'%';
    tC.classList.toggle('warn', f.c<97);
    // emphasize the axis being swept
    tY.style.opacity = cur==='y'?'1':'.6'; tX.style.opacity = cur==='x'?'1':'.6';
    vX.className = 'v'+(cur==='x'?' amb':'');
    vY.className = 'v'+(cur==='y'?' seal':'');
    meta.innerHTML = f.serial+' &middot; strap '+f.strap+' &middot; '+(idx+1)+' / '+s.length;
    marker();
  }

  // ---- sweep plot (value vs frame), doubles as scrubber ----
  var PW=600, PH=118, PAD=14;
  function buildPlot(){
    var s=seq(); if(!s.length){plot.innerHTML='';return;}
    var key=cur; var vals=s.map(function(f){return f[key];});
    var mn=Math.min.apply(null,vals), mx=Math.max.apply(null,vals);
    if(mn===mx){mn-=1;mx+=1;}
    plot.setAttribute('viewBox','0 0 '+PW+' '+PH);
    var X=function(i){return PAD + i*(PW-2*PAD)/Math.max(1,s.length-1);};
    var Y=function(v){return PAD + (mx-v)*(PH-2*PAD)/(mx-mn);};
    var pts=s.map(function(f,i){return X(i)+','+Y(f[key]);}).join(' ');
    var zero = (mn<0&&mx>0)? ('<line x1="'+PAD+'" y1="'+Y(0)+'" x2="'+(PW-PAD)+'" y2="'+Y(0)+
      '" stroke="var(--line)" stroke-dasharray="4 4" />'
      +'<text x="'+(PW-PAD)+'" y="'+(Y(0)-4)+'" text-anchor="end" font-family="var(--mono)" '
      +'font-size="9" fill="var(--muted)">0%</text>') : '';
    plot.innerHTML =
      zero +
      '<polyline points="'+pts+'" fill="none" stroke="var(--'+(cur==='x'?'amber':'seal')+')" '+
        'stroke-width="2" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>' +
      s.map(function(f,i){return '<circle cx="'+X(i)+'" cy="'+Y(f[key])+'" r="2" '+
        'fill="var(--muted)" opacity=".45"/>';}).join('') +
      '<line id="pmark" x1="0" y1="'+PAD+'" x2="0" y2="'+(PH-PAD)+'" stroke="var(--ink)" '+
        'stroke-width="1.5"/>'+
      '<circle id="pdot" r="4.5" fill="var(--'+(cur==='x'?'amber':'seal')+')" '+
        'stroke="var(--panel)" stroke-width="2"/>';
    plot._X=X; plot._Y=Y; plot._n=s.length;
    $('plotTitle').textContent = (cur==='y'?'Seal Y':'Seal X')+' across the sweep';
    $('plotRange').textContent = sign(mn)+'%  to  '+sign(mx)+'%';
  }
  function marker(){
    var m=$('pmark'), d=$('pdot'); if(!m||!plot._X) return;
    var s=seq(), x=plot._X(idx), y=plot._Y(s[idx][cur]);
    m.setAttribute('x1',x); m.setAttribute('x2',x);
    d.setAttribute('cx',x); d.setAttribute('cy',y);
  }
  function scrubFromEvent(ev){
    var r=plot.getBoundingClientRect();
    var px=(ev.touches?ev.touches[0].clientX:ev.clientX)-r.left;
    var frac=px/r.width; var i=Math.round(frac*(plot._n-1));
    idx=Math.max(0,Math.min(plot._n-1,i)); paint();
  }
  var dragging=false;
  plot.addEventListener('pointerdown',function(e){dragging=true;stop();scrubFromEvent(e);plot.setPointerCapture(e.pointerId);});
  plot.addEventListener('pointermove',function(e){if(dragging)scrubFromEvent(e);});
  plot.addEventListener('pointerup',function(){dragging=false;});

  // ---- transport ----
  function tick(){
    var s=seq();
    idx+=dir;
    if(idx>=s.length){ if(pingpong){dir=-1;idx=s.length-2;} else {idx=0;} }
    else if(idx<0){ dir=1; idx=1; }
    if(idx<0)idx=0; if(idx>=s.length)idx=s.length-1;
    paint();
  }
  function play(){ if(playing)return; playing=true; dir=1;
    $('playIcon').innerHTML='&#10073;&#10073;'; $('playTxt').textContent='Pause';
    timer=setInterval(tick,stepMs);
  }
  function stop(){ playing=false; if(timer){clearInterval(timer);timer=null;}
    $('playIcon').innerHTML='&#9658;'; $('playTxt').textContent='Play';
  }
  $('play').addEventListener('click',function(){ playing?stop():play(); });
  function setSpeed(ms,btn){ stepMs=ms; ['sp0','sp1','sp2'].forEach(function(id){
      $(id).setAttribute('aria-pressed', $(id)===btn ? 'true':'false'); });
    if(playing){stop();play();}
  }
  $('sp0').addEventListener('click',function(){setSpeed(420,this);});
  $('sp1').addEventListener('click',function(){setSpeed(200,this);});
  $('sp2').addEventListener('click',function(){setSpeed(90,this);});
  $('pingpong').addEventListener('change',function(){pingpong=this.checked;});

  function setAxis(a){ cur=a; idx=0; dir=1;
    $('axY').setAttribute('aria-pressed', a==='y'?'true':'false');
    $('axX').setAttribute('aria-pressed', a==='x'?'true':'false');
    buildPlot(); paint();
  }
  $('axY').addEventListener('click',function(){setAxis('y');});
  $('axX').addEventListener('click',function(){setAxis('x');});

  // ---- theme toggle (per-viewer) ----
  var root=document.documentElement;
  try{ var sv=localStorage.getItem('sealshift-theme'); if(sv)root.setAttribute('data-theme',sv); }catch(e){}
  $('theme').addEventListener('click',function(){
    var dark=getComputedStyle(root).getPropertyValue('color-scheme').indexOf('dark')>=0;
    var next=dark?'light':'dark'; root.setAttribute('data-theme',next);
    try{localStorage.setItem('sealshift-theme',next);}catch(e){}
    buildPlot(); marker();
  });

  // boot
  buildPlot(); paint();
  if(seq().length>1) play();
})();
</script>
'''

OUTHTML.write_text(HTML.replace('__PAYLOAD__', payload))
print(f"wrote {OUTHTML}  ({OUTHTML.stat().st_size/1e6:.1f} MB)")
