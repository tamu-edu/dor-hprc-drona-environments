// Integrated AlphaFold viewer — three-way linked:
//   PAE heatmap (Plotly)  ↔  NGL 3D structure  ↔  sequence bar
//
// Select a residue in any panel → all three update:
//   • PAE heatmap highlights that row (how confidently other residues are
//     positioned relative to the selected one)
//   • 3D structure recolors by those PAE values (blue=confident, orange=uncertain)
//   • Sequence bar highlights the selected position

if (!data || data.error) {
  container.innerHTML = `<div style="padding:16px;color:#dc3545;font-size:0.9em">
    ${data && data.error ? data.error : 'No data received'}</div>`;
  throw new Error('No data');
}

const { matrix, data_type, matrix_label, matrix_max,
        plddt_per_residue, plddt_mean, ptm, iptm,
        model, n_residues, all_models, pdbs, chains, sequence } = data;
const n      = n_residues;
const matMax = matrix_max || 30;
const seq    = sequence || '';

// ── Color helpers ─────────────────────────────────────────────────────────────
function paeRGB(val) {
  const t = Math.min(Math.max(val / matMax, 0), 1);
  let r, g, b;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(s * 80);        g = Math.round(53 + s * 183);   b = Math.round(186 + s * 28);
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(80 + s * 175);  g = Math.round(236 - s * 116);  b = Math.round(214 - s * 164);
  }
  return [r, g, b];
}
function paeToHex(val)  { const [r,g,b] = paeRGB(val); return (r<<16)|(g<<8)|b; }
function plddtColor(b)  {
  if (b >= 90) return '#0053D6';
  if (b >= 70) return '#65CBF3';
  if (b >= 50) return '#FFDB13';
  return '#FF7D45';
}
function plddtToHex(b)  {
  if (b >= 90) return 0x0053D6;
  if (b >= 70) return 0x65CBF3;
  if (b >= 50) return 0xFFDB13;
  return 0xFF7D45;
}

// ── State ─────────────────────────────────────────────────────────────────────
let selectedRes  = null;   // 0-indexed, null = pLDDT mode
let hoveredRes   = null;   // 0-indexed, for cursor highlight only
let baseRep      = null;
let currentComp  = null;
let plotlyReady  = false;

// ── Layout ────────────────────────────────────────────────────────────────────
container.style.cssText =
  'font-family:system-ui,sans-serif;font-size:13px;display:flex;flex-direction:column;gap:5px;padding:4px;height:100%;box-sizing:border-box;';

// Header
const header = document.createElement('div');
header.style.cssText = 'display:flex;align-items:center;gap:7px;flex-wrap:wrap;flex-shrink:0';
function badge(label, val, bg) {
  const s = document.createElement('span');
  s.style.cssText = `padding:2px 9px;border-radius:20px;background:${bg};font-size:12px;white-space:nowrap`;
  s.innerHTML = `<span style="color:#666">${label}</span> <strong>${val}</strong>`;
  return s;
}
header.appendChild(badge('Model', model.replace(/_pred_0$/, ''), '#e7f5ff'));
header.appendChild(badge('Mean pLDDT', plddt_mean.toFixed(1), '#ebfbee'));
if (ptm  != null) header.appendChild(badge('pTM',  ptm.toFixed(3),  '#fff9db'));
if (iptm != null) header.appendChild(badge('ipTM', iptm.toFixed(3), '#fff0f6'));
header.appendChild(badge('Residues', n, '#f8f9fa'));
if (data_type === 'distance_map') {
  const w = document.createElement('span');
  w.style.cssText = 'font-size:11px;color:#868e96';
  w.textContent = 'ℹ️ Expected Cβ distance (no true PAE for monomer preset)';
  header.appendChild(w);
}
container.appendChild(header);

// Main row
const mainRow = document.createElement('div');
mainRow.style.cssText = 'display:flex;gap:10px;flex:1;min-height:0;';

// ── LEFT: PAE heatmap ─────────────────────────────────────────────────────────
const leftPanel = document.createElement('div');
leftPanel.style.cssText = 'display:flex;flex-direction:column;gap:3px;flex-shrink:0;width:300px';

const paeLabel = document.createElement('div');
paeLabel.style.cssText = 'font-weight:600;font-size:11px;color:#555';
paeLabel.textContent = (matrix_label || 'Predicted Aligned Error (Å)') + ' — click to select residue';
leftPanel.appendChild(paeLabel);

const paeDiv = document.createElement('div');
paeDiv.style.cssText = 'flex:1;min-height:0;';
leftPanel.appendChild(paeDiv);
mainRow.appendChild(leftPanel);

// ── RIGHT: NGL viewer ─────────────────────────────────────────────────────────
const rightPanel = document.createElement('div');
rightPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:3px;min-width:0';

const nglHeader = document.createElement('div');
nglHeader.style.cssText = 'display:flex;align-items:center;gap:8px;flex-shrink:0';
nglHeader.innerHTML = '<span style="font-weight:600;font-size:11px;color:#555">3D Structure — hover or click a residue</span>';

const availableRanks = Object.keys(pdbs || {}).sort();
if (availableRanks.length > 1) {
  const sel = document.createElement('select');
  sel.style.cssText = 'padding:2px 6px;border:1px solid #ced4da;border-radius:4px;font-size:11px';
  for (const rk of availableRanks) {
    const o = document.createElement('option');
    o.value = rk; o.textContent = 'Rank ' + rk.replace('rank_', '');
    sel.appendChild(o);
  }
  sel.addEventListener('change', e => loadRank(e.target.value));
  nglHeader.appendChild(sel);
}

const resetBtn = document.createElement('button');
resetBtn.textContent = 'Reset';
resetBtn.title = 'Reset to pLDDT coloring';
resetBtn.style.cssText =
  'margin-left:auto;padding:1px 8px;font-size:11px;border:1px solid #ced4da;border-radius:4px;background:#f8f9fa;cursor:pointer';
resetBtn.onclick = () => selectResidue(null);
nglHeader.appendChild(resetBtn);
rightPanel.appendChild(nglHeader);

const nglDiv = document.createElement('div');
nglDiv.style.cssText =
  'flex:1;min-height:0;border:1px solid #dee2e6;border-radius:4px;background:#fff;position:relative';
rightPanel.appendChild(nglDiv);

const nglLegend = document.createElement('div');
nglLegend.style.cssText = 'display:flex;gap:8px;font-size:11px;flex-wrap:wrap;flex-shrink:0';
rightPanel.appendChild(nglLegend);

mainRow.appendChild(rightPanel);
container.appendChild(mainRow);

// ── Sequence bar ──────────────────────────────────────────────────────────────
const seqSection = document.createElement('div');
seqSection.style.cssText = 'flex-shrink:0;display:flex;flex-direction:column;gap:2px';

const seqLabel = document.createElement('div');
seqLabel.style.cssText = 'font-size:10px;color:#888;font-weight:600';
seqLabel.textContent = 'Sequence (pLDDT coloring) — click to select';
seqSection.appendChild(seqLabel);

const SEQ_H = 22;
const seqCanvas = document.createElement('canvas');
seqCanvas.height = SEQ_H;
seqCanvas.style.cssText = `width:100%;height:${SEQ_H}px;cursor:pointer;border-radius:3px;display:block`;
seqSection.appendChild(seqCanvas);
container.appendChild(seqSection);

// Status bar
const statusBar = document.createElement('div');
statusBar.style.cssText =
  'flex-shrink:0;font-size:11px;color:#555;padding:2px 6px;background:#f8f9fa;border-radius:4px;min-height:16px';
container.appendChild(statusBar);

// ── Core: select a residue (0-indexed), null = reset ─────────────────────────
function selectResidue(idx) {
  selectedRes = idx;
  updateNGLColor();
  updatePAEShapes();
  drawSeqBar();
  updateLegendAndStatus();
}

// ── NGL ───────────────────────────────────────────────────────────────────────
const stage = new NGL.Stage(nglDiv, { backgroundColor: 'white' });

const schemeId = NGL.ColormakerRegistry.addScheme(function() {
  this.atomColor = function(atom) {
    if (selectedRes !== null) {
      const row = matrix[selectedRes];
      const j   = atom.residueIndex;
      if (j >= 0 && j < row.length) return paeToHex(row[j]);
    }
    return plddtToHex(atom.bfactor);
  };
});

function updateNGLColor() {
  if (!currentComp) return;
  if (baseRep) currentComp.removeRepresentation(baseRep);
  baseRep = currentComp.addRepresentation('cartoon', { colorScheme: schemeId });
}

function loadRank(rankKey) {
  stage.removeAllComponents();
  currentComp = null; baseRep = null;
  const b64 = pdbs && pdbs[rankKey];
  if (!b64) { nglDiv.innerHTML = '<div style="padding:10px;color:#dc3545;font-size:12px">PDB not found</div>'; return; }
  stage.loadFile('data:text/plain;base64,' + b64, { ext: 'pdb' }).then(comp => {
    currentComp = comp;
    updateNGLColor();
    comp.autoView();
  }).catch(err => {
    nglDiv.innerHTML = `<div style="padding:10px;color:#dc3545;font-size:12px">Error: ${err.message}</div>`;
  });
}

// NGL hover → highlight sequence bar cursor + PAE row cursor
let hoverTimer = null;
stage.signals.hovered.add(function(proxy) {
  const idx = (proxy && proxy.atom) ? proxy.atom.residueIndex : null;
  if (idx === hoveredRes) return;
  hoveredRes = idx;
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    drawSeqBar();
    updatePAEShapes();
  }, 30);  // 30ms debounce — keeps NGL interaction smooth
});

// NGL click → select that residue (triggers full recolor + all updates)
stage.signals.clicked.add(function(proxy) {
  if (proxy && proxy.atom) {
    selectResidue(proxy.atom.residueIndex);
  }
});

if (availableRanks.length > 0) setTimeout(() => loadRank(availableRanks[0]), 50);

// ── Plotly PAE heatmap ────────────────────────────────────────────────────────
const residueNums = Array.from({ length: n }, (_, i) => i + 1);
const paeColorscale = [];
for (let i = 0; i <= 10; i++) {
  const [r, g, b] = paeRGB(i * matMax / 10);
  paeColorscale.push([i / 10, `rgb(${r},${g},${b})`]);
}

Plotly.newPlot(paeDiv, [{
  type: 'heatmap', z: matrix, x: residueNums, y: residueNums,
  colorscale: paeColorscale, zmin: 0, zmax: matMax,
  hoverongaps: false,
  hovertemplate: 'Scored: %{x}<br>Aligned to: %{y}<br>%{z:.1f} Å<extra></extra>',
  colorbar: { title: { text: 'Å', side: 'right', font: { size: 10 } }, thickness: 12, len: 0.85, tickfont: { size: 9 } },
}], {
  margin: { l: 48, r: 55, t: 8, b: 42 },
  xaxis: { title: { text: 'Scored residue', font: { size: 10 } }, tickfont: { size: 9 }, showgrid: false },
  yaxis: { title: { text: 'Aligned to', font: { size: 10 } }, tickfont: { size: 9 }, autorange: 'reversed', showgrid: false },
  paper_bgcolor: 'white', plot_bgcolor: 'white',
}, { responsive: true, displayModeBar: false }).then(() => { plotlyReady = true; });

// PAE click → select that row's residue
paeDiv.on('plotly_click', function(ev) {
  const pt = ev.points[0];
  selectResidue(pt.y - 1);   // pt.y is 1-indexed
});

// PAE hover → update sequence bar cursor (but don't trigger full recolor)
paeDiv.on('plotly_hover', function(ev) {
  const pt = ev.points[0];
  hoveredRes = pt.y - 1;
  drawSeqBar();
});
paeDiv.on('plotly_unhover', function() {
  hoveredRes = null;
  drawSeqBar();
});

function updatePAEShapes() {
  if (!plotlyReady) return;
  const shapes = [];
  if (selectedRes !== null) {
    shapes.push({
      type: 'rect',
      x0: 0.5, x1: n + 0.5,
      y0: selectedRes + 0.5, y1: selectedRes + 1.5,
      xref: 'x', yref: 'y',
      line: { color: 'white', width: 2 },
      fillcolor: 'rgba(255,255,255,0.15)',
    });
  }
  if (hoveredRes !== null && hoveredRes !== selectedRes) {
    shapes.push({
      type: 'line',
      x0: 0.5, x1: n + 0.5,
      y0: hoveredRes + 1, y1: hoveredRes + 1,
      xref: 'x', yref: 'y',
      line: { color: 'rgba(255,255,255,0.5)', width: 1, dash: 'dot' },
    });
  }
  Plotly.relayout(paeDiv, { shapes });
}

// ── Sequence bar ──────────────────────────────────────────────────────────────
function drawSeqBar() {
  const W = seqCanvas.clientWidth || seqCanvas.parentElement.clientWidth || 600;
  if (seqCanvas.width !== W) seqCanvas.width = W;
  const sc  = seqCanvas.getContext('2d');
  const pld = plddt_per_residue || [];
  const cw  = W / n;

  sc.clearRect(0, 0, W, SEQ_H);

  for (let i = 0; i < n; i++) {
    sc.fillStyle = plddtColor(pld[i] != null ? pld[i] : 50);
    sc.fillRect(i * cw, 0, Math.max(cw - 0.3, 1), SEQ_H);
  }

  // Chain boundaries
  if (chains && chains.length > 1) {
    sc.strokeStyle = '#fff'; sc.lineWidth = 2;
    for (const ch of chains) {
      if (ch.start > 0) {
        const x = (ch.start / n) * W;
        sc.beginPath(); sc.moveTo(x, 0); sc.lineTo(x, SEQ_H); sc.stroke();
      }
    }
  }

  // Amino acid letters (show when cells are wide enough)
  if (cw >= 7 && seq) {
    const fontSize = Math.min(Math.floor(cw * 0.75), 11);
    sc.font = `${fontSize}px monospace`;
    sc.textAlign = 'center';
    sc.textBaseline = 'middle';
    for (let i = 0; i < seq.length && i < n; i++) {
      const v = pld[i] != null ? pld[i] : 50;
      sc.fillStyle = v >= 50 ? 'rgba(0,0,0,0.65)' : 'rgba(255,255,255,0.85)';
      sc.fillText(seq[i], (i + 0.5) * cw, SEQ_H / 2);
    }
  }

  // Hovered residue (subtle outline)
  if (hoveredRes !== null && hoveredRes !== selectedRes) {
    sc.strokeStyle = 'rgba(0,0,0,0.4)'; sc.lineWidth = 1;
    sc.strokeRect(hoveredRes * cw + 0.5, 0.5, Math.max(cw - 1, 1), SEQ_H - 1);
  }

  // Selected residue (bright outline)
  if (selectedRes !== null) {
    sc.strokeStyle = '#fff'; sc.lineWidth = 2.5;
    sc.strokeRect(selectedRes * cw + 1.5, 1.5, Math.max(cw - 3, 1), SEQ_H - 3);
    sc.strokeStyle = '#111'; sc.lineWidth = 1;
    sc.strokeRect(selectedRes * cw + 0.5, 0.5, Math.max(cw - 1, 1), SEQ_H - 1);
  }
}

seqCanvas.addEventListener('click', function(e) {
  const cw  = seqCanvas.clientWidth / n;
  const idx = Math.floor(e.offsetX / cw);
  if (idx >= 0 && idx < n) selectResidue(idx);
});

seqCanvas.addEventListener('mousemove', function(e) {
  const cw  = seqCanvas.clientWidth / n;
  const idx = Math.floor(e.offsetX / cw);
  if (idx >= 0 && idx < n && idx !== hoveredRes) {
    hoveredRes = idx;
    drawSeqBar();
  }
});

seqCanvas.addEventListener('mouseleave', function() {
  hoveredRes = null;
  drawSeqBar();
});

// ── Legend & status ───────────────────────────────────────────────────────────
function updateLegendAndStatus() {
  if (selectedRes === null) {
    nglLegend.innerHTML = [
      ['#0053D6','>90'], ['#65CBF3','70–90'], ['#FFDB13','50–70'], ['#FF7D45','<50'],
    ].map(([c,l]) => `<span><span style="color:${c};font-size:13px">●</span> ${l}</span>`).join('');
    statusBar.textContent = 'Showing pLDDT confidence. Click any residue in the heatmap, structure, or sequence bar to explore PAE.';
  } else {
    const aa = seq[selectedRes] || '';
    const [r0,g0,b0] = paeRGB(0);
    const [rm,gm,bm] = paeRGB(matMax * 0.5);
    const [rh,gh,bh] = paeRGB(matMax);
    nglLegend.innerHTML =
      `<span><span style="color:rgb(${r0},${g0},${b0});font-size:13px">●</span> 0 Å (very confident)</span>` +
      `<span><span style="color:rgb(${rm},${gm},${bm});font-size:13px">●</span> ${(matMax*0.5).toFixed(0)} Å</span>` +
      `<span><span style="color:rgb(${rh},${gh},${bh});font-size:13px">●</span> ${matMax} Å (uncertain)</span>`;
    statusBar.textContent =
      `Residue ${selectedRes + 1}${aa ? ' (' + aa + ')' : ''} selected — structure colored by PAE relative to this residue. Click Reset or click again elsewhere to change.`;
  }
}

// ── Initial draw + resize ────────────────────────────────────────────────────
setTimeout(() => { drawSeqBar(); updateLegendAndStatus(); }, 100);

window.addEventListener('resize', () => {
  stage.handleResize();
  Plotly.Plots.resize(paeDiv);
  drawSeqBar();
});
