// AlphaFold 3 combined viewer
// Left:  toggle between PAE heatmap (Plotly) and per-residue pLDDT rankings (Plotly)
// Right: NGL 3D structure viewer with sample selector (CIF format)

if (!data || data.error) {
  container.innerHTML = `<div style="padding:16px;color:#dc3545;font-size:0.9em">
    ${data && data.error ? data.error : 'No data received'}</div>`;
  throw new Error('No data');
}

const { matrix, matrix_max, matrix_label,
        best_sample, n_residues, cifs, samples,
        ranking_score, ptm, iptm } = data;
const n      = n_residues;
const matMax = matrix_max || 31.75;

// ── Helpers ───────────────────────────────────────────────────────────────────
function paeRGB(val) {
  const t = Math.min(Math.max(val / matMax, 0), 1);
  let r, g, b;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(s * 80);       g = Math.round(53 + s * 183);  b = Math.round(186 + s * 28);
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(80 + s * 175); g = Math.round(236 - s * 116); b = Math.round(214 - s * 164);
  }
  return [r, g, b];
}

// ── Outer shell ───────────────────────────────────────────────────────────────
container.style.cssText =
  'font-family:system-ui,sans-serif;font-size:13px;display:flex;flex-direction:column;' +
  'gap:6px;padding:4px;height:100%;box-sizing:border-box;';

// ── Header badges ─────────────────────────────────────────────────────────────
const header = document.createElement('div');
header.style.cssText = 'display:flex;align-items:center;gap:7px;flex-wrap:wrap;flex-shrink:0';
function badge(lbl, val, bg) {
  const s = document.createElement('span');
  s.style.cssText = `padding:2px 9px;border-radius:20px;background:${bg};font-size:12px;white-space:nowrap`;
  s.innerHTML = `<span style="color:#666">${lbl}</span> <strong>${val}</strong>`;
  return s;
}
const bestLabel = samples && samples[best_sample] ? samples[best_sample].label : `Sample ${best_sample + 1}`;
header.appendChild(badge('Best',      bestLabel,                                     '#e7f5ff'));
if (ranking_score != null) header.appendChild(badge('Score', ranking_score.toFixed(3), '#ebfbee'));
if (ptm  != null) header.appendChild(badge('pTM',  ptm.toFixed(3),  '#fff9db'));
if (iptm != null) header.appendChild(badge('ipTM', iptm.toFixed(3), '#fff0f6'));
header.appendChild(badge('Residues', n, '#f8f9fa'));
container.appendChild(header);

// ── Main row ──────────────────────────────────────────────────────────────────
const mainRow = document.createElement('div');
mainRow.style.cssText = 'display:flex;gap:10px;flex:1;min-height:0;';

// ── LEFT panel ────────────────────────────────────────────────────────────────
const leftPanel = document.createElement('div');
leftPanel.style.cssText = 'display:flex;flex-direction:column;gap:5px;width:46%;flex-shrink:0;min-height:0;';

const leftCtrl = document.createElement('div');
leftCtrl.style.cssText = 'display:flex;align-items:center;gap:6px;flex-shrink:0';

function mkToggleBtn(label, active, rounded) {
  const b = document.createElement('button');
  b.textContent = label;
  b.dataset.active = active ? '1' : '0';
  const radius = rounded === 'left'  ? '4px 0 0 4px' :
                 rounded === 'right' ? '0 4px 4px 0'  : '4px';
  b.style.cssText =
    `padding:3px 12px;font-size:12px;cursor:pointer;border:1px solid #dee2e6;` +
    `background:${active ? '#0053D6' : '#f8f9fa'};color:${active ? '#fff' : '#333'};` +
    `border-radius:${radius};`;
  return b;
}

const btnPAE   = mkToggleBtn('PAE',      true,  'left');
const btnRanks = mkToggleBtn('Rankings', false, 'right');
leftCtrl.appendChild(btnPAE);
leftCtrl.appendChild(btnRanks);
leftPanel.appendChild(leftCtrl);

const plotDiv = document.createElement('div');
plotDiv.style.cssText = 'flex:1;min-height:0;';
leftPanel.appendChild(plotDiv);
mainRow.appendChild(leftPanel);

// ── RIGHT panel ───────────────────────────────────────────────────────────────
const rightPanel = document.createElement('div');
rightPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:5px;min-width:0;';

const rightCtrl = document.createElement('div');
rightCtrl.style.cssText = 'display:flex;align-items:center;gap:8px;flex-shrink:0';
rightCtrl.innerHTML = '<span style="font-weight:600;font-size:12px;color:#444">3D Structure</span>';

const availableSamples = Object.keys(cifs || {}).sort();
if (availableSamples.length > 1) {
  const sel = document.createElement('select');
  sel.style.cssText = 'padding:2px 6px;border:1px solid #ced4da;border-radius:4px;font-size:12px';
  for (const sk of availableSamples) {
    const o   = document.createElement('option');
    const idx = parseInt(sk.replace('sample_', ''));
    const s   = samples && samples[idx];
    const scoreStr = (s && s.ranking_score != null) ? `  Score ${s.ranking_score.toFixed(3)}` : '';
    o.value       = sk;
    o.textContent = (s ? s.label : sk) + scoreStr;
    if (idx === best_sample) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener('change', e => loadSample(e.target.value));
  rightCtrl.appendChild(sel);
}
rightPanel.appendChild(rightCtrl);

const nglDiv = document.createElement('div');
nglDiv.style.cssText =
  'flex:1;min-height:0;border:1px solid #dee2e6;border-radius:4px;background:#fff;position:relative';
rightPanel.appendChild(nglDiv);

const nglLegend = document.createElement('div');
nglLegend.style.cssText = 'display:flex;gap:10px;font-size:11px;flex-wrap:wrap;flex-shrink:0';
nglLegend.innerHTML = [
  ['#0053D6','>90'], ['#65CBF3','70–90'], ['#FFDB13','50–70'], ['#FF7D45','<50'],
].map(([c,l]) => `<span><span style="color:${c};font-size:13px">●</span> pLDDT ${l}</span>`).join('');
rightPanel.appendChild(nglLegend);

mainRow.appendChild(rightPanel);
container.appendChild(mainRow);

// ── NGL ───────────────────────────────────────────────────────────────────────
const stage = new NGL.Stage(nglDiv, { backgroundColor: 'white' });

function loadSample(sampleKey) {
  stage.removeAllComponents();
  const b64 = cifs && cifs[sampleKey];
  if (!b64) {
    nglDiv.innerHTML = '<div style="padding:10px;color:#dc3545;font-size:12px">CIF not found</div>';
    return;
  }
  stage.loadFile('data:text/plain;base64,' + b64, { ext: 'cif' }).then(comp => {
    comp.addRepresentation('cartoon', {
      colorScheme: 'bfactor',
      colorScale:  ['#FF7D45', '#FFDB13', '#65CBF3', '#0053D6'],
      colorDomain: [0, 100],
    });
    comp.autoView();
  }).catch(err => {
    nglDiv.innerHTML = `<div style="padding:10px;color:#dc3545;font-size:12px">Error: ${err.message}</div>`;
  });
}

const defaultSample = `sample_${best_sample}`;
if (availableSamples.length > 0) setTimeout(() => loadSample(defaultSample), 50);

// ── Plotly — PAE heatmap ──────────────────────────────────────────────────────
const residueNums = Array.from({ length: n }, (_, i) => i + 1);
const paeColorscale = Array.from({ length: 11 }, (_, i) => {
  const [r, g, b] = paeRGB(i * matMax / 10);
  return [i / 10, `rgb(${r},${g},${b})`];
});

const paeTrace = {
  type: 'heatmap', z: matrix, x: residueNums, y: residueNums,
  colorscale: paeColorscale, zmin: 0, zmax: matMax,
  hoverongaps: false,
  hovertemplate: 'Scored: %{x}<br>Aligned to: %{y}<br>%{z:.1f} Å<extra></extra>',
  colorbar: {
    title: { text: 'Å', side: 'right', font: { size: 10 } },
    thickness: 12, len: 0.85, tickfont: { size: 9 },
  },
};

const paeLayout = {
  margin: { l: 48, r: 55, t: 8, b: 42 },
  xaxis: { title: { text: 'Scored residue', font: { size: 10 } }, tickfont: { size: 9 }, showgrid: false },
  yaxis: { title: { text: 'Aligned to', font: { size: 10 } }, tickfont: { size: 9 }, autorange: 'reversed', showgrid: false },
  paper_bgcolor: 'white', plot_bgcolor: 'white',
};

// ── Plotly — Rankings (per-residue pLDDT all samples) ────────────────────────
const rankColors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6'];

const rankTraces = (samples || []).map((s, i) => ({
  type: 'scatter', mode: 'lines',
  name: s.label + (s.ranking_score != null ? `  Score ${s.ranking_score.toFixed(3)}` : ''),
  x: Array.from({ length: (s.plddt || []).length }, (_, k) => k + 1),
  y: s.plddt || [],
  line: { color: rankColors[i % rankColors.length], width: i === best_sample ? 2 : 1.5 },
}));

const confidenceBands = [
  { y0: 90, y1: 100, color: 'rgba(0,83,214,0.07)'   },
  { y0: 70, y1: 90,  color: 'rgba(101,203,243,0.07)' },
  { y0: 50, y1: 70,  color: 'rgba(255,219,19,0.07)'  },
  { y0: 0,  y1: 50,  color: 'rgba(255,125,69,0.07)'  },
].map(b => ({
  type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: b.y0, y1: b.y1,
  fillcolor: b.color, line: { width: 0 }, layer: 'below',
}));

const rankLayout = {
  margin: { l: 45, r: 10, t: 8, b: 42 },
  xaxis: { title: { text: 'Residue position', font: { size: 10 } }, tickfont: { size: 9 }, showgrid: false },
  yaxis: { title: { text: 'pLDDT', font: { size: 10 } }, tickfont: { size: 9 }, range: [0, 100], showgrid: false },
  shapes: confidenceBands,
  legend: { orientation: 'h', y: -0.18, font: { size: 10 } },
  paper_bgcolor: 'white', plot_bgcolor: 'white',
};

// ── Toggle logic ──────────────────────────────────────────────────────────────
let activeTab = 'pae';
let plotlyReady = false;

function setActiveTab(tab) {
  activeTab = tab;
  const isPAE = tab === 'pae';
  btnPAE.style.background   = isPAE  ? '#0053D6' : '#f8f9fa';
  btnPAE.style.color        = isPAE  ? '#fff'    : '#333';
  btnRanks.style.background = !isPAE ? '#0053D6' : '#f8f9fa';
  btnRanks.style.color      = !isPAE ? '#fff'    : '#333';

  if (!plotlyReady) {
    Plotly.newPlot(plotDiv, isPAE ? [paeTrace] : rankTraces,
                            isPAE ? paeLayout  : rankLayout,
                   { responsive: true, displayModeBar: false })
      .then(() => { plotlyReady = true; });
  } else {
    Plotly.react(plotDiv, isPAE ? [paeTrace] : rankTraces,
                          isPAE ? paeLayout  : rankLayout);
  }
}

btnPAE.addEventListener('click',   () => setActiveTab('pae'));
btnRanks.addEventListener('click', () => setActiveTab('ranks'));
setActiveTab('pae');

// ── Resize ────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  stage.handleResize();
  Plotly.Plots.resize(plotDiv);
});
