// PAE (Predicted Aligned Error) viewer for AlphaFold2 results.
// Renders a canvas-based heatmap + per-residue pLDDT chart + model table.
// No CDN libraries required.

if (!data || data.error) {
  container.innerHTML = `<div style="padding:16px;color:#dc3545;font-size:0.9em">
    ${data && data.error ? data.error : 'No PAE data received'}
  </div>`;
  throw new Error('No data');
}

const { matrix, data_type, matrix_label, matrix_max,
        plddt_per_residue, plddt_mean, ptm, iptm, model, n_residues, all_models } = data;
const pae = matrix;  // unified name internally

// ── Layout ────────────────────────────────────────────────────────────────────
container.style.cssText = 'display:flex;flex-direction:column;gap:16px;padding:4px;font-family:system-ui,sans-serif;font-size:13px;';

// ── Header row: model info + metrics ─────────────────────────────────────────
const header = document.createElement('div');
header.style.cssText = 'display:flex;align-items:center;gap:12px;flex-wrap:wrap';

const modelBadge = (label, value, color) => {
  const s = document.createElement('span');
  s.style.cssText = `display:inline-flex;align-items:center;gap:4px;padding:4px 10px;
    border-radius:20px;background:${color};font-size:12px;font-weight:500`;
  s.innerHTML = `<span style="color:#555">${label}</span> <strong>${value}</strong>`;
  return s;
};

header.appendChild(modelBadge('Best model', model.replace(/_pred_0$/, ''), '#e7f5ff'));
header.appendChild(modelBadge('Mean pLDDT', plddt_mean.toFixed(1), '#ebfbee'));
if (ptm  != null) header.appendChild(modelBadge('pTM',  ptm.toFixed(3),  '#fff9db'));
if (iptm != null) header.appendChild(modelBadge('ipTM', iptm.toFixed(3), '#fff0f6'));
header.appendChild(modelBadge('Residues', n_residues, '#f8f9fa'));

if (data_type === 'distance_map') {
  const note = document.createElement('div');
  note.style.cssText = 'width:100%;font-size:11px;color:#868e96;padding:2px 0';
  note.textContent = 'ℹ️ PAE requires monomer_ptm or multimer preset. Showing expected Cβ distance from distogram instead.';
  header.appendChild(note);
}

container.appendChild(header);

// ── Main row: heatmap + pLDDT chart ──────────────────────────────────────────
const mainRow = document.createElement('div');
mainRow.style.cssText = 'display:flex;gap:20px;align-items:flex-start;';

// ── PAE heatmap ───────────────────────────────────────────────────────────────
const heatmapWrap = document.createElement('div');
heatmapWrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:6px;';

const heatmapTitle = document.createElement('div');
heatmapTitle.style.cssText = 'font-weight:600;font-size:13px;color:#333;align-self:flex-start';
heatmapTitle.textContent = matrix_label || 'Predicted Aligned Error (Å)';
heatmapWrap.appendChild(heatmapTitle);

// Canvas for PAE heatmap
const SIZE = 300;
const canvas = document.createElement('canvas');
canvas.width  = SIZE;
canvas.height = SIZE;
canvas.style.cssText = `width:${SIZE}px;height:${SIZE}px;border:1px solid #dee2e6;border-radius:4px;cursor:crosshair`;

const ctx = canvas.getContext('2d');
const n = pae.length;

// PAE color scale: 0Å = dark blue → 15Å = light cyan → 30Å = yellow/orange
function paeRGB(angstrom) {
  const t = Math.min(Math.max(angstrom / (matrix_max || 30), 0), 1);
  let r, g, b;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(0   + s * 80);
    g = Math.round(53  + s * 183);
    b = Math.round(186 + s * 28);
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(80  + s * 220);
    g = Math.round(236 + s * -116);
    b = Math.round(214 + s * -164);
  }
  return [r, g, b];
}

// Draw using ImageData for performance (handles large proteins fast)
const imgData = ctx.createImageData(n, n);
for (let i = 0; i < n; i++) {
  for (let j = 0; j < n; j++) {
    const [r, g, b] = paeRGB(pae[i][j]);
    const idx = (i * n + j) * 4;
    imgData.data[idx]     = r;
    imgData.data[idx + 1] = g;
    imgData.data[idx + 2] = b;
    imgData.data[idx + 3] = 255;
  }
}

// Scale from n×n to SIZE×SIZE
const tmp = document.createElement('canvas');
tmp.width  = n;
tmp.height = n;
tmp.getContext('2d').putImageData(imgData, 0, 0);
ctx.imageSmoothingEnabled = false;
ctx.drawImage(tmp, 0, 0, SIZE, SIZE);

// Axis labels
ctx.fillStyle = '#666';
ctx.font = '11px system-ui';
ctx.fillText('Scored residue', 6, SIZE - 6);
ctx.save();
ctx.translate(12, SIZE - 40);
ctx.rotate(-Math.PI / 2);
ctx.fillText('Aligned to residue', 0, 0);
ctx.restore();

heatmapWrap.appendChild(canvas);

// Tooltip overlay
const tooltip = document.createElement('div');
tooltip.style.cssText = 'font-size:11px;color:#555;height:16px;';
heatmapWrap.appendChild(tooltip);
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = n / SIZE, scaleY = n / SIZE;
  const j = Math.floor((e.clientX - rect.left) * scaleX);
  const i = Math.floor((e.clientY - rect.top)  * scaleY);
  if (i >= 0 && i < n && j >= 0 && j < n) {
    tooltip.textContent = `Residue ${j + 1} → ${i + 1}: ${pae[i][j].toFixed(1)} Å`;
  }
});
canvas.addEventListener('mouseleave', () => { tooltip.textContent = ''; });

// Color legend bar
const legendWrap = document.createElement('div');
legendWrap.style.cssText = 'display:flex;align-items:center;gap:8px;font-size:11px;color:#555;';
const legendCanvas = document.createElement('canvas');
legendCanvas.width  = SIZE;
legendCanvas.height = 12;
legendCanvas.style.cssText = `width:${SIZE}px;height:12px;border-radius:3px`;
const lctx = legendCanvas.getContext('2d');
const lgrad = lctx.createLinearGradient(0, 0, SIZE, 0);
for (let step = 0; step <= 10; step++) {
  const [r, g, b] = paeRGB(step * 3);
  lgrad.addColorStop(step / 10, `rgb(${r},${g},${b})`);
}
lctx.fillStyle = lgrad;
lctx.fillRect(0, 0, SIZE, 12);

legendWrap.innerHTML = '<span>0 Å</span>';
legendWrap.appendChild(legendCanvas);
legendWrap.innerHTML += `<span>${matrix_max || 30} Å</span>`;
heatmapWrap.appendChild(legendWrap);

mainRow.appendChild(heatmapWrap);

// ── Per-residue pLDDT chart (canvas) ─────────────────────────────────────────
if (plddt_per_residue && plddt_per_residue.length > 0) {
  const chartWrap = document.createElement('div');
  chartWrap.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:6px;min-width:0';

  const chartTitle = document.createElement('div');
  chartTitle.style.cssText = 'font-weight:600;font-size:13px;color:#333';
  chartTitle.textContent = 'Per-residue pLDDT';
  chartWrap.appendChild(chartTitle);

  const CWIDTH = 360, CHEIGHT = 220;
  const MARGIN = { top: 10, right: 10, bottom: 30, left: 36 };
  const plotW = CWIDTH - MARGIN.left - MARGIN.right;
  const plotH = CHEIGHT - MARGIN.top  - MARGIN.bottom;

  const cc = document.createElement('canvas');
  cc.width  = CWIDTH;
  cc.height = CHEIGHT;
  cc.style.cssText = `width:${CWIDTH}px;height:${CHEIGHT}px;border:1px solid #dee2e6;border-radius:4px`;
  const cx = cc.getContext('2d');

  // Background
  cx.fillStyle = '#fff';
  cx.fillRect(0, 0, CWIDTH, CHEIGHT);

  // Grid lines + Y axis
  const yTicks = [0, 25, 50, 70, 90, 100];
  cx.strokeStyle = '#e9ecef';
  cx.lineWidth = 1;
  cx.fillStyle = '#888';
  cx.font = '10px system-ui';
  cx.textAlign = 'right';
  for (const y of yTicks) {
    const yPx = MARGIN.top + plotH - (y / 100) * plotH;
    cx.beginPath();
    cx.moveTo(MARGIN.left, yPx);
    cx.lineTo(MARGIN.left + plotW, yPx);
    cx.stroke();
    cx.fillText(y, MARGIN.left - 4, yPx + 3);
  }

  // pLDDT color bands
  const bands = [[90, 100, 'rgba(0,83,214,0.08)'], [70, 90, 'rgba(101,203,243,0.08)'],
                 [50, 70, 'rgba(255,219,19,0.08)'],  [0,  50, 'rgba(255,125,69,0.08)']];
  for (const [lo, hi, col] of bands) {
    const y1 = MARGIN.top + plotH - (hi / 100) * plotH;
    const y2 = MARGIN.top + plotH - (lo / 100) * plotH;
    cx.fillStyle = col;
    cx.fillRect(MARGIN.left, y1, plotW, y2 - y1);
  }

  // pLDDT line
  function plddt_color(v) {
    if (v >= 90) return '#0053D6';
    if (v >= 70) return '#65CBF3';
    if (v >= 50) return '#FFDB13';
    return '#FF7D45';
  }

  const vals = plddt_per_residue;
  const xStep = plotW / Math.max(vals.length - 1, 1);

  // Filled area under curve (gradient)
  const areaGrad = cx.createLinearGradient(0, MARGIN.top, 0, MARGIN.top + plotH);
  areaGrad.addColorStop(0,   'rgba(52,152,219,0.25)');
  areaGrad.addColorStop(1,   'rgba(52,152,219,0)');
  cx.beginPath();
  cx.moveTo(MARGIN.left, MARGIN.top + plotH);
  for (let i = 0; i < vals.length; i++) {
    const xPx = MARGIN.left + i * xStep;
    const yPx = MARGIN.top  + plotH - (vals[i] / 100) * plotH;
    cx.lineTo(xPx, yPx);
  }
  cx.lineTo(MARGIN.left + (vals.length - 1) * xStep, MARGIN.top + plotH);
  cx.closePath();
  cx.fillStyle = areaGrad;
  cx.fill();

  // Line
  cx.beginPath();
  cx.lineWidth = 1.5;
  cx.strokeStyle = '#3498db';
  for (let i = 0; i < vals.length; i++) {
    const xPx = MARGIN.left + i * xStep;
    const yPx = MARGIN.top  + plotH - (vals[i] / 100) * plotH;
    i === 0 ? cx.moveTo(xPx, yPx) : cx.lineTo(xPx, yPx);
  }
  cx.stroke();

  // X axis label
  cx.fillStyle = '#888';
  cx.font = '10px system-ui';
  cx.textAlign = 'center';
  cx.fillText('Residue position', MARGIN.left + plotW / 2, CHEIGHT - 4);

  chartWrap.appendChild(cc);

  // pLDDT legend
  const plLegend = document.createElement('div');
  plLegend.style.cssText = 'display:flex;gap:12px;font-size:11px;flex-wrap:wrap;';
  plLegend.innerHTML = [
    ['#0053D6','Very high (>90)'], ['#65CBF3','Confident (70-90)'],
    ['#FFDB13','Low (50-70)'],     ['#FF7D45','Very low (<50)']
  ].map(([c,l]) => `<span><span style="color:${c};font-size:14px">●</span> ${l}</span>`).join('');
  chartWrap.appendChild(plLegend);

  mainRow.appendChild(chartWrap);
}

container.appendChild(mainRow);

// ── All-models comparison table ───────────────────────────────────────────────
if (all_models && all_models.length > 1) {
  const tableWrap = document.createElement('div');
  tableWrap.style.cssText = 'display:flex;flex-direction:column;gap:6px';

  const tableTitle = document.createElement('div');
  tableTitle.style.cssText = 'font-weight:600;font-size:13px;color:#333';
  tableTitle.textContent = 'All Models — Ranking';
  tableWrap.appendChild(tableTitle);

  const table = document.createElement('table');
  table.style.cssText = 'border-collapse:collapse;font-size:12px;width:100%';
  table.innerHTML = `
    <thead>
      <tr style="background:#f8f9fa;color:#495057">
        <th style="padding:5px 10px;text-align:left;border-bottom:2px solid #dee2e6">Rank</th>
        <th style="padding:5px 10px;text-align:left;border-bottom:2px solid #dee2e6">Model</th>
        <th style="padding:5px 10px;text-align:right;border-bottom:2px solid #dee2e6">Mean pLDDT</th>
        <th style="padding:5px 10px;text-align:left;border-bottom:2px solid #dee2e6;width:120px"></th>
      </tr>
    </thead>
    <tbody>
      ${all_models.map(m => {
        const best = m.rank === 1;
        const barPct = (m.plddt / 100 * 100).toFixed(0);
        const barColor = m.plddt >= 90 ? '#0053D6' : m.plddt >= 70 ? '#65CBF3' : '#FFDB13';
        return `<tr style="border-bottom:1px solid #f1f3f5;${best ? 'background:#f0fff4' : ''}">
          <td style="padding:5px 10px;font-weight:${best?'700':'400'};color:${best?'#2f9e44':'#333'}">
            ${best ? '★ 1' : m.rank}
          </td>
          <td style="padding:5px 10px;font-family:monospace;color:#555">${m.name.replace(/_pred_0$/, '')}</td>
          <td style="padding:5px 10px;text-align:right;font-weight:600">${m.plddt.toFixed(1)}</td>
          <td style="padding:5px 10px">
            <div style="background:#e9ecef;border-radius:3px;height:8px;overflow:hidden">
              <div style="width:${barPct}%;height:100%;background:${barColor};border-radius:3px"></div>
            </div>
          </td>
        </tr>`;
      }).join('')}
    </tbody>`;
  tableWrap.appendChild(table);
  container.appendChild(tableWrap);
}
