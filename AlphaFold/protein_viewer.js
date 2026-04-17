// Parse PDB and extract pLDDT scores
function parsePDB(pdbText) {
  const lines = pdbText.split('\n');
  const plddt = [];
  const residues = [];
  let lastResidue = null;

  for (const line of lines) {
    if (line.startsWith('ATOM') && line.substring(13, 15).trim() === 'CA') {
      const resNum = parseInt(line.substring(22, 26).trim());
      const bfactor = parseFloat(line.substring(60, 66).trim());

      if (resNum !== lastResidue) {
        residues.push(resNum);
        plddt.push(bfactor);
        lastResidue = resNum;
      }
    }
  }
  return { residues, plddt };
}

// Decode base64 PDB data
function decodePDB(base64Data) {
  try {
    return atob(base64Data);
  } catch(e) {
    console.error('Failed to decode base64:', e);
    return null;
  }
}

// Colors for each rank
const rankColors = [
  { border: '#3498db', bg: 'rgba(52, 152, 219, 0.2)' },
  { border: '#e67e22', bg: 'rgba(230, 126, 34, 0.2)' },
  { border: '#2ecc71', bg: 'rgba(46, 204, 113, 0.2)' },
  { border: '#e74c3c', bg: 'rgba(231, 76, 60, 0.2)' },
  { border: '#9b59b6', bg: 'rgba(155, 89, 182, 0.2)' }
];

// Parse input data
const ranksData = {};
if (typeof data === 'object') {
  for (let i = 1; i <= 5; i++) {
    const rankKey = `rank_${i}`;
    if (data[rankKey]) {
      const decoded = decodePDB(data[rankKey]);
      if (decoded && decoded.length > 100) {
        ranksData[rankKey] = decoded;
      }
    }
  }
}

console.log('Loaded ranks:', Object.keys(ranksData));

if (Object.keys(ranksData).length === 0) {
  container.innerHTML = '<div style="padding: 20px; color: red;">Error: No valid PDB data received</div>';
  throw new Error('No valid ranks');
}

// Create side-by-side layout
const mainLayout = document.createElement('div');
mainLayout.style.cssText = 'display: flex; gap: 20px; width: 100%; height: 100%;';

// Left side - 3D Structure
const leftPanel = document.createElement('div');
leftPanel.style.cssText = 'flex: 1; display: flex; flex-direction: column;';

const leftHeader = document.createElement('div');
leftHeader.style.cssText = 'display: flex; align-items: center; gap: 15px; padding: 10px 0; border-bottom: 1px solid #dee2e6; margin-bottom: 10px;';
leftHeader.innerHTML = `
  <span style="font-weight: 600; font-size: 14px;">3D Structure</span>
  ${Object.keys(ranksData).length > 1 ? `
    <span style="color: #6c757d; font-size: 13px;">Rank:</span>
    <select id="rank-selector" style="padding: 4px 8px; border: 1px solid #ced4da; border-radius: 4px; font-size: 13px;">
      ${Object.keys(ranksData).map((rk, idx) => `<option value="${rk}">Rank ${idx + 1}</option>`).join('')}
    </select>
  ` : ''}
`;
leftPanel.appendChild(leftHeader);

const viewer3D = document.createElement('div');
viewer3D.id = 'viewer-3d';
viewer3D.style.cssText = 'flex: 1; position: relative; border: 1px solid #dee2e6; border-radius: 4px; background: white;';
leftPanel.appendChild(viewer3D);

const leftFooter = document.createElement('div');
leftFooter.style.cssText = 'margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; text-align: center;';
leftFooter.innerHTML = '<span id="residue-count">Loading...</span> colored by pIDDT';
leftPanel.appendChild(leftFooter);

const leftLegend = document.createElement('div');
leftLegend.style.cssText = 'display: flex; justify-content: center; gap: 15px; margin-top: 8px; font-size: 12px;';
leftLegend.innerHTML = `
  <span><span style="color: #0053D6; font-size: 14px;">●</span> >90</span>
  <span><span style="color: #65CBF3; font-size: 14px;">●</span> 70-90</span>
  <span><span style="color: #FFDB13; font-size: 14px;">●</span> 50-70</span>
  <span><span style="color: #FF7D45; font-size: 14px;">●</span> <50</span>
`;
leftPanel.appendChild(leftLegend);

// Right side - Plot
const rightPanel = document.createElement('div');
rightPanel.style.cssText = 'flex: 1; display: flex; flex-direction: column;';

const rightHeader = document.createElement('div');
rightHeader.style.cssText = 'padding: 10px 0; border-bottom: 1px solid #dee2e6; margin-bottom: 10px;';
rightHeader.innerHTML = `
  <div style="font-weight: 600; font-size: 14px;">Predicted lDDT per position</div>
  <div style="color: #6c757d; font-size: 12px; margin-top: 2px;">All ${Object.keys(ranksData).length} ranked models</div>
`;
rightPanel.appendChild(rightHeader);

const plotContainer = document.createElement('div');
plotContainer.style.cssText = 'flex: 1; position: relative;';
const canvas = document.createElement('canvas');
canvas.id = 'plddt-chart';
plotContainer.appendChild(canvas);
rightPanel.appendChild(plotContainer);

// Add panels to main layout
mainLayout.appendChild(leftPanel);
mainLayout.appendChild(rightPanel);
container.appendChild(mainLayout);

// Add note at bottom
const note = document.createElement('div');
note.style.cssText = 'margin-top: 15px; padding: 10px; background: #e7f3ff; border-left: 3px solid #2196F3; font-size: 12px; color: #555;';
note.innerHTML = '<strong>Note:</strong> PDB files retrieved via retriever script. Close model agreement indicates high prediction confidence.';
container.appendChild(note);

// Initialize NGL Stage
const stage = new NGL.Stage(viewer3D, { backgroundColor: 'white' });
let currentComponent = null;

// Function to load a specific rank
function loadRank(rankName) {
  const pdbText = ranksData[rankName];
  if (!pdbText) {
    console.error('Rank not found:', rankName);
    return;
  }

  console.log('Loading rank:', rankName, 'PDB length:', pdbText.length);

  if (currentComponent) {
    stage.removeComponent(currentComponent);
  }

  const dataUrl = 'data:text/plain;base64,' + btoa(pdbText);

  stage.loadFile(dataUrl, { ext: 'pdb' }).then(component => {
    console.log('Rank loaded successfully!');
    currentComponent = component;

    component.addRepresentation('cartoon', {
      colorScheme: 'bfactor',
      colorScale: ['#FF7D45', '#FFDB13', '#65CBF3', '#0053D6'],
      colorDomain: [0, 100]
    });

    component.autoView();
    stage.setParameters({ clipDist: 0 });

    // Update residue count
    const { residues } = parsePDB(pdbText);
    document.getElementById('residue-count').textContent = `${residues.length} residues`;
  }).catch(error => {
    console.error('Error loading rank:', error);
    viewer3D.innerHTML = '<div style="padding: 20px; color: red;">Error loading structure: ' + error.message + '</div>';
  });
}

// Load initial rank
setTimeout(() => loadRank('rank_1'), 100);

// Rank selector change handler
if (document.getElementById('rank-selector')) {
  document.getElementById('rank-selector').addEventListener('change', (e) => {
    loadRank(e.target.value);
  });
}

// Parse all ranks for plotting
const allRanks = {};
let commonResidues = null;

for (const [rankName, pdbText] of Object.entries(ranksData)) {
  const { residues, plddt } = parsePDB(pdbText);
  allRanks[rankName] = { residues, plddt };
  if (!commonResidues) commonResidues = residues;
}

console.log('Parsed ranks for plotting:', Object.keys(allRanks));

// Create multi-rank pIDDT plot
const datasets = [];
let rankIndex = 0;

for (const [rankName, rankData] of Object.entries(allRanks)) {
  datasets.push({
    label: 'R' + (rankIndex + 1),
    data: rankData.plddt,
    borderColor: rankColors[rankIndex].border,
    backgroundColor: rankColors[rankIndex].bg,
    borderWidth: 2,
    pointRadius: 0,
    fill: false,
    tension: 0.1
  });
  rankIndex++;
}

const ctx = canvas.getContext('2d');
const plddtChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: commonResidues,
    datasets: datasets
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: false
      },
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          boxWidth: 12,
          padding: 10,
          font: { size: 11 }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Positions',
          font: { size: 12 }
        },
        ticks: {
          font: { size: 10 }
        }
      },
      y: {
        title: {
          display: true,
          text: 'pIDDT',
          font: { size: 12 }
        },
        min: 0,
        max: 100,
        ticks: {
          font: { size: 10 }
        }
      }
    }
  }
});

window.addEventListener('resize', () => {
  stage.handleResize();
  plddtChart.resize();
});
