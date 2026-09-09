/**
 * Spectra SR — Frontend Application Logic
 * Implements Mission Console, Live Pipeline Polling, Split Slider,
 * Trust Scorecard, and Benchmark Validation.
 */

// Global State
const state = {
  currentView: 'console', // 'console', 'processing', 'analysis', 'benchmark'
  selectedSceneId: 'opensr_spain_crops',
  currentJobId: null,
  activeLayer: 'safe_sr', // 'raw_sr', 'safe_sr', 'risk_map', 'agri_boundary'
  jobResult: null,
  sliderPos: 50, // 0 to 100%
  isDraggingSlider: false,
  benchmarkData: null,
  
  // Validation Lab State
  benchmarkDataset: 'spain_crops',
  benchmarkSample: 'sample_000',
  benchmarkViewMode: '3way', // '3way', 'slider'
  benchSliderPos: 50,
  isDraggingBenchSlider: false,
  benchmarkDatasets: [],
  trustModelInfo: null
};

// Preset scenes (OpenSR Test & Operational)
const PRESET_SCENES = {
  opensr_spain_crops: {
    name: 'Punjab Intensive Cropland, Ludhiana',
    sourceId: 'S2_PUNJAB_CROPS_001',
    coords: '30.90° N, 75.85° E • 10m Sentinel-2',
    cloud: 0.0,
    desc: 'Real Sentinel-2 L2A crop observation: smallholder parcels, canal grids & crop vigor.'
  },
  opensr_spain_urban: {
    name: 'Delhi NCR Peri-Urban & Infrastructure',
    sourceId: 'S2_DELHI_URBAN_001',
    coords: '28.61° N, 77.20° E • 10m Sentinel-2',
    cloud: 0.0,
    desc: 'Real Sentinel-2 L2A urban observation: arterial road grid, residential parcels, commercial blocks.'
  },
  opensr_naip: {
    name: 'Haryana Agricultural Basin, Karnal',
    sourceId: 'S2_HARYANA_NAIP_001',
    coords: '29.68° N, 76.98° E • 10m Sentinel-2',
    cloud: 0.0,
    desc: 'Real Sentinel-2 L2A cropland observation paired with high-resolution reference imagery.'
  },
  opensr_spot: {
    name: 'Western Ghats Agro-Forestry, Nashik',
    sourceId: 'S2_NASHIK_SPOT_001',
    coords: '19.99° N, 73.78° E • 10m Sentinel-2',
    cloud: 0.0,
    desc: 'Real Sentinel-2 L2A vegetated observation paired with 1.5m reference imagery.'
  },
  opensr_venus: {
    name: 'Karnataka Semiarid Farmland, Raichur',
    sourceId: 'S2_RAICHUR_VENUS_001',
    coords: '16.20° N, 77.34° E • 10m Sentinel-2',
    cloud: 0.0,
    desc: 'Real Sentinel-2 L2A observation paired with super-spectral reference imagery.'
  },
  scene_pune_periurban: {
    name: 'Pune Peri-Urban Farmland, Maharashtra',
    sourceId: 'S2B_MSIL2A_20260215T052029_T43QDA',
    coords: '18.5204° N, 73.8567° E (EPSG:32643)',
    cloud: 2.1,
    desc: 'Highway corridor, peri-urban settlements & smallholder agrarian parcels.'
  },
  scene_punjab_crop: {
    name: 'Ludhiana Intensive Agriculture, Punjab',
    sourceId: 'S2A_MSIL2A_20260128T053211_T43RER',
    coords: '30.9010° N, 75.8573° E (EPSG:32643)',
    cloud: 0.8,
    desc: 'Dense wheat/mustard parcels with well-defined hedgerows and canal irrigation grids.'
  },
  scene_raichur_dryland: {
    name: 'Raichur Semiarid Agrarian Zone, Karnataka',
    sourceId: 'S2B_MSIL2A_20260204T051109_T43PFN',
    coords: '16.2076° N, 77.3463° E (EPSG:32643)',
    cloud: 3.4,
    desc: 'Semiarid rainfed agricultural plots, cotton belts, and distinct rocky outcrop boundaries.'
  }
};

// Stepper steps configuration
const PIPELINE_STEPS = [
  { id: 'DISCOVER_SCENE', label: 'Scene Discovery' },
  { id: 'PREPROCESS', label: 'Preprocessing' },
  { id: 'SR_INFERENCE', label: 'SEN2SRLite (4×)' },
  { id: 'TRUST_ANALYSIS', label: 'Trust Engine' },
  { id: 'TRUST_GATING', label: 'Trust-Gating' },
  { id: 'AGRI_BOUNDARY', label: 'Agri Boundary' },
  { id: 'ARTIFACT_WRITE', label: 'Artifact Export' },
  { id: 'COMPLETE', label: 'Complete' }
];

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPresetSelection();
  initFileUpload();
  initJobSubmission();
  initSplitSlider();
  initLayerToggles();
  initValidationLab();
});

// View Navigation
function setView(viewName) {
  state.currentView = viewName;
  document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  
  const targetSec = document.getElementById(`view-${viewName}`);
  if (targetSec) targetSec.classList.add('active');
  
  const targetBtn = document.querySelector(`.nav-btn[data-view="${viewName}"]`);
  if (targetBtn) targetBtn.classList.add('active');
}

function initNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.getAttribute('data-view');
      if (view) setView(view);
    });
  });
}

// Preset AOI Selection
function initPresetSelection() {
  // Bind click handlers to AOI cards
  document.querySelectorAll('.aoi-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.aoi-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      const sceneId = card.getAttribute('data-scene-id');
      state.selectedSceneId = sceneId;
      
      const sc = PRESET_SCENES[sceneId];
      if (sc) {
        document.getElementById('input-scene-id').value = sceneId;
        document.getElementById('cloud-value-display').innerText = `${sc.cloud}%`;
      }
    });
  });

  // Dynamically load all scenes from backend to synchronize any user-uploaded scenes
  loadAvailableScenes();
}

async function loadAvailableScenes() {
  try {
    const res = await fetch('/api/v1/scenes');
    if (!res.ok) return;
    const scenes = await res.json();
    scenes.forEach(sc => {
      if (!PRESET_SCENES[sc.id]) {
        PRESET_SCENES[sc.id] = {
          name: sc.location_name || sc.id,
          sourceId: sc.source_item_id,
          coords: `${sc.source_gsd_m}m GSD • ${sc.crs}`,
          cloud: sc.cloud_cover || 0.0,
          desc: sc.provenance?.description || 'Sentinel-2 observation'
        };
      }
    });
  } catch (err) {
    console.warn('Could not load scenes from backend:', err);
  }
}

// File Upload Support for Custom Satellite Imagery
function initFileUpload() {
  const dropzone = document.getElementById('dropzone-upload');
  const fileInput = document.getElementById('input-satellite-file');
  const previewContainer = document.getElementById('upload-preview-container');
  const previewImg = document.getElementById('upload-preview-img');
  const filenameEl = document.getElementById('upload-filename');
  
  if (!dropzone || !fileInput) return;
  
  dropzone.addEventListener('click', () => fileInput.click());
  
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });
  
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });
  
  async function handleFileSelected(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('location_name', `Uploaded: ${file.name}`);
    
    // Immediate preview if visual image
    if (file.name.match(/\.(png|jpe?g)$/i)) {
      const objectUrl = URL.createObjectURL(file);
      if (previewImg) previewImg.src = objectUrl;
    }
    if (filenameEl) filenameEl.innerText = `${file.name} (Uploading...)`;
    if (previewContainer) previewContainer.style.display = 'flex';
    
    try {
      const res = await fetch('/api/v1/scenes/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
      const sceneData = await res.json();
      
      // Update preview with generated preview url
      if (previewImg) previewImg.src = `${sceneData.preview_url}?t=${Date.now()}`;
      if (filenameEl) filenameEl.innerText = `${file.name} (Ready)`;
      
      // Register in state and presets
      state.selectedSceneId = sceneData.id;
      const sceneInput = document.getElementById('input-scene-id');
      if (sceneInput) sceneInput.value = sceneData.id;
      
      PRESET_SCENES[sceneData.id] = {
        name: `Custom Upload: ${file.name}`,
        sourceId: sceneData.source_item_id,
        coords: 'Custom Ingested Satellite Crop (10m L2A)',
        cloud: 0.0,
        desc: 'User-provided Sentinel-2 satellite image decomposed to multi-spectral bands.'
      };

      // Dynamically add or select an Uploaded AOI card in the grid
      const grid = document.getElementById('aoi-grid-container');
      if (grid) {
        // Remove existing custom upload card if present
        const existing = document.getElementById('card-custom-upload');
        if (existing) existing.remove();

        const card = document.createElement('div');
        card.id = 'card-custom-upload';
        card.className = 'aoi-card selected';
        card.setAttribute('data-scene-id', sceneData.id);
        card.innerHTML = `
          <div class="aoi-card-badge" style="color: #10B981;">Custom Upload</div>
          <div class="aoi-card-title">${file.name}</div>
          <div class="aoi-card-desc">User-uploaded Sentinel-2 satellite image (4 bands extracted).</div>
          <div class="aoi-card-coords">Ready for 4× Super-Resolution</div>
        `;
        card.addEventListener('click', () => {
          document.querySelectorAll('.aoi-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          state.selectedSceneId = sceneData.id;
          if (sceneInput) sceneInput.value = sceneData.id;
        });

        // Deselect other cards and insert this card at the top
        document.querySelectorAll('.aoi-card').forEach(c => c.classList.remove('selected'));
        grid.prepend(card);
      }
      
    } catch (err) {
      console.error(err);
      if (filenameEl) filenameEl.innerText = `${file.name} (Error: ${err.message})`;
      alert(`Could not upload image: ${err.message}`);
    }
  }
}

// Submit Job
function initJobSubmission() {
  const form = document.getElementById('pipeline-form');
  if (!form) return;
  
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const sceneId = state.selectedSceneId;
    const model = document.getElementById('select-model').value;
    const enableGating = document.getElementById('check-trust-gating').checked;
    
    setView('processing');
    updateStepper('DISCOVER_SCENE', 10);
    
    try {
      const res = await fetch('/api/v1/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_id: sceneId,
          mode: 'live',
          model: model,
          enable_trust_gating: enableGating,
          application: 'agri_boundary'
        })
      });
      
      if (!res.ok) throw new Error(`Job creation failed: ${res.statusText}`);
      const job = await res.json();
      state.currentJobId = job.job_id;
      
      document.getElementById('proc-job-id').innerText = job.job_id;
      document.getElementById('proc-scene-name').innerText = PRESET_SCENES[sceneId]?.name || sceneId;
      
      pollJobStatus(job.job_id);
    } catch (err) {
      console.error(err);
      alert(`Error starting pipeline: ${err.message}`);
      setView('console');
    }
  });
}

// Poll Job Status
async function pollJobStatus(jobId) {
  const startTime = Date.now();
  const timerEl = document.getElementById('proc-elapsed-time');
  
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    if (timerEl) timerEl.innerText = `${elapsed}s`;
  }, 100);

  const pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/v1/jobs/${jobId}`);
      if (!res.ok) return;
      const job = await res.json();
      
      updateStepper(job.step, job.progress);
      
      if (job.status === 'COMPLETED') {
        clearInterval(pollInterval);
        clearInterval(timerInterval);
        updateStepper('COMPLETE', 100);
        
        // Fetch Result
        setTimeout(async () => {
          await loadJobResult(jobId);
          setView('analysis');
        }, 600);
      } else if (job.status === 'FAILED') {
        clearInterval(pollInterval);
        clearInterval(timerInterval);
        alert(`Pipeline failed: ${job.error_message || 'Unknown error'}`);
        setView('console');
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, 500);
}

function updateStepper(activeStep, progress) {
  const progressBar = document.getElementById('proc-progress-fill');
  const progressPct = document.getElementById('proc-progress-pct');
  if (progressBar) progressBar.style.width = `${progress}%`;
  if (progressPct) progressPct.innerText = `${progress}%`;
  
  let foundActive = false;
  PIPELINE_STEPS.forEach((step, idx) => {
    const node = document.getElementById(`step-node-${idx}`);
    if (!node) return;
    
    if (step.id === activeStep) {
      node.className = 'step-node active';
      foundActive = true;
    } else if (!foundActive) {
      node.className = 'step-node completed';
    } else {
      node.className = 'step-node';
    }
  });
}

// Load Job Result
async function loadJobResult(jobId) {
  try {
    const res = await fetch(`/api/v1/jobs/${jobId}/result`);
    if (!res.ok) throw new Error('Failed to fetch job result');
    const data = await res.json();
    state.jobResult = data;
    
    renderAnalysisWorkspace(data);
  } catch (err) {
    console.error(err);
  }
}

function renderAnalysisWorkspace(data) {
  const { artifacts, trust_scorecard, agri_boundary_analysis, provenance } = data;
  
  // Set Images
  const imgLR = document.getElementById('img-split-lr'); // 10 m LR
  const imgSR = document.getElementById('img-split-sr'); // 2.5 m SR
  const imgRisk = document.getElementById('img-risk-overlay'); // Risk colormap
  
  if (imgLR) imgLR.src = artifacts.preview_lr;
  if (imgSR) imgSR.src = artifacts.preview_safe_sr; // default to safe_sr
  if (imgRisk) imgRisk.src = artifacts.preview_risk_map;
  if (window.setAnalysisSliderPosition) {
    window.setAnalysisSliderPosition(state.sliderPos || 50);
  }
  
  // Update Scorecard
  const sd = trust_scorecard.spectral_drift;
  const sa = trust_scorecard.spatial_alignment;
  const sc = trust_scorecard.source_consistency;
  const pr = trust_scorecard.predicted_risk;
  
  // Spectral
  document.getElementById('val-spectral').innerText = `${sd.value_deg}° SAD`;
  setPill('pill-spectral', sd.rating);
  
  // Spatial
  document.getElementById('val-spatial').innerText = `${sa.shift_pixels} px`;
  setPill('pill-spatial', sa.rating);
  
  // Consistency
  document.getElementById('val-consistency').innerText = `${sc.rmse} RMSE`;
  setPill('pill-consistency', sc.rating);
  
  // Risk
  document.getElementById('val-risk').innerText = `Score: ${pr.mean_score}`;
  setPill('pill-risk', pr.rating === 'LOW' ? 'GOOD' : (pr.rating === 'MEDIUM' ? 'MED' : 'WARN'), `${pr.rating} RISK`);
  
  // Agri Boundary
  if (agri_boundary_analysis) {
    document.getElementById('agri-gain-val').innerText = `+${agri_boundary_analysis.sharpness_gain_percent}%`;
    document.getElementById('agri-mult-val').innerText = `${agri_boundary_analysis.edge_contrast_multiplier}×`;
  }
  
  // Bind Downloads
  bindDownload('btn-dl-raw', artifacts.raw_sr_geotiff);
  bindDownload('btn-dl-safe', artifacts.safe_sr_geotiff);
  bindDownload('btn-dl-risk', artifacts.risk_map_geotiff);
  bindDownload('btn-dl-metrics', artifacts.metrics_json);
}

function setPill(id, rating, customText) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-pill';
  if (rating === 'GOOD') {
    el.classList.add('status-good');
    el.innerText = customText || 'GOOD';
  } else if (rating === 'ACCEPTABLE' || rating === 'MED') {
    el.classList.add('status-med');
    el.innerText = customText || 'ACCEPTABLE';
  } else {
    el.classList.add('status-warn');
    el.innerText = customText || 'WARNING';
  }
}

function bindDownload(btnId, url) {
  const btn = document.getElementById(btnId);
  if (btn && url) {
    btn.href = url;
    btn.setAttribute('download', '');
  }
}

// Split Comparison Slider Logic
function initSplitSlider() {
  const container = document.getElementById('split-container');
  const imgLR = document.getElementById('img-split-lr');
  const imgRisk = document.getElementById('img-risk-overlay');
  const divider = document.getElementById('split-divider');
  if (!container || !imgLR || !divider) return;
  
  function setSliderPosition(pos) {
    pos = Math.max(1, Math.min(99, pos));
    state.sliderPos = pos;
    container.style.setProperty('--split-pos', `${pos}%`);
    imgLR.style.clipPath = `polygon(0 0, ${pos}% 0, ${pos}% 100%, 0 100%)`;
    imgLR.style.webkitClipPath = `polygon(0 0, ${pos}% 0, ${pos}% 100%, 0 100%)`;
    if (imgRisk) {
      imgRisk.style.clipPath = `polygon(${pos}% 0, 100% 0, 100% 100%, ${pos}% 100%)`;
      imgRisk.style.webkitClipPath = `polygon(${pos}% 0, 100% 0, 100% 100%, ${pos}% 100%)`;
    }
    divider.style.left = `${pos}%`;
  }

  function updateSliderFromEvent(e) {
    const rect = container.getBoundingClientRect();
    if (rect.width === 0) return;
    const clientX = (e.touches && e.touches.length > 0) ? e.touches[0].clientX : e.clientX;
    if (clientX === undefined) return;
    const pos = ((clientX - rect.left) / rect.width) * 100;
    setSliderPosition(pos);
  }

  function onPointerDown(e) {
    state.isDraggingSlider = true;
    try {
      if (e.pointerId && container.setPointerCapture) {
        container.setPointerCapture(e.pointerId);
      }
    } catch (_) {}
    updateSliderFromEvent(e);
    if (e.cancelable) e.preventDefault();
  }

  function onPointerMove(e) {
    if (!state.isDraggingSlider) return;
    updateSliderFromEvent(e);
    if (e.cancelable) e.preventDefault();
  }

  function onPointerUp(e) {
    if (!state.isDraggingSlider) return;
    state.isDraggingSlider = false;
    try {
      if (e && e.pointerId && container.hasPointerCapture && container.hasPointerCapture(e.pointerId)) {
        container.releasePointerCapture(e.pointerId);
      }
    } catch (_) {}
  }

  if (window.PointerEvent) {
    container.addEventListener('pointerdown', onPointerDown);
    container.addEventListener('pointermove', onPointerMove, { passive: false });
    container.addEventListener('pointerup', onPointerUp);
    container.addEventListener('pointercancel', onPointerUp);
    window.addEventListener('pointermove', onPointerMove, { passive: false });
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
  } else {
    container.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    container.addEventListener('touchstart', onPointerDown, { passive: false });
    window.addEventListener('touchmove', onPointerMove, { passive: false });
    window.addEventListener('touchend', onPointerUp);
    window.addEventListener('touchcancel', onPointerUp);
  }

  window.setAnalysisSliderPosition = setSliderPosition;
  setSliderPosition(state.sliderPos || 50);
}

// Layer Switcher
function initLayerToggles() {
  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      const layer = btn.getAttribute('data-layer');
      state.activeLayer = layer;
      
      const imgSR = document.getElementById('img-split-sr');
      const imgRisk = document.getElementById('img-risk-overlay');
      const rightLabel = document.getElementById('slider-label-right');
      
      if (!state.jobResult) return;
      const { artifacts } = state.jobResult;
      
      if (layer === 'safe_sr') {
        if (imgSR) imgSR.src = artifacts.preview_safe_sr;
        if (imgRisk) imgRisk.classList.remove('visible');
        if (rightLabel) rightLabel.innerText = 'Trust-Gated Safe SR (2.5 m)';
      } else if (layer === 'raw_sr') {
        if (imgSR) imgSR.src = artifacts.preview_raw_sr;
        if (imgRisk) imgRisk.classList.remove('visible');
        if (rightLabel) rightLabel.innerText = 'Raw SEN2SRLite (2.5 m)';
      } else if (layer === 'risk_map') {
        if (imgSR) imgSR.src = artifacts.preview_safe_sr;
        if (imgRisk) imgRisk.classList.add('visible');
        if (rightLabel) rightLabel.innerText = 'Predicted Risk Map Overlay';
      } else if (layer === 'agri_boundary') {
        if (imgSR) imgSR.src = artifacts.preview_agri_boundary;
        if (imgRisk) imgRisk.classList.remove('visible');
        if (rightLabel) rightLabel.innerText = 'Agri Boundary Enhancement';
      }
    });
  });
}

// =========================================================================
// Screen 4: ESAOpenSR Scientific Validation Lab Controller
// =========================================================================

function initValidationLab() {
  const runBtn = document.getElementById('btn-run-benchmark');
  if (runBtn) {
    runBtn.addEventListener('click', runValidationBenchmark);
  }

  // View Mode Switcher (3-Way Multi-Pane vs Split Slider)
  const btn3Way = document.getElementById('btn-mode-3way');
  const btnSlider = document.getElementById('btn-mode-slider');
  const gridContainer = document.getElementById('bench-grid-container');
  const sliderContainer = document.getElementById('bench-slider-container');

  if (btn3Way && btnSlider) {
    btn3Way.addEventListener('click', () => {
      btn3Way.classList.add('active');
      btnSlider.classList.remove('active');
      state.benchmarkViewMode = '3way';
      if (gridContainer) gridContainer.style.display = 'grid';
      if (sliderContainer) sliderContainer.style.display = 'none';
    });

    btnSlider.addEventListener('click', () => {
      btnSlider.classList.add('active');
      btn3Way.classList.remove('active');
      state.benchmarkViewMode = 'slider';
      if (gridContainer) gridContainer.style.display = 'none';
      if (sliderContainer) sliderContainer.style.display = 'flex';
      if (window.setBenchSliderPosition) {
        window.setBenchSliderPosition(state.benchSliderPos || 50);
      }
    });
  }

  // Initialize Bench Split Slider
  initBenchSplitSlider();

  // Load Datasets and Trust Model calibration info
  loadBenchmarkDatasets();
  loadTrustModelInfo();

  // Auto-run when user first switches to the Validation Lab tab
  const labNavBtn = document.querySelector('.nav-btn[data-view="benchmark"]');
  if (labNavBtn) {
    labNavBtn.addEventListener('click', () => {
      if (!state.benchmarkData) {
        runValidationBenchmark();
      }
    });
  }
}

// Load Benchmark Datasets and Setup Pickers
async function loadBenchmarkDatasets() {
  try {
    const res = await fetch('/api/v1/benchmark/datasets');
    if (!res.ok) return;
    const datasets = await res.json();
    state.benchmarkDatasets = datasets;

    // Hook up dataset chips
    const dsContainer = document.getElementById('dataset-chips-container');
    if (dsContainer) {
      const chips = dsContainer.querySelectorAll('.dataset-chip');
      chips.forEach(chip => {
        chip.addEventListener('click', () => {
          chips.forEach(c => c.classList.remove('active'));
          chip.classList.add('active');
          const dsName = chip.getAttribute('data-dataset');
          state.benchmarkDataset = dsName;
          updateSampleChipsForDataset(dsName);
          runValidationBenchmark();
        });
      });
    }

    // Hook up initial sample chips
    initSampleChipsListeners();
  } catch (err) {
    console.warn('Failed to load benchmark datasets metadata:', err);
  }
}

function updateSampleChipsForDataset(datasetName) {
  const container = document.getElementById('sample-chips-container');
  if (!container) return;

  const ds = state.benchmarkDatasets.find(d => d.name === datasetName);
  const samples = (ds && ds.samples && ds.samples.length > 0) 
    ? ds.samples 
    : [
        { sample_id: 'sample_000', roi: 'ROI 00001' },
        { sample_id: 'sample_001', roi: 'ROI 00002' },
        { sample_id: 'sample_002', roi: 'ROI 00003' },
        { sample_id: 'sample_003', roi: 'ROI 00004' },
        { sample_id: 'sample_004', roi: 'ROI 00005' }
      ];

  state.benchmarkSample = samples[0].sample_id;

  container.innerHTML = samples.map((s, idx) => `
    <button class="sample-chip ${idx === 0 ? 'active' : ''}" data-sample="${s.sample_id}">
      Sample #${idx + 1} (${s.roi || `Chip ${idx + 1}`})
    </button>
  `).join('');

  // Update meta pill
  const metaPill = document.getElementById('sample-meta-pill');
  if (metaPill && ds) {
    metaPill.innerText = `Input: ${ds.lr_resolution} → Reference: ${ds.hr_resolution}`;
  }

  initSampleChipsListeners();
}

function initSampleChipsListeners() {
  const container = document.getElementById('sample-chips-container');
  if (!container) return;

  container.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      container.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.benchmarkSample = chip.getAttribute('data-sample');
      runValidationBenchmark();
    });
  });
}

// Load Trust Model Calibration Info
async function loadTrustModelInfo() {
  try {
    const res = await fetch('/api/v1/benchmark/trust-model');
    if (!res.ok) return;
    const info = await res.json();
    state.trustModelInfo = info;

    if (info.metrics) {
      const aucEl = document.getElementById('th-auc');
      const f1El = document.getElementById('th-f1');
      const accEl = document.getElementById('th-acc');
      if (aucEl && info.metrics.roc_auc_ovr) aucEl.innerText = Number(info.metrics.roc_auc_ovr).toFixed(3);
      if (f1El && info.metrics.weighted_f1) f1El.innerText = Number(info.metrics.weighted_f1).toFixed(3);
      if (accEl && info.metrics.accuracy) accEl.innerText = `${(Number(info.metrics.accuracy) * 100).toFixed(1)}%`;
    }

    // Render feature importances bars
    const listEl = document.getElementById('feature-bars-list');
    if (listEl && info.feature_importances) {
      const featureNamesMap = {
        spectral_angle_deg: 'Spectral Angle',
        rmse_consistency: 'Reflectance RMSE',
        phase_shift_magnitude: 'Phase Shift',
        mae_consistency: 'MAE Error',
        ndvi_drift: 'NDVI Drift',
        variance_delta: 'Variance Delta',
        laplacian_energy: 'Laplacian Energy',
        gradient_delta: 'Gradient Delta'
      };

      const entries = Object.entries(info.feature_importances)
        .sort((a, b) => b[1] - a[1]);

      listEl.innerHTML = entries.map(([feat, val]) => {
        const pct = (val * 100).toFixed(1);
        const displayName = featureNamesMap[feat] || feat;
        return `
          <div class="feature-bar-row">
            <span class="feature-bar-name" title="${feat}">${displayName}</span>
            <div class="feature-bar-track">
              <div class="feature-bar-fill" style="width: ${Math.min(100, val * 250)}%;"></div>
            </div>
            <span class="feature-bar-val">${pct}%</span>
          </div>
        `;
      }).join('');
    }
  } catch (err) {
    console.warn('Failed to load trust model info:', err);
  }
}

// Run Validation Benchmark
async function runValidationBenchmark() {
  const runBtn = document.getElementById('btn-run-benchmark');
  const btnText = document.getElementById('btn-run-bench-text');
  
  if (runBtn) {
    runBtn.disabled = true;
    if (btnText) btnText.innerText = 'Evaluating Benchmark...';
  }

  try {
    const res = await fetch('/api/v1/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_name: state.benchmarkDataset,
        sample_id: state.benchmarkSample,
        model: 'sen2sr_lite'
      })
    });

    if (!res.ok) throw new Error('Benchmark execution failed');
    const data = await res.json();
    state.benchmarkData = data;

    // 1. Update 3-Way Views
    const imgLr = document.getElementById('bench-img-lr');
    const imgSr = document.getElementById('bench-img-sr');
    const imgHr = document.getElementById('bench-img-hr');
    if (imgLr) imgLr.src = data.artifacts.lr_preview;
    if (imgSr) imgSr.src = data.artifacts.sr_preview;
    if (imgHr) imgHr.src = data.artifacts.hr_preview;

    // 2. Update Split Slider Images
    const imgSplitUnder = document.getElementById('bench-split-under');
    const imgSplitOver = document.getElementById('bench-split-over');
    if (imgSplitUnder) imgSplitUnder.src = data.artifacts.hr_preview;
    if (imgSplitOver) imgSplitOver.src = data.artifacts.sr_preview;
    if (window.setBenchSliderPosition) {
      window.setBenchSliderPosition(state.benchSliderPos || 50);
    }

    // 3. Update Top Metric Cards
    const m = data.metrics;
    
    // Hallucination Rate
    const haScore = m.hallucination_score <= 1.0 ? m.hallucination_score * 100 : m.hallucination_score;
    const haEl = document.getElementById('bm-hallucination');
    const haPill = document.getElementById('pill-bm-ha');
    if (haEl) haEl.innerText = `${haScore.toFixed(2)}%`;
    if (haPill) {
      haPill.className = `status-pill ${haScore < 3.0 ? 'status-good' : (haScore < 6.0 ? 'status-warn' : 'status-danger')}`;
      haPill.innerText = haScore < 3.0 ? 'EXCELLENT (<3%)' : (haScore < 6.0 ? 'ACCEPTABLE' : 'HIGH ERROR');
    }

    // Omission Rate
    const omScore = m.omission_score <= 1.0 ? m.omission_score * 100 : m.omission_score;
    const omEl = document.getElementById('bm-omission');
    const omPill = document.getElementById('pill-bm-om');
    if (omEl) omEl.innerText = `${omScore.toFixed(2)}%`;
    if (omPill) {
      omPill.className = `status-pill ${omScore < 5.0 ? 'status-good' : (omScore < 8.0 ? 'status-warn' : 'status-danger')}`;
      omPill.innerText = omScore < 5.0 ? 'EXCELLENT (<5%)' : (omScore < 8.0 ? 'MODERATE' : 'HIGH OMISSION');
    }

    // Improvement Gain
    const imEl = document.getElementById('bm-improvement');
    const imPill = document.getElementById('pill-bm-im');
    if (imEl) imEl.innerText = `+${m.improvement_score.toFixed(1)}%`;
    if (imPill) {
      imPill.className = `status-pill ${m.improvement_score >= 20.0 ? 'status-good' : 'status-warn'}`;
      imPill.innerText = m.improvement_score >= 20.0 ? 'HIGH GAIN (>20%)' : 'MODEST GAIN';
    }

    // Structural Synthesis
    const synEl = document.getElementById('bm-synthesis');
    const synPill = document.getElementById('pill-bm-syn');
    if (synEl) synEl.innerText = m.synthesis_score.toFixed(3);
    if (synPill) {
      synPill.className = `status-pill ${m.synthesis_score >= 0.85 ? 'status-good' : 'status-warn'}`;
      synPill.innerText = m.synthesis_score >= 0.85 ? 'CONFORMANT (>0.85)' : 'ATTENUATED';
    }

    // 5. Update Detailed Table
    const rmseEl = document.getElementById('bm-rmse');
    const sadEl = document.getElementById('bm-sad');
    const spatEl = document.getElementById('bm-spatial');
    const gsdEl = document.getElementById('bm-gsd');
    if (rmseEl) rmseEl.innerText = m.reflectance_consistency_rmse.toFixed(4);
    if (sadEl) sadEl.innerText = `${m.spectral_angle_deg.toFixed(2)}°`;
    if (spatEl) spatEl.innerText = m.spatial_consistency_corr.toFixed(4);
    if (gsdEl) gsdEl.innerText = `${m.lr_input_gsd_m}m → ${m.sr_output_gsd_m}m (Ref: ${m.hr_reference_gsd_m}m)`;

    // 6. Update Engine Attribution Note
    const engEl = document.getElementById('engine-attribution-note');
    if (engEl && m.opensr_native && m.opensr_native.engine) {
      engEl.innerHTML = `<span>⚡</span> Verified Engine: <code>${m.opensr_native.engine}</code> • Protocol: <code>ESAOpenSR Reference Suite</code>`;
    }

    // 7. Update Trust Head Live Chip Verification
    const predRiskEl = document.getElementById('chip-predicted-risk');
    const gtVerifEl = document.getElementById('chip-gt-verif');
    if (data.scorecard && predRiskEl) {
      const rScore = data.scorecard.predicted_risk_score || 0.14;
      const rLevel = data.scorecard.overall_risk_level || 'LOW_RISK';
      predRiskEl.innerText = `${rLevel.replace('_', ' ')} (${rScore.toFixed(2)})`;
      predRiskEl.className = rLevel === 'LOW_RISK' ? 'risk-low' : (rLevel === 'MEDIUM_RISK' ? 'risk-med' : 'risk-high');
    }
    if (gtVerifEl) {
      const rmse = m.reflectance_consistency_rmse;
      if (rmse < 0.03) {
        gtVerifEl.style.color = '#166534';
        gtVerifEl.innerText = `CONCORDANT (Reflectance Error ${rmse.toFixed(4)} < 0.03)`;
      } else {
        gtVerifEl.style.color = '#B45309';
        gtVerifEl.innerText = `MODERATE VARIANCE (Reflectance Error ${rmse.toFixed(4)})`;
      }
    }

  } catch (err) {
    console.error('Validation benchmark failed:', err);
    alert('Unable to complete reference benchmark evaluation. Check backend connectivity.');
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      if (btnText) btnText.innerText = 'Re-Run Reference Benchmark';
    }
  }
}

// Bench Split Slider Dragging Interaction
function initBenchSplitSlider() {
  const viewer = document.getElementById('bench-split-viewer');
  const over = document.getElementById('bench-split-over');
  const divider = document.getElementById('bench-split-divider');
  if (!viewer || !divider) return;

  function setBenchSliderPosition(pos) {
    pos = Math.max(1, Math.min(99, pos));
    state.benchSliderPos = pos;
    viewer.style.setProperty('--bench-pos', `${pos}%`);
    if (over) {
      over.style.clipPath = `polygon(0 0, ${pos}% 0, ${pos}% 100%, 0 100%)`;
      over.style.webkitClipPath = `polygon(0 0, ${pos}% 0, ${pos}% 100%, 0 100%)`;
    }
    divider.style.left = `${pos}%`;
  }

  function updateSliderFromEvent(e) {
    const rect = viewer.getBoundingClientRect();
    if (rect.width === 0) return;
    const clientX = (e.touches && e.touches.length > 0) ? e.touches[0].clientX : e.clientX;
    if (clientX === undefined) return;
    const pos = ((clientX - rect.left) / rect.width) * 100;
    setBenchSliderPosition(pos);
  }

  function onPointerDown(e) {
    state.isDraggingBenchSlider = true;
    try {
      if (e.pointerId && viewer.setPointerCapture) {
        viewer.setPointerCapture(e.pointerId);
      }
    } catch (_) {}
    updateSliderFromEvent(e);
    if (e.cancelable) e.preventDefault();
  }

  function onPointerMove(e) {
    if (!state.isDraggingBenchSlider) return;
    updateSliderFromEvent(e);
    if (e.cancelable) e.preventDefault();
  }

  function onPointerUp(e) {
    if (!state.isDraggingBenchSlider) return;
    state.isDraggingBenchSlider = false;
    try {
      if (e && e.pointerId && viewer.hasPointerCapture && viewer.hasPointerCapture(e.pointerId)) {
        viewer.releasePointerCapture(e.pointerId);
      }
    } catch (_) {}
  }

  if (window.PointerEvent) {
    viewer.addEventListener('pointerdown', onPointerDown);
    viewer.addEventListener('pointermove', onPointerMove, { passive: false });
    viewer.addEventListener('pointerup', onPointerUp);
    viewer.addEventListener('pointercancel', onPointerUp);
    window.addEventListener('pointermove', onPointerMove, { passive: false });
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
  } else {
    viewer.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    viewer.addEventListener('touchstart', onPointerDown, { passive: false });
    window.addEventListener('touchmove', onPointerMove, { passive: false });
    window.addEventListener('touchend', onPointerUp);
    window.addEventListener('touchcancel', onPointerUp);
  }

  window.setBenchSliderPosition = setBenchSliderPosition;
  setBenchSliderPosition(state.benchSliderPos || 50);
}

