/* ── GLOBALS ─────────────────────────────────────────────── */
const API_BASE = 'https://loan-risk-predictor.onrender.com';
let predictionHistory = [];

/* ── DOM REFS ────────────────────────────────────────────── */
const form           = document.getElementById('loanForm');
const predictBtn     = document.getElementById('predictBtn');
const resultCard     = document.getElementById('resultCard');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent  = document.getElementById('resultContent');
const resultLoading  = document.getElementById('resultLoading');
const resultError    = document.getElementById('resultError');
const errorMsg       = document.getElementById('errorMsg');
const apiBadge       = document.getElementById('apiBadge');
const historyList    = document.getElementById('historyList');
const historyCount   = document.getElementById('historyCount');

/* ── API HEALTH CHECK ────────────────────────────────────── */
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      apiBadge.textContent = '';
      apiBadge.innerHTML = `<span class="dot"></span> System Online`;
      apiBadge.className = 'api-badge connected';
      if (!data.model_loaded) {
        apiBadge.innerHTML = `<span class="dot"></span> System Error`;
        apiBadge.className = 'api-badge error';
      }
    } else {
      throw new Error('not ok');
    }
  } catch {
    apiBadge.innerHTML = `<span class="dot"></span> System Offline`;
    apiBadge.className = 'api-badge error';
  }
}

checkApiHealth();
setInterval(checkApiHealth, 15000);

/* ── THEME TOGGLE ────────────────────────────────────────── */
const themeToggle = document.getElementById('themeToggle');
const iconMoon = document.querySelector('.icon-moon');
const iconSun = document.querySelector('.icon-sun');

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    if (document.body.classList.contains('light-theme')) {
      iconMoon.classList.add('hidden');
      iconSun.classList.remove('hidden');
    } else {
      iconMoon.classList.remove('hidden');
      iconSun.classList.add('hidden');
    }
  });
}

/* ── TOGGLE VALUES ───────────────────────────────────────── */
function bindToggle(id, valId) {
  const el = document.getElementById(id);
  const vEl = document.getElementById(valId);
  el.addEventListener('change', () => { vEl.textContent = el.checked ? 'Yes' : 'No'; });
}
bindToggle('has_cosigner', 'cosignerVal');
bindToggle('has_dependents', 'dependentsVal');
bindToggle('has_mortgage', 'mortgageVal');

/* ── QUICK FILL ──────────────────────────────────────────── */
const DEMOS = {
  low: {
    age:18, income:120000, loan_amount:5000, credit_score:820,
    loan_term:12, interest_rate:4.5, dti_ratio:0.12,
    months_employed:60, num_credit_lines:8,
    loan_purpose:'Auto', employment_type:'Full-time',
    education:"Bachelor's", marital_status:'Married',
    has_cosigner:true, has_dependents:false, has_mortgage:false,
  },
  medium: {
    age:35, income:55000, loan_amount:18000, credit_score:650,
    loan_term:36, interest_rate:13.0, dti_ratio:0.38,
    months_employed:18, num_credit_lines:3,
    loan_purpose:'Business', employment_type:'Full-time',
    education:'High School', marital_status:'Single',
    has_cosigner:false, has_dependents:true, has_mortgage:false,
  },
  high: {
    age:52, income:28000, loan_amount:45000, credit_score:490,
    loan_term:60, interest_rate:24.0, dti_ratio:0.72,
    months_employed:3, num_credit_lines:1,
    loan_purpose:'Other', employment_type:'Part-time',
    education:'High School', marital_status:'Divorced',
    has_cosigner:false, has_dependents:true, has_mortgage:true,
  },
};

function fillDemo(level) {
  const d = DEMOS[level];
  for (const [k, v] of Object.entries(d)) {
    const el = document.getElementById(k);
    if (!el) continue;
    if (el.type === 'checkbox') {
      el.checked = v;
      el.dispatchEvent(new Event('change'));
    } else {
      el.value = v;
    }
  }
}

function clearForm() {
  form.reset();
  ['cosignerVal','dependentsVal','mortgageVal'].forEach(id => {
    document.getElementById(id).textContent = 'No';
  });
  showState('placeholder');
}

/* ── SHOW STATES ─────────────────────────────────────────── */
function showState(state) {
  resultPlaceholder.classList.add('hidden');
  resultContent.classList.add('hidden');
  resultLoading.classList.add('hidden');
  resultError.classList.add('hidden');
  if (state === 'placeholder') resultPlaceholder.classList.remove('hidden');
  else if (state === 'content')  resultContent.classList.remove('hidden');
  else if (state === 'loading')  resultLoading.classList.remove('hidden');
  else if (state === 'error')    resultError.classList.remove('hidden');
}

/* ── BUILD PAYLOAD ───────────────────────────────────────── */
function buildPayload() {
  const g = id => document.getElementById(id);
  return {
    age:             parseInt(g('age').value),
    income:          parseFloat(g('income').value),
    loan_amount:     parseFloat(g('loan_amount').value),
    credit_score:    parseInt(g('credit_score').value),
    loan_term:       parseInt(g('loan_term').value),
    interest_rate:   parseFloat(g('interest_rate').value),
    dti_ratio:       parseFloat(g('dti_ratio').value),
    months_employed: parseInt(g('months_employed').value),
    num_credit_lines:parseInt(g('num_credit_lines').value),
    loan_purpose:    g('loan_purpose').value,
    employment_type: g('employment_type').value,
    education:       g('education').value,
    marital_status:  g('marital_status').value,
    has_cosigner:    g('has_cosigner').checked ? 'Yes' : 'No',
    has_dependents:  g('has_dependents').checked ? 'Yes' : 'No',
    has_mortgage:    g('has_mortgage').checked ? 'Yes' : 'No',
  };
}

/* ── RENDER RESULT ───────────────────────────────────────── */
function renderResult(data, payload) {
  const prob   = data.probability;   // 0-1
  const risk   = data.risk;          // Low / Medium / High
  const action = data.action;

  // Risk badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = risk + ' Risk';
  badge.className = `risk-badge risk-${risk}`;

  // Probability number (display as percentage 0-100)
  const pct = (prob * 100).toFixed(1);
  document.getElementById('probNumber').textContent = pct;

  // Bar fill colour
  const fill = document.getElementById('probBarFill');
  const colour = risk === 'High' ? 'var(--high)' : risk === 'Medium' ? 'var(--medium)' : 'var(--low)';
  fill.style.width = `${Math.min(prob * 100, 100)}%`;
  fill.style.background = `linear-gradient(90deg, ${colour}88, ${colour})`;

  // Action box
  const icons = { Low:'📧', Medium:'⚠️', High:'🚨' };
  document.getElementById('actionIcon').textContent = icons[risk] || '📋';
  document.getElementById('actionDetail').textContent = action;
  const actionBox = document.getElementById('actionBox');
  actionBox.style.borderColor = risk === 'High' ? 'rgba(239,68,68,0.3)' : risk === 'Medium' ? 'rgba(245,158,11,0.3)' : 'rgba(34,197,94,0.3)';

  // Input summary
  const s = data.input_summary || {};
  document.getElementById('inputSummary').innerHTML = [
    { k:'Age', v: payload.age + ' yrs' },
    { k:'Income', v:'$' + Number(payload.income).toLocaleString() },
    { k:'Loan Amount', v:'$' + Number(payload.loan_amount).toLocaleString() },
    { k:'Credit Score', v: payload.credit_score },
    { k:'DTI Ratio', v: payload.dti_ratio },
    { k:'Employment', v: payload.employment_type },
  ].map(item => `
    <div class="summary-item">
      <div class="summary-key">${item.k}</div>
      <div class="summary-val">${item.v}</div>
    </div>`).join('');


  showState('content');

  // Add to history
  addHistory({ prob, risk, action, payload, ts: new Date() });
}

/* ── HISTORY ─────────────────────────────────────────────── */
function addHistory(entry) {
  predictionHistory.unshift(entry);
  if (predictionHistory.length > 20) predictionHistory.pop();
  renderHistory();
}

function renderHistory() {
  if (!predictionHistory.length) {
    historyList.innerHTML = '<div class="history-empty">Your past predictions will appear here.</div>';
    historyCount.textContent = 'No predictions yet';
    return;
  }
  historyCount.textContent = `${predictionHistory.length} prediction${predictionHistory.length > 1 ? 's' : ''}`;
  historyList.innerHTML = predictionHistory.map((e, i) => {
    const colour = e.risk === 'High' ? 'risk-HIGH' : e.risk === 'Medium' ? 'risk-MEDIUM' : 'risk-LOW';
    const timeStr = e.ts.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
    const icons = { Low:'📧', Medium:'⚠️', High:'🚨' };
    return `
      <div class="history-item" onclick="replayHistory(${i})">
        <div class="hi-badge ${colour}">${e.risk}</div>
        <div class="hi-prob">${(e.prob*100).toFixed(1)}%</div>
        <div class="hi-info">
          <div>${icons[e.risk] || ''} ${e.action.split(' - ')[1] || e.action}</div>
          <div class="hi-action">Age ${e.payload.age} · $${Number(e.payload.loan_amount).toLocaleString()} · Score ${e.payload.credit_score}</div>
        </div>
        <div class="hi-time">${timeStr}</div>
      </div>`;
  }).join('');
}

function replayHistory(i) {
  const e = predictionHistory[i];
  if (!e) return;
  renderResult({ probability: e.prob, risk: e.risk, action: e.action, feature_count: null }, e.payload);
}

function clearHistory() {
  predictionHistory = [];
  renderHistory();
}

/* ── FORM SUBMIT ─────────────────────────────────────────── */
form.addEventListener('submit', async (evt) => {
  evt.preventDefault();

  let valid = true;
  form.querySelectorAll('.field').forEach(f => f.classList.remove('error'));

  // Basic field validation
  form.querySelectorAll('input[required], select[required]').forEach(el => {
    const field = el.closest('.field');
    if (!el.value && el.value !== '0') {
      if (field) field.classList.add('error');
      valid = false;
    }
  });
  if (!valid) return;

  predictBtn.disabled = true;
  predictBtn.classList.add('loading');
  predictBtn.querySelector('.btn-text').textContent = 'Analysing…';
  showState('loading');

  const payload = buildPayload();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data, payload);

  } catch (err) {
    errorMsg.textContent = err.message || 'Could not reach the system.';
    showState('error');
  } finally {
    predictBtn.disabled = false;
    predictBtn.classList.remove('loading');
    predictBtn.querySelector('.btn-text').textContent = 'Analyse Risk';
  }
});
