/* ═══════════════════════════════════════════════════════════
   Quant MVP v6.0 — Trading Terminal JavaScript v3
   ═══════════════════════════════════════════════════════════ */

// ── Globals ────────────────────────────────────────────
let mainChart, subChart, equityChart;
let candleSeries, sma20Series, sma60Series, bbUpperSeries, bbLowerSeries, costLinePriceLine;
let volSeries, volMaSeries, rsiSeries, macdLineSeries, macdSignalSeries, macdHistSeries;
let equitySeries;
let currentSymbol = null;
let currentSub = 'volume';
let portfolioData = null;
let candleCache = {};

// ── Helpers ────────────────────────────────────────────
function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function pnlClass(v) { return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : ''; }
function pnlSign(v) { return v > 0 ? `+$${Math.abs(v).toLocaleString()}` : v < 0 ? `-$${Math.abs(v).toLocaleString()}` : '$0'; }
function pctSign(v) { return v > 0 ? `+${v.toFixed(2)}%` : `${v.toFixed(2)}%`; }

function colorVal(el, v) {
  if (!el) return;
  el.classList.remove('pnl-pos', 'pnl-neg');
  if (v > 0) el.classList.add('pnl-pos');
  else if (v < 0) el.classList.add('pnl-neg');
}

// ── Chart Init ─────────────────────────────────────────
function initCharts() {
  const chartOpts = {
    layout: { background: { color: 'transparent' }, textColor: '#6b7a99', fontFamily: "'JetBrains Mono', monospace", fontSize: 10 },
    grid: { vertLines: { color: 'rgba(68,138,255,0.04)' }, horzLines: { color: 'rgba(68,138,255,0.04)' } },
    crosshair: { mode: 0, vertLine: { color: 'rgba(68,138,255,0.3)', width: 1, style: 2 }, horzLine: { color: 'rgba(68,138,255,0.3)', width: 1, style: 2 } },
    timeScale: { borderColor: 'rgba(68,138,255,0.1)', timeVisible: false },
    rightPriceScale: { borderColor: 'rgba(68,138,255,0.1)' },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true },
  };

  // Main chart
  const mainEl = document.getElementById('chartMain');
  mainChart = LightweightCharts.createChart(mainEl, { ...chartOpts, width: mainEl.clientWidth, height: mainEl.clientHeight });
  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#00e676', downColor: '#ff5252', borderUpColor: '#00e676', borderDownColor: '#ff5252',
    wickUpColor: '#00e676', wickDownColor: '#ff5252',
  });
  sma20Series = mainChart.addLineSeries({ color: '#448aff', lineWidth: 1, title: 'SMA20', priceLineVisible: false, lastValueVisible: false });
  sma60Series = mainChart.addLineSeries({ color: '#ffd740', lineWidth: 1, title: 'SMA60', priceLineVisible: false, lastValueVisible: false });
  bbUpperSeries = mainChart.addLineSeries({ color: 'rgba(156,39,176,0.4)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
  bbLowerSeries = mainChart.addLineSeries({ color: 'rgba(156,39,176,0.4)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });

  // Sub chart
  const subEl = document.getElementById('chartSub');
  subChart = LightweightCharts.createChart(subEl, {
    ...chartOpts, width: subEl.clientWidth, height: subEl.clientHeight,
    rightPriceScale: { borderColor: 'rgba(68,138,255,0.1)', scaleMargins: { top: 0.1, bottom: 0.05 } },
  });

  // Volume (default)
  volSeries = subChart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '', lastValueVisible: false, priceLineVisible: false });
  volMaSeries = subChart.addLineSeries({ color: '#ffd740', lineWidth: 1, priceScaleId: '', priceLineVisible: false, lastValueVisible: false });

  // RSI (hidden initially)
  rsiSeries = subChart.addLineSeries({ color: '#448aff', lineWidth: 1.5, priceScaleId: 'rsi', priceLineVisible: false, lastValueVisible: false, visible: false });

  // MACD (hidden initially)
  macdLineSeries = subChart.addLineSeries({ color: '#448aff', lineWidth: 1.5, priceScaleId: 'macd', priceLineVisible: false, lastValueVisible: false, visible: false });
  macdSignalSeries = subChart.addLineSeries({ color: '#ff9800', lineWidth: 1, lineStyle: 2, priceScaleId: 'macd', priceLineVisible: false, lastValueVisible: false, visible: false });
  macdHistSeries = subChart.addHistogramSeries({ priceScaleId: 'macd', priceLineVisible: false, lastValueVisible: false, visible: false });

  // Equity chart
  const eqEl = document.getElementById('equityChart');
  if (eqEl) {
    equityChart = LightweightCharts.createChart(eqEl, {
      ...chartOpts, width: eqEl.clientWidth, height: 90,
      rightPriceScale: { borderColor: 'rgba(68,138,255,0.1)', scaleMargins: { top: 0.1, bottom: 0.05 } },
      timeScale: { borderColor: 'rgba(68,138,255,0.1)', visible: true },
    });
    equitySeries = equityChart.addAreaSeries({
      lineColor: '#448aff', topColor: 'rgba(68,138,255,0.3)', bottomColor: 'rgba(68,138,255,0.02)',
      lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true,
    });
  }

  // Resize handler
  const ro = new ResizeObserver(() => {
    mainChart.resize(mainEl.clientWidth, mainEl.clientHeight);
    subChart.resize(subEl.clientWidth, subEl.clientHeight);
    if (equityChart && eqEl) equityChart.resize(eqEl.clientWidth, 90);
  });
  ro.observe(mainEl);
  ro.observe(subEl);
  if (eqEl) ro.observe(eqEl);
}

// ── Sub Chart Switching ────────────────────────────────
function switchSub(btn) {
  document.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentSub = btn.dataset.sub;
  const label = document.getElementById('subLabel');
  if (label) label.textContent = currentSub.toUpperCase();

  // Toggle visibility
  const isVol = currentSub === 'volume';
  const isRsi = currentSub === 'rsi';
  const isMacd = currentSub === 'macd';

  volSeries.applyOptions({ visible: isVol });
  volMaSeries.applyOptions({ visible: isVol });
  rsiSeries.applyOptions({ visible: isRsi });
  macdLineSeries.applyOptions({ visible: isMacd });
  macdSignalSeries.applyOptions({ visible: isMacd });
  macdHistSeries.applyOptions({ visible: isMacd });

  subChart.timeScale().fitContent();
}

// ── Load Symbols ───────────────────────────────────────
async function loadSymbols() {
  try {
    const resp = await fetch('/api/symbols');
    const data = await resp.json();
    const sel = document.getElementById('symbolSelect');
    if (!sel || !data.symbols) return;

    // Put held symbols first
    const held = portfolioData ? portfolioData.positions.map(p => p.symbol) : [];
    const sortedSymbols = [...held.filter(s => data.symbols.includes(s)), ...data.symbols.filter(s => !held.includes(s))];

    sel.innerHTML = sortedSymbols.map(s => {
      const isHeld = held.includes(s);
      return `<option value="${s}" ${isHeld ? 'class="held"' : ''}>${isHeld ? '● ' : ''}${s}</option>`;
    }).join('');

    if (data.date) setEl('headerDate', data.date);

    // Default to first held symbol or first available
    const defaultSym = held[0] || sortedSymbols[0];
    if (defaultSym) {
      sel.value = defaultSym;
      loadCandles(defaultSym);
    }

    sel.onchange = () => loadCandles(sel.value);
  } catch (e) {
    console.error('loadSymbols:', e);
  }
}

// ── Load Candles + Indicators ──────────────────────────
async function loadCandles(symbol) {
  currentSymbol = symbol;
  try {
    const resp = await fetch(`/api/candles/${symbol}?days=250`);
    const data = await resp.json();
    candleCache[symbol] = data;

    if (!data.candles || data.candles.length === 0) return;

    const candles = data.candles;
    const last = candles[candles.length - 1];

    // Price label
    const pl = document.getElementById('priceLabel');
    if (pl) {
      const chg = last.close && candles.length > 1 ? ((last.close / candles[candles.length - 2].close - 1) * 100) : 0;
      pl.innerHTML = `<span style="color:var(--text-bright)">${symbol}</span> <span style="color:${chg >= 0 ? 'var(--green)' : 'var(--red)'}">$${last.close} ${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</span>`;
    }

    // Candlestick data
    candleSeries.setData(candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));

    // SMA overlays
    sma20Series.setData(candles.filter(c => c.sma20).map(c => ({ time: c.time, value: c.sma20 })));
    sma60Series.setData(candles.filter(c => c.sma60).map(c => ({ time: c.time, value: c.sma60 })));

    // Bollinger Bands
    bbUpperSeries.setData(candles.filter(c => c.bb_upper).map(c => ({ time: c.time, value: c.bb_upper })));
    bbLowerSeries.setData(candles.filter(c => c.bb_lower).map(c => ({ time: c.time, value: c.bb_lower })));

    // Cost line
    if (costLinePriceLine) { try { candleSeries.removePriceLine(costLinePriceLine); } catch(e) {} }
    if (data.cost_line) {
      costLinePriceLine = candleSeries.createPriceLine({
        price: data.cost_line,
        color: '#ff9800',
        lineWidth: 1,
        lineStyle: 2,  // dashed
        axisLabelVisible: true,
        title: `成本 $${data.cost_line}`,
      });
    }

    // Trade markers on candlestick chart
    if (data.trades && data.trades.length > 0) {
      const markers = data.trades.map(t => ({
        time: t.time,
        position: t.action === 'BUY' ? 'belowBar' : 'aboveBar',
        color: t.action === 'BUY' ? '#00e676' : '#ff5252',
        shape: t.action === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: `${t.action} ${t.qty}@${t.price}`,
      })).sort((a, b) => a.time < b.time ? -1 : 1);
      candleSeries.setMarkers(markers);
    } else {
      candleSeries.setMarkers([]);
    }

    // ── Sub chart data ──
    // Volume
    volSeries.setData(candles.map(c => ({
      time: c.time, value: c.volume,
      color: c.close >= c.open ? 'rgba(0,230,118,0.35)' : 'rgba(255,82,82,0.35)',
    })));
    volMaSeries.setData(candles.filter(c => c.vol_ma).map(c => ({ time: c.time, value: c.vol_ma })));

    // RSI
    rsiSeries.setData(candles.filter(c => c.rsi != null).map(c => ({ time: c.time, value: c.rsi })));

    // MACD
    macdLineSeries.setData(candles.filter(c => c.macd != null).map(c => ({ time: c.time, value: c.macd })));
    macdSignalSeries.setData(candles.filter(c => c.macd_signal != null).map(c => ({ time: c.time, value: c.macd_signal })));
    macdHistSeries.setData(candles.filter(c => c.macd_hist != null).map(c => ({
      time: c.time, value: c.macd_hist,
      color: c.macd_hist >= 0 ? 'rgba(0,230,118,0.5)' : 'rgba(255,82,82,0.5)',
    })));

    mainChart.timeScale().fitContent();
    subChart.timeScale().fitContent();

    // Load decision chain for this symbol
    loadDecisionChain(symbol);
  } catch (e) {
    console.error('loadCandles:', e);
  }
}

// ── Load Decision Chain ────────────────────────────────
async function loadDecisionChain(symbol) {
  const container = document.getElementById('decisionChain');
  const symLabel = document.getElementById('decisionSymbol');
  const factorsEl = document.getElementById('keyFactors');
  if (!container) return;

  if (symLabel) symLabel.textContent = symbol;

  try {
    const resp = await fetch(`/api/decision/${symbol}`);
    const data = await resp.json();

    // Build pipeline nodes
    const nodes = [];

    // Model nodes
    const modelOrder = [
      { key: 'sma', label: 'SMA' },
      { key: 'momentum', label: 'MOM' },
      { key: 'mean_reversion', label: 'MR' },
    ];

    for (const m of modelOrder) {
      const md = data.models[m.key] || {};
      const side = md.side || 0;
      let badgeClass = 'inactive';
      let badgeText = '—';
      let detail = '';

      if (side > 0) { badgeClass = 'buy'; badgeText = '↑ BUY'; }
      else if (side < 0) { badgeClass = 'sell'; badgeText = '↓ SELL'; }

      if (md.prob != null) detail = `P=${(md.prob * 100).toFixed(0)}%`;
      if (md.realized_vol != null) detail += ` σ=${(md.realized_vol * 100).toFixed(1)}%`;

      nodes.push({ name: m.label, badgeClass, badgeText, detail });
    }

    // Consensus node
    const cons = data.consensus || {};
    let consBadge = 'inactive', consText = '—';
    if (cons.passed) {
      consBadge = cons.synth_side > 0 ? 'buy' : 'sell';
      consText = cons.synth_side > 0 ? '✓ BUY' : '✓ SELL';
    } else {
      consBadge = 'neutral';
      consText = '✗ 未过';
    }
    nodes.push({ name: '合议', badgeClass: consBadge, badgeText: consText, detail: `vote=${cons.vote || 0}` });

    // Oracle node
    const orc = data.oracle || {};
    let orcBadge = 'inactive', orcText = 'N/A';
    if (orc.action === 'approve') { orcBadge = 'buy'; orcText = '✓ 批准'; }
    else if (orc.action === 'veto') { orcBadge = 'sell'; orcText = '✗ 否决'; }
    else if (orc.action === 'neutral') { orcBadge = 'neutral'; orcText = '~ 中性'; }
    const orcDetail = orc.pred_ret != null ? `ret=${(orc.pred_ret * 100).toFixed(1)}%` : '';
    nodes.push({ name: 'Kronos', badgeClass: orcBadge, badgeText: orcText, detail: orcDetail });

    // Final weight node
    const fw = data.final_weight || 0;
    const fwBadge = fw > 0 ? 'buy' : fw < 0 ? 'sell' : 'inactive';
    const fwText = fw !== 0 ? `${(fw * 100).toFixed(1)}%` : '0%';
    nodes.push({ name: '权重', badgeClass: fwBadge, badgeText: fwText, detail: '' });

    // Render
    let html = '';
    nodes.forEach((n, i) => {
      if (i > 0) {
        const prevPassed = nodes[i - 1].badgeClass === 'buy' || nodes[i - 1].badgeClass === 'sell';
        html += `<div class="chain-arrow ${prevPassed ? 'pass' : 'fail'}">→</div>`;
      }
      html += `<div class="chain-node">
        <div class="chain-node-name">${n.name}</div>
        <div class="chain-node-badge ${n.badgeClass}">${n.badgeText}</div>
        ${n.detail ? `<div class="chain-node-detail">${n.detail}</div>` : ''}
      </div>`;
    });
    container.innerHTML = html;

    // Key factors
    if (factorsEl && data.key_factors) {
      const factorLabels = {
        rsi_14: 'RSI', adx_14: 'ADX', rv_20d: 'Vol20d', rv_60d: 'Vol60d',
        returns_5d: 'Ret5d', returns_10d: 'Ret10d', returns_20d: 'Ret20d', returns_60d: 'Ret60d',
        market_breadth: 'Breadth', vix_change_5d: 'VIX∆5d',
        price_vs_sma20_zscore: 'Z_SMA20', price_vs_sma60_zscore: 'Z_SMA60',
        macd_line_pct: 'MACD%', macd_histogram_pct: 'MACDhist%',
        relative_volume_20d: 'RelVol', regime_combined: 'Regime',
      };

      let fhtml = '';
      for (const [k, v] of Object.entries(data.key_factors)) {
        const label = factorLabels[k] || k;
        let valStr = typeof v === 'string' ? v : (Math.abs(v) < 1 ? (v * 100).toFixed(1) + '%' : v.toFixed(1));
        let valColor = '';
        if (typeof v === 'number') {
          if (k === 'rsi_14') valColor = v > 70 ? 'text-red' : v < 30 ? 'text-green' : '';
          else if (k.startsWith('returns_')) valColor = v > 0 ? 'text-green' : 'text-red';
        }
        fhtml += `<div class="factor-chip"><span class="fc-label">${label}</span><span class="fc-value ${valColor}">${valStr}</span></div>`;
      }
      factorsEl.innerHTML = fhtml;
    }
  } catch (e) {
    container.innerHTML = '<div class="chain-placeholder">暂无该股决策链数据</div>';
    if (factorsEl) factorsEl.innerHTML = '';
  }
}

// ── Load Portfolio ─────────────────────────────────────
async function loadPortfolio() {
  try {
    const resp = await fetch('/api/portfolio');
    const data = await resp.json();
    if (data.error) return;
    portfolioData = data;

    // Top stats
    setEl('tsNav', `$${data.nav.toLocaleString()}`);
    const cumEl = document.getElementById('tsCumReturn');
    if (cumEl) { cumEl.textContent = pctSign(data.cum_return); colorVal(cumEl, data.cum_return); }
    const unrealEl = document.getElementById('tsUnrealized');
    if (unrealEl) { unrealEl.textContent = pnlSign(data.unrealized_pnl); colorVal(unrealEl, data.unrealized_pnl); }
    const realEl = document.getElementById('tsRealized');
    if (realEl) { realEl.textContent = pnlSign(data.realized_pnl); colorVal(realEl, data.realized_pnl); }
    setEl('tsMaxDD', `-${data.max_drawdown.toFixed(2)}%`);
    setEl('tsInvested', `${data.invested_pct}%`);
    setEl('tsCash', `$${Math.round(data.cash).toLocaleString()}`);
    setEl('tsFriction', `$${Math.round(data.total_friction)}`);

    // Holdings table
    setEl('holdingCount', data.positions.length);
    const hb = document.getElementById('holdingsBody');
    if (hb) {
      if (data.positions.length === 0) {
        hb.innerHTML = '<tr><td colspan="7" class="empty">当前空仓</td></tr>';
      } else {
        hb.innerHTML = data.positions.map(p => {
          const pc = pnlClass(p.pnl);
          return `<tr class="holding-row" onclick="onHoldingClick('${p.symbol}')">
            <td>${p.symbol}</td>
            <td>${p.qty}</td>
            <td>$${p.avg_cost}</td>
            <td>$${p.price}</td>
            <td class="${pc}">${pnlSign(p.pnl)} (${p.pnl_pct > 0 ? '+' : ''}${p.pnl_pct}%)</td>
            <td>${p.weight}%</td>
            <td>${p.days_held || '-'}d</td>
          </tr>`;
        }).join('');
      }
    }

    // Equity curve
    if (data.equity_curve && data.equity_curve.length > 0 && equitySeries) {
      equitySeries.setData(data.equity_curve.map(e => ({ time: e.date, value: e.nav })));
      equityChart.timeScale().fitContent();
    }

    // Daily summary
    const db = document.getElementById('dailyBody');
    if (db && data.equity_curve) {
      db.innerHTML = data.equity_curve.map(e => {
        const drc = pnlClass(e.daily_return);
        const cpc = pnlClass(e.cum_pnl);
        return `<tr>
          <td>${e.date}</td>
          <td>$${e.nav.toLocaleString()}</td>
          <td class="${drc}">${e.daily_return > 0 ? '+' : ''}${e.daily_return.toFixed(3)}%</td>
          <td class="${cpc}">${pnlSign(e.cum_pnl)}</td>
          <td>${e.positions_count}</td>
          <td>${e.trades}笔</td>
          <td>$${e.friction}</td>
        </tr>`;
      }).reverse().join('');
    }
  } catch (e) {
    console.error('loadPortfolio:', e);
  }
}

// ── Holdings click → switch chart to that symbol ───────
function onHoldingClick(symbol) {
  const sel = document.getElementById('symbolSelect');
  if (sel) { sel.value = symbol; }
  loadCandles(symbol);

  // Highlight active row
  document.querySelectorAll('.holding-row').forEach(r => r.classList.remove('active'));
  event.currentTarget.classList.add('active');
}

// ── Load Trades ────────────────────────────────────────
async function loadTrades() {
  try {
    const resp = await fetch('/api/trades?limit=50');
    const data = await resp.json();
    setEl('tradeCount', data.trades.length);
    const tb = document.getElementById('tradesBody');
    if (!tb) return;
    if (data.trades.length === 0) {
      tb.innerHTML = '<tr><td colspan="7" class="empty">暂无交易记录</td></tr>';
      return;
    }
    tb.innerHTML = data.trades.map(t => {
      const ac = t.action === 'BUY' ? 'action-buy' : 'action-sell';
      const pc = t.pnl != null ? pnlClass(t.pnl) : '';
      return `<tr>
        <td>${t.date}</td>
        <td class="${ac}">${t.action === 'BUY' ? '买入' : '卖出'}</td>
        <td>${t.symbol}</td>
        <td>${t.qty}</td>
        <td>$${t.price}</td>
        <td>$${t.friction}</td>
        <td class="${pc}">${t.pnl != null ? pnlSign(t.pnl) : '-'}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    console.error('loadTrades:', e);
  }
}

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  await loadPortfolio();
  await loadSymbols();
  await loadTrades();
});
