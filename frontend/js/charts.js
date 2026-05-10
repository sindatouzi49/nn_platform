// Chart.js wrappers tuned to the dark instrument theme.
const Charts = (() => {

  const palette = {
    mint:  '#7df9c7',
    pink:  '#ff86c8',
    cyan:  '#82c9ff',
    amber: '#ffc05c',
    rose:  '#ff7e7e',
    violet:'#b58cff',
  };

  // Common axis / grid styling.
  const axis = {
    grid:  { color: '#1f2738', drawBorder: false, drawTicks: false },
    border:{ display: false },
    ticks: {
      color: '#6e7896',
      font: { family: 'JetBrains Mono', size: 10 },
      padding: 6,
    },
  };

  function baseOptions(opts = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          align: 'end',
          labels: {
            color: '#b3bccf',
            usePointStyle: true,
            pointStyle: 'rectRounded',
            boxWidth: 10,
            boxHeight: 10,
            font: { family: 'Inter', size: 11 },
          },
        },
        tooltip: {
          backgroundColor: '#0e1320',
          titleColor: '#e6ebf5',
          bodyColor: '#b3bccf',
          borderColor: '#2a3349',
          borderWidth: 1,
          padding: 10,
          titleFont: { family: 'Inter', size: 11, weight: 600 },
          bodyFont:  { family: 'JetBrains Mono', size: 11 },
          displayColors: false,
        },
      },
      scales: {
        x: { ...axis, title: { display: false } },
        y: { ...axis, title: { display: false } },
      },
      ...opts,
    };
  }

  function lineDataset(label, data, color, dash = false) {
    return {
      label,
      data,
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 1.8,
      borderDash: dash ? [4, 4] : [],
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: color,
      pointHoverBorderColor: '#0a0d14',
      fill: dash ? false : 'origin',
    };
  }

  // ── Loss & accuracy charts (line) ─────────────────────────────────────
  function makeLine(canvas) {
    return new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: baseOptions({
        scales: {
          x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 8 } },
          y: { ...axis, beginAtZero: true },
        },
      }),
    });
  }

  function updateLoss(chart, history) {
    const labels = history.train_loss.map((_, i) => i + 1);
    chart.data.labels = labels;
    chart.data.datasets = [
      lineDataset('train', history.train_loss, palette.mint),
      lineDataset('validation', history.val_loss, palette.pink, true),
    ];
    chart.update('none');
  }

  function updateAcc(chart, history) {
    if (history.train_acc[0] == null) return false;
    const labels = history.train_acc.map((_, i) => i + 1);
    chart.data.labels = labels;
    chart.data.datasets = [
      lineDataset('train', history.train_acc.map(v => v * 100), palette.cyan),
      lineDataset('validation', history.val_acc.map(v => v * 100), palette.amber, true),
    ];
    chart.options.scales.y.suggestedMax = 100;
    chart.options.scales.y.suggestedMin = 0;
    chart.update('none');
    return true;
  }

  // ── Scatter chart (input data preview) ────────────────────────────────
  function makeScatter(canvas) {
    return new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets: [] },
      options: baseOptions({
        scales: {
          x: { ...axis },
          y: { ...axis },
        },
        plugins: { legend: { display: false } },
      }),
    });
  }

  function updateScatter(chart, X, y) {
    if (!X || X.length === 0 || X[0].length < 2) {
      chart.data.datasets = [];
      chart.update('none');
      return;
    }
    // Multi-class palette — first two match the binary cyan/pink convention.
    const multi = [palette.cyan, palette.pink, palette.mint, palette.amber, palette.violet, palette.rose];
    const classOf = (yi) => {
      if (Array.isArray(yi)) {
        if (yi.length === 1) return yi[0] < 0.5 ? 0 : 1;
        let mi = 0; for (let k = 1; k < yi.length; k++) if (yi[k] > yi[mi]) mi = k;
        return mi;
      }
      return yi < 0.5 ? 0 : 1;
    };
    const groups = {};
    y.forEach((yi, i) => {
      const k = String(classOf(yi));
      (groups[k] ||= []).push({ x: X[i][0], y: X[i][1] });
    });
    chart.data.datasets = Object.entries(groups)
      .sort((a, b) => +a[0] - +b[0])
      .map(([k, pts]) => {
        const col = multi[(+k) % multi.length];
        return {
          label: 'class ' + k,
          data: pts,
          backgroundColor: col + 'cc',
          borderColor: col,
          borderWidth: 0.8,
          pointRadius: 3.5,
          pointHoverRadius: 5,
        };
      });
    chart.update('none');
  }

  return { makeLine, updateLoss, updateAcc, makeScatter, updateScatter, palette };
})();
