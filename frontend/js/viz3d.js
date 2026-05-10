// 3D visualisations powered by Plotly: probability surface + loss landscape.
window.Viz3D = (() => {

  // Dark-friendly Plotly layout used by both surfaces.
  function darkLayout(extra = {}) {
    const axis = {
      backgroundcolor: '#0a0d14',
      gridcolor: '#1f2531',
      zerolinecolor: '#1f2531',
      color: '#9aa3b2',
      tickfont: { color: '#9aa3b2', size: 10 },
      titlefont: { color: '#cbd2dd', size: 11 },
    };
    return Object.assign({
      paper_bgcolor: '#0a0d14',
      plot_bgcolor:  '#0a0d14',
      font: { color: '#cbd2dd', family: 'Inter, sans-serif' },
      margin: { l: 0, r: 0, t: 6, b: 0 },
      scene: {
        xaxis: { ...axis },
        yaxis: { ...axis },
        zaxis: { ...axis },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } },
      },
    }, extra);
  }

  // 3D probability surface from a 2D boundary grid (height = probability).
  function renderSurface(el, boundary) {
    if (!window.Plotly || !el) return;
    const grid = boundary.grid;        // [res][res] of probability
    const ext  = boundary.extent;      // [x0min, x0max, x1min, x1max]
    const res  = boundary.resolution || grid.length;

    const xs = Array.from({ length: res }, (_, i) =>
      ext[0] + (ext[1] - ext[0]) * i / (res - 1));
    const ys = Array.from({ length: res }, (_, i) =>
      ext[2] + (ext[3] - ext[2]) * i / (res - 1));

    const surf = {
      type: 'surface',
      x: xs, y: ys, z: grid,
      colorscale: [
        [0.0, '#5b8def'],
        [0.5, '#0a0d14'],
        [1.0, '#7df9c7'],
      ],
      showscale: false,
      contours: { z: { show: true, usecolormap: true, highlightcolor: '#7df9c7', project: { z: true } } },
      lighting: { ambient: 0.55, diffuse: 0.6, specular: 0.2 },
    };

    // Overlay training points at z = their label, if present.
    const traces = [surf];
    if (boundary.train) {
      const Xt = boundary.train.X, yt = boundary.train.y;
      traces.push({
        type: 'scatter3d', mode: 'markers',
        x: Xt.map(p => p[0]),
        y: Xt.map(p => p[1]),
        z: yt.map(v => Array.isArray(v) ? v[0] : v),
        marker: {
          size: 3,
          color: yt.map(v => (Array.isArray(v) ? v[0] : v)),
          colorscale: [[0, '#5b8def'], [1, '#7df9c7']],
          line: { width: 0 },
        },
        name: 'train',
      });
    }

    Plotly.newPlot(el, traces, darkLayout({
      scene: {
        xaxis: { title: 'x₀' },
        yaxis: { title: 'x₁' },
        zaxis: { title: 'P(y=1)', range: [0, 1] },
        camera: { eye: { x: 1.4, y: 1.4, z: 1.2 } },
      },
    }), { displayModeBar: false, responsive: true });
  }

  // 3D loss landscape from {alpha, beta, loss}.
  function renderLandscape(el, land) {
    if (!window.Plotly || !el) return;
    const z = land.loss;
    const trace = {
      type: 'surface',
      x: land.alpha, y: land.beta, z,
      colorscale: 'Viridis',
      showscale: false,
      contours: { z: { show: true, usecolormap: true, project: { z: true } } },
      lighting: { ambient: 0.5, diffuse: 0.6, specular: 0.2 },
    };
    // Mark the trained weights at α=0, β=0.
    const center = { type: 'scatter3d', mode: 'markers',
      x: [0], y: [0], z: [z[Math.floor(z.length/2)][Math.floor(z[0].length/2)]],
      marker: { size: 5, color: '#7df9c7', symbol: 'diamond' },
      name: 'trained',
    };
    Plotly.newPlot(el, [trace, center], darkLayout({
      scene: {
        xaxis: { title: 'α (direction 1)' },
        yaxis: { title: 'β (direction 2)' },
        zaxis: { title: 'loss' },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } },
      },
    }), { displayModeBar: false, responsive: true });
  }

  return { renderSurface, renderLandscape };
})();
