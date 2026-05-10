// Animated SVG network: layers as columns of neurons, edges weighted by
// |w| (thickness/opacity) and signed by colour (mint=positive, pink=negative).
const NetworkViz = (() => {
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs = {}) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // Render network architecture (no weights yet) — used before training.
  function renderArchitecture(container, layers) {
    return renderWeights(container, layers, null);
  }

  // Compute neuron positions for a given (layers, viewport).
  function layoutPositions(layers, w, h, padX = 30, padY = 20) {
    const cols = layers.length;
    const xs = [];
    for (let i = 0; i < cols; i++) {
      xs.push(padX + (cols === 1 ? w / 2 : (i * (w - 2 * padX)) / (cols - 1)));
    }
    const positions = layers.map((n, i) => {
      const stepY = (h - 2 * padY) / Math.max(1, n - 1);
      const startY = n === 1 ? h / 2 : padY;
      const arr = [];
      for (let k = 0; k < n; k++) arr.push({ x: xs[i], y: n === 1 ? h / 2 : startY + k * stepY });
      return arr;
    });
    return positions;
  }

  // Render network with current weights. weights = list of 2D arrays sized
  // (layers[i] x layers[i+1]); pass null to draw empty edges.
  function renderWeights(container, layers, weights) {
    const w = container.clientWidth || 600;
    const h = container.clientHeight || 280;
    const positions = layoutPositions(layers, w, h);

    // If we have already rendered at this layout, animate; else replace.
    let svg = container.querySelector('svg');
    if (!svg || svg.dataset.layers !== JSON.stringify(layers)
            || +svg.dataset.w !== w || +svg.dataset.h !== h) {
      clear(container);
      svg = el('svg', { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'xMidYMid meet' });
      svg.dataset.layers = JSON.stringify(layers);
      svg.dataset.w = w;
      svg.dataset.h = h;
      container.appendChild(svg);

      // Layer labels
      const labelGroup = el('g');
      layers.forEach((n, i) => {
        const txt = el('text', {
          x: positions[i][0].x,
          y: 14,
          'text-anchor': 'middle',
          fill: '#6e7896',
          'font-family': 'JetBrains Mono',
          'font-size': '9.5',
          'letter-spacing': '0.1em',
        });
        const role = i === 0 ? 'INPUT' : (i === layers.length - 1 ? 'OUTPUT' : `H${i}`);
        txt.textContent = `${role} · ${n}`;
        labelGroup.appendChild(txt);
      });
      svg.appendChild(labelGroup);

      // Edge group + neuron group placeholders.
      const eg = el('g', { id: 'edges' });
      const ng = el('g', { id: 'neurons' });
      svg.appendChild(eg);
      svg.appendChild(ng);
    }

    const eg = svg.querySelector('#edges');
    const ng = svg.querySelector('#neurons');
    clear(eg);
    clear(ng);

    // Compute weight magnitude scale across the network for normalisation.
    let maxAbs = 0.0;
    if (weights) for (const W of weights) for (const row of W) for (const v of row)
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    if (maxAbs === 0) maxAbs = 1;

    // Edges
    for (let li = 0; li < layers.length - 1; li++) {
      const left  = positions[li];
      const right = positions[li + 1];
      const W = weights ? weights[li] : null;
      for (let i = 0; i < left.length; i++) {
        for (let j = 0; j < right.length; j++) {
          const wv = W ? W[i][j] : 0;
          const norm = Math.abs(wv) / maxAbs;     // 0..1
          const sign = wv >= 0;
          const stroke = sign ? '#7df9c7' : '#ff86c8';
          const op = weights ? Math.max(0.06, Math.min(0.85, 0.12 + norm * 0.8)) : 0.10;
          const sw = weights ? (0.4 + norm * 2.4) : 0.6;

          const line = el('line', {
            x1: left[i].x, y1: left[i].y,
            x2: right[j].x, y2: right[j].y,
            stroke, 'stroke-opacity': op, 'stroke-width': sw,
            'stroke-linecap': 'round',
          });
          eg.appendChild(line);
        }
      }
    }

    // Neurons (circles with subtle gradients)
    for (let li = 0; li < layers.length; li++) {
      const role = li === 0 ? 'in' : (li === layers.length - 1 ? 'out' : 'hidden');
      const fill =
        role === 'in'   ? '#82c9ff' :
        role === 'out'  ? '#ff86c8' :
                          '#7df9c7';
      for (const p of positions[li]) {
        const ring = el('circle', {
          cx: p.x, cy: p.y, r: 11,
          fill: 'none',
          stroke: fill,
          'stroke-opacity': 0.18,
          'stroke-width': 1,
        });
        const core = el('circle', {
          cx: p.x, cy: p.y, r: 5,
          fill,
          'fill-opacity': 0.85,
          class: 'neuron',
          filter: `drop-shadow(0 0 6px ${fill}aa)`,
        });
        ng.appendChild(ring);
        ng.appendChild(core);
      }
    }
  }

  // Brief glow on output neurons during training updates.
  function pulse(container) {
    const svg = container.querySelector('svg');
    if (!svg) return;
    const out = svg.querySelectorAll('#neurons circle.neuron');
    out.forEach(c => {
      c.classList.remove('firing');
      // restart animation
      void c.getBoundingClientRect();
      c.classList.add('firing');
    });
  }

  function archDescription(layers) {
    return layers.join(' · ');
  }

  return { renderArchitecture, renderWeights, pulse, archDescription };
})();
