// Decision boundary renderer. Takes a 2D probability grid + extent + scatter
// points, paints the heatmap to canvas, then overlays the points.
const Boundary = (() => {

  // Diverging colour ramp: cyan (0) → mid surface → pink (1).
  // We use perceptually-balanced steps and add a faint glow band near 0.5.
  function rampRgba(v) {
    v = Math.max(0, Math.min(1, v));
    // mid-toned base palette
    // cyan-ish low: #82c9ff -> rgb(130,201,255)
    // pink-ish high: #ff86c8 -> rgb(255,134,200)
    const r = Math.round(130 + (255 - 130) * v);
    const g = Math.round(201 + (134 - 201) * v);
    const b = Math.round(255 + (200 - 255) * v);
    // alpha: stronger toward extremes, soft near 0.5
    const dist = Math.abs(v - 0.5) * 2;
    const a = 0.08 + 0.45 * dist;
    return `rgba(${r},${g},${b},${a})`;
  }

  function render(canvas, payload) {
    const { grid, extent, train, test, scatter } = payload;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0e1320';
    ctx.fillRect(0, 0, W, H);

    if (!grid || !grid.length) return;

    // Paint heatmap from the grid using imageData for speed.
    const res = grid.length;
    const off = document.createElement('canvas');
    off.width = res; off.height = res;
    const octx = off.getContext('2d');
    const img = octx.createImageData(res, res);
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        // grid[row=y][col=x] — but image y flips because plot y goes up
        const v = grid[res - 1 - y][x];
        const c = rampRgba(v);
        // parse rgba()
        const m = c.match(/rgba\((\d+),(\d+),(\d+),([\d.]+)\)/);
        const idx = (y * res + x) * 4;
        img.data[idx]   = +m[1];
        img.data[idx+1] = +m[2];
        img.data[idx+2] = +m[3];
        img.data[idx+3] = Math.round(parseFloat(m[4]) * 255);
      }
    }
    octx.putImageData(img, 0, 0);

    // Draw scaled-up image with smoothing.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, W, H);

    // Subtle inner darkening at borders for depth.
    const grad = ctx.createRadialGradient(W/2, H/2, Math.min(W,H)/3, W/2, H/2, Math.max(W,H)/1.2);
    grad.addColorStop(0, 'rgba(0,0,0,0)');
    grad.addColorStop(1, 'rgba(0,0,0,0.35)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Plot scatter points with extent → canvas mapping.
    const [xmin, xmax, ymin, ymax] = extent;
    const toCanvas = (x, y) => [
      (x - xmin) / (xmax - xmin) * W,
      H - (y - ymin) / (ymax - ymin) * H,
    ];

    const pointSets = [];
    if (scatter) {
      pointSets.push({ X: scatter.X, y: scatter.y, edge: '#ffffff' });
    } else {
      if (train) pointSets.push({ X: train.X, y: train.y, edge: 'rgba(255,255,255,0.85)', radius: 3 });
      if (test)  pointSets.push({ X: test.X,  y: test.y,  edge: '#ffd166', radius: 3.6 });
    }

    // Multi-class palette — first two match the binary heatmap.
    const PALETTE = ['#82c9ff', '#ff86c8', '#7df9c7', '#ffd166', '#b48cff', '#ff9a76'];
    const classOf = (yi) => {
      if (Array.isArray(yi)) {
        if (yi.length === 1) return yi[0] < 0.5 ? 0 : 1;
        let mi = 0; for (let k = 1; k < yi.length; k++) if (yi[k] > yi[mi]) mi = k;
        return mi;
      }
      return yi < 0.5 ? 0 : 1;
    };

    for (const set of pointSets) {
      for (let i = 0; i < set.X.length; i++) {
        const cls = classOf(set.y[i]);
        const fill = PALETTE[cls % PALETTE.length];
        const [cx, cy] = toCanvas(set.X[i][0], set.X[i][1]);
        const r = set.radius || 3;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.globalAlpha = 0.95;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = set.edge || '#0a0d14';
        ctx.stroke();
      }
    }
  }

  return { render };
})();
