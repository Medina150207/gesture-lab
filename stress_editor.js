// Two one-dimensional passes give the same Gaussian blur as a 2D kernel.
function smooth(pixels, width, height, sigma) {
  if (sigma === 0) return pixels;
  const radius = Math.ceil(sigma * 3);
  const weights = Array.from({length: 2 * radius + 1}, (_, i) =>
    Math.exp(-((i - radius) ** 2) / (2 * sigma ** 2)));
  const sum = weights.reduce((a, b) => a + b);
  const kernel = weights.map(w => w / sum);
  function pass(input, horizontal) {
    const output = new Float32Array(input.length);
    const length = horizontal ? width : height;
    const stride = horizontal ? 4 : 4 * width;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = 4 * (y * width + x);
        const pos = horizontal ? x : y;
        let red = 0, green = 0, blue = 0;
        for (let k = -radius; k <= radius; k++) {
          const neighbor = Math.max(0, Math.min(length - 1, pos + k));
          const j = i + (neighbor - pos) * stride;
          const w = kernel[k + radius];
          red += input[j] * w;
          green += input[j + 1] * w;
          blue += input[j + 2] * w;
        }
        output[i] = red; output[i + 1] = green; output[i + 2] = blue;
      }
    }
    return output;
  }
  return pass(pass(pixels, true), false);
}

export default function({parentElement, data, setStateValue, setTriggerValue}) {
  const find = selector => parentElement.querySelector(selector);
  const canvas = find("canvas");
  const light = find("#light"), blur = find("#blur");
  const test = find("#test"), reset = find("#reset");
  const ctx = canvas.getContext("2d", {willReadFrequently: true});
  // The widget key changes for each photo/model. Keep its pixels across reruns.
  const state = canvas.gesturePreview ??= {};
  const settings = () => [Number(light.value), Number(blur.value)];
  function draw() {
    if (!state.original) return;
    const [gain, sigma] = settings();
    if (state.sigma !== sigma) {
      state.smoothed = smooth(state.original, canvas.width, canvas.height, sigma);
      state.sigma = sigma;
    }
    const result = state.output.data;
    for (let i = 0; i < result.length; i += 4) {
      for (let c = 0; c < 3; c++) result[i + c] = gain * state.smoothed[i + c];
      result[i + 3] = 255;
    }
    ctx.putImageData(state.output, 0, 0);
    find("#light-value").textContent = gain.toFixed(1) + "×";
    find("#blur-value").textContent = sigma + " px";
  }
  for (const slider of [light, blur]) {
    slider.oninput = () => {
      cancelAnimationFrame(state.frame);
      state.frame = requestAnimationFrame(draw);
    };
    slider.onchange = () => setStateValue("settings", settings());
  }
  reset.onclick = () => {
    light.value = 1; blur.value = 0;
    draw();
    setStateValue("settings", settings());
    setTriggerValue("reset", true);
  };
  test.onclick = () => {
    draw();
    setTriggerValue("test", {settings: settings(), pixels: canvas.toDataURL("image/png").split(",")[1]});
  };
  if (!state.loading) {
    const img = new Image();
    state.loading = true;
    img.onload = () => {
      canvas.width = img.width; canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      state.original = ctx.getImageData(0, 0, img.width, img.height).data;
      state.output = ctx.createImageData(img.width, img.height);
      draw();
      for (const control of [light, blur, test, reset]) control.disabled = false;
    };
    img.onerror = () => { find("[role=status]").textContent = "Could not load this photo. Try another image."; };
    img.src = "data:image/png;base64," + data.photo;
  }
}
