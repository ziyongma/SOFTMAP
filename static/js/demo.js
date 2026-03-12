/* ============================================================
   SOFTMAP — Browser Interactive Demo
   ============================================================
   Runs the FingerMLP (+ optional CorrNet) via ONNX Runtime Web
   and renders the predicted 3D point cloud with Three.js.

   Keyboard controls (click the viewer to activate):
     <- / ->     Servo 1  +/-100 ticks
     Up / Dn     Servo 2  +/-100 ticks
     A / D       Servo 1  +/-1000 ticks
     W / S       Servo 2  +/-1000 ticks
     R           Set current position as zero reference
     Space       Return to zero reference
     C           Reset both servos to absolute zero

   Expected assets in ./static/demo_assets/:
     main_model.onnx      – FingerMLP  (2 -> 1644)
     correction_net.onnx  – CorrNet    (2 -> 1644)  [optional]
     model_stats.json     – normalisation stats + calibration
   ============================================================ */

(function () {
  "use strict";

  /* ---- constants ---- */
  var ASSETS      = "./static/demo_assets/";
  var POS_MIN     = -7000;
  var POS_MAX     =  7000;
  var STEP_SMALL  = 100;
  var STEP_LARGE  = 1000;
  var POINT_SIZE_PX = 3.0;

  /* ---- state ---- */
  var stats       = null;
  var mainSession = null;
  var corrSession = null;
  var nVerts      = 548;

  var pos1 = 0, pos2 = 0;
  var zeroPos1 = 0, zeroPos2 = 0;
  var keyboardActive = false;

  /* ---- Three.js objects ---- */
  var scene, camera, renderer, controls, points;
  var canvasWrap;

  /* ---- DOM handles ---- */
  var slider1, slider2, label1, label2, statusEl, canvas, activateOverlay;

  /* ============================================================
     Helpers
     ============================================================ */

  function clamp(v) { return Math.max(POS_MIN, Math.min(POS_MAX, v)); }

  function syncSliders() {
    if (slider1) { slider1.value = pos1; label1.textContent = pos1; }
    if (slider2) { slider2.value = pos2; label2.textContent = pos2; }
  }

  function updateStatus(note) {
    if (!statusEl) return;
    var rel1 = pos1 - zeroPos1;
    var rel2 = pos2 - zeroPos2;
    var pipe = (stats && stats.calib_A ? "CAL" : "---");
    pipe += (corrSession ? "+" : "-");
    pipe += (corrSession ? "CRR" : "---");
    var txt = "S1=" + pos1 + " (\u0394" + rel1 + ")  S2=" + pos2 + " (\u0394" + rel2 + ")  [" + pipe + "]";
    if (note) txt += "  \u2014 " + note;
    statusEl.textContent = txt;
  }

  /* ============================================================
     Initialisation
     ============================================================ */

  async function init() {
    canvasWrap      = document.querySelector(".demo-canvas-wrap");
    canvas          = document.getElementById("demo-canvas");
    slider1         = document.getElementById("servo-slider-1");
    slider2         = document.getElementById("servo-slider-2");
    label1          = document.getElementById("servo-label-1");
    label2          = document.getElementById("servo-label-2");
    statusEl        = document.getElementById("demo-status");
    activateOverlay = document.getElementById("demo-activate-overlay");

    if (!canvasWrap || !canvas) return;

    updateStatus("Loading model assets\u2026");

    try {
      var resp = await fetch(ASSETS + "model_stats.json");
      if (!resp.ok) throw new Error("model_stats.json not found");
      stats  = await resp.json();
      nVerts = stats.n_vertices || 548;

      mainSession = await ort.InferenceSession.create(ASSETS + "main_model.onnx");

      try {
        corrSession = await ort.InferenceSession.create(ASSETS + "correction_net.onnx");
      } catch (_) {
        corrSession = null;
      }

      updateStatus("Model loaded \u2014 click to activate keyboard");
    } catch (e) {
      updateStatus("Demo assets not yet available.");
      console.warn("[demo]", e);
      disableControls();
      initScene(true);
      return;
    }

    initScene(false);
    bindSliders();
    bindKeyboard();
    predict();
  }

  /* ============================================================
     Three.js scene
     ============================================================ */

  function initScene(isEmpty) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111118);

    var w = canvasWrap.clientWidth;
    var h = canvasWrap.clientHeight;
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(0, 0, 120);

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.target.set(0, 0, 0);

    var geom = new THREE.BufferGeometry();
    var n = isEmpty ? 1 : nVerts;
    var pos  = new Float32Array(n * 3);
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    var mat  = new THREE.PointsMaterial({
      size: POINT_SIZE_PX,
      sizeAttenuation: false,
      color: isEmpty ? 0x555560 : 0x33c77f,
    });
    points = new THREE.Points(geom, mat);
    scene.add(points);

    var axes = new THREE.AxesHelper(15);
    axes.material.opacity = 0.25;
    axes.material.transparent = true;
    scene.add(axes);

    window.addEventListener("resize", onResize);
    animate();
  }

  function onResize() {
    if (!canvasWrap || !renderer) return;
    var w = canvasWrap.clientWidth;
    var h = canvasWrap.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer) renderer.render(scene, camera);
  }

  function updatePointCloud(vertices) {
    var geom = points.geometry;
    var attr = geom.getAttribute("position");

    if (attr.count !== nVerts) {
      geom.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(nVerts * 3), 3));
      attr = geom.getAttribute("position");
    }

    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < nVerts; i++) {
      cx += vertices[i * 3];
      cy += vertices[i * 3 + 1];
      cz += vertices[i * 3 + 2];
    }
    cx /= nVerts; cy /= nVerts; cz /= nVerts;

    for (var i = 0; i < nVerts; i++) {
      attr.setXYZ(i,
        vertices[i * 3]     - cx,
        vertices[i * 3 + 1] - cy,
        vertices[i * 3 + 2] - cz
      );
    }
    attr.needsUpdate = true;
    geom.computeBoundingSphere();

    if (!updatePointCloud._fitted) {
      var bs = geom.boundingSphere;
      if (bs && bs.radius > 0) {
        camera.position.set(0, 0, bs.radius * 3.2);
        controls.target.set(0, 0, 0);
        controls.update();
      }
      updatePointCloud._fitted = true;
    }
  }

  /* ============================================================
     ONNX inference (mirrors model.py predict_vertices)
     ============================================================ */

  async function predict() {
    if (!mainSession || !stats) return;

    var rel1 = pos1 - zeroPos1;
    var rel2 = pos2 - zeroPos2;

    var x;
    if (stats.calib_A) {
      var A = stats.calib_A;
      var b = stats.calib_b;
      x = [
        A[0][0] * rel1 + A[0][1] * rel2 + b[0],
        A[1][0] * rel1 + A[1][1] * rel2 + b[1],
      ];
    } else {
      var scale0 = 3.0 * stats.input_std[0] / POS_MAX;
      var scale1 = 3.0 * stats.input_std[1] / POS_MAX;
      x = [rel1 * scale0, rel2 * scale1];
    }

    var xNorm = new Float32Array([
      (x[0] - stats.input_mean[0]) / (stats.input_std[0] + 1e-8),
      (x[1] - stats.input_mean[1]) / (stats.input_std[1] + 1e-8),
    ]);

    var inputTensor = new ort.Tensor("float32", xNorm, [1, 2]);
    var mainOut = await mainSession.run({ servo_input: inputTensor });
    var pNorm  = mainOut.vertex_output.data;

    var verts = new Float32Array(nVerts * 3);
    for (var i = 0; i < nVerts * 3; i++) {
      verts[i] = pNorm[i] * stats.output_std[i] + stats.output_mean[i];
    }

    if (corrSession && stats.corrnet_input_mean) {
      var xCorr = new Float32Array([
        (x[0] - stats.corrnet_input_mean[0]) / (stats.corrnet_input_std[0] + 1e-8),
        (x[1] - stats.corrnet_input_mean[1]) / (stats.corrnet_input_std[1] + 1e-8),
      ]);
      var corrTensor = new ort.Tensor("float32", xCorr, [1, 2]);
      var corrOut = await corrSession.run({ sim_input: corrTensor });
      var delta   = corrOut.delta_output.data;
      for (var i = 0; i < nVerts * 3; i++) {
        verts[i] += delta[i];
      }
    }

    for (var i = 0; i < nVerts; i++) {
      verts[i * 3] *= -1;
    }

    updatePointCloud(verts);
  }

  /* ============================================================
     Keyboard controls + HUD key highlights
     ============================================================ */

  function step(d1, d2, note) {
    pos1 = clamp(pos1 + d1);
    pos2 = clamp(pos2 + d2);
    syncSliders();
    updateStatus(note || "");
    predict();
  }

  function setZero() {
    zeroPos1 = pos1;
    zeroPos2 = pos2;
    updateStatus("Zero reference set");
    predict();
  }

  function gotoZero() {
    pos1 = zeroPos1;
    pos2 = zeroPos2;
    syncSliders();
    updateStatus("Returned to zero");
    predict();
  }

  function resetOrigin() {
    pos1 = 0; pos2 = 0;
    zeroPos1 = 0; zeroPos2 = 0;
    syncSliders();
    updateStatus("Reset to origin");
    predict();
  }

  /* Map e.key values to HUD data-key attributes */
  var keyToDataKey = {
    "ArrowUp": "ArrowUp", "ArrowDown": "ArrowDown",
    "ArrowLeft": "ArrowLeft", "ArrowRight": "ArrowRight",
    "w": "w", "W": "w", "a": "a", "A": "a",
    "s": "s", "S": "s", "d": "d", "D": "d",
    "r": "r", "R": "r", "c": "c", "C": "c", " ": " ",
  };

  function highlightKey(dataKey, on) {
    var el = canvasWrap.querySelector('.demo-key[data-key="' + dataKey + '"]');
    if (el) {
      if (on) el.classList.add("is-pressed");
      else    el.classList.remove("is-pressed");
    }
  }

  function bindKeyboard() {
    /* Activate on click */
    canvasWrap.setAttribute("tabindex", "0");
    canvasWrap.style.outline = "none";

    function activate() {
      keyboardActive = true;
      canvasWrap.focus();
      if (activateOverlay) activateOverlay.classList.add("is-hidden");
      updateStatus("");
    }

    if (activateOverlay) {
      activateOverlay.addEventListener("click", activate);
    }
    canvasWrap.addEventListener("mousedown", activate);

    /* Deactivate when clicking outside */
    document.addEventListener("mousedown", function (e) {
      if (!canvasWrap.contains(e.target)) {
        keyboardActive = false;
      }
    });

    /* Keydown — action + highlight */
    document.addEventListener("keydown", function (e) {
      if (!keyboardActive || !mainSession) return;

      var dk = keyToDataKey[e.key];
      if (dk) highlightKey(dk, true);

      var handled = true;
      switch (e.key) {
        case "ArrowLeft":  step(-STEP_SMALL, 0); break;
        case "ArrowRight": step(+STEP_SMALL, 0); break;
        case "ArrowUp":    step(0, +STEP_SMALL); break;
        case "ArrowDown":  step(0, -STEP_SMALL); break;
        case "a": case "A": step(-STEP_LARGE, 0); break;
        case "d": case "D": step(+STEP_LARGE, 0); break;
        case "w": case "W": step(0, +STEP_LARGE); break;
        case "s": case "S": step(0, -STEP_LARGE); break;
        case "r": case "R": setZero(); break;
        case " ":           gotoZero(); break;
        case "c": case "C": resetOrigin(); break;
        default: handled = false;
      }
      if (handled) e.preventDefault();
    });

    /* Keyup — remove highlight */
    document.addEventListener("keyup", function (e) {
      var dk = keyToDataKey[e.key];
      if (dk) highlightKey(dk, false);
    });
  }

  /* ============================================================
     Slider UI
     ============================================================ */

  function bindSliders() {
    slider1.addEventListener("input", function () {
      pos1 = parseInt(slider1.value, 10);
      label1.textContent = pos1;
      updateStatus();
      predict();
    });
    slider2.addEventListener("input", function () {
      pos2 = parseInt(slider2.value, 10);
      label2.textContent = pos2;
      updateStatus();
      predict();
    });
  }

  function disableControls() {
    if (slider1) { slider1.disabled = true; slider1.style.opacity = 0.4; }
    if (slider2) { slider2.disabled = true; slider2.style.opacity = 0.4; }
  }

  /* ---- boot ---- */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();
