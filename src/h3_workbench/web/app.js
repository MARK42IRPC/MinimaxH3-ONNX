const TELEMETRY_ENDPOINT = "/api/hardware/snapshot";
const TELEMETRY_STREAM_ENDPOINT = "/api/hardware/stream";
const TELEMETRY_MODE = new URLSearchParams(window.location.search).get("telemetry") || "stream";
const TELEMETRY_MAX_POINTS = 180;
const JOB_POLL_INTERVAL_MS = 2000;
const TELEMETRY_POLL_INTERVAL_MS = 1000;

const state = {
  system: null,
  models: [],
  exportPresets: [],
  modelWorkspace: null,
  jobs: [],
  profiles: [],
  frameImages: {
    start: { path: null, info: null, previewUrl: null, uploading: false, error: null, fileName: null, uploadId: 0 },
    end: { path: null, info: null, previewUrl: null, uploading: false, error: null, fileName: null, uploadId: 0 },
  },
  frameUploadSerial: 0,
  superVideo: null,
  superPromptEdited: false,
  superUploadActive: false,
  loaded: { system: false, models: false, presets: false, jobs: false, profiles: false },
  activePage: "models",
  jobFilter: "all",
  selectedJobId: null,
  jobEvents: new Map(),
  jobSignatures: new Map(),
  telemetry: [],
  telemetrySource: "none",
  telemetryError: null,
  telemetryEndpointAvailable: null,
  telemetryRetryAt: 0,
  lastJobUpdate: null,
};

let toastTimer = null;
let jobsPollActive = false;
let telemetryPollActive = false;
let systemPollActive = false;
let chartFrame = null;
let telemetryEventSource = null;
let telemetryLastReceivedAt = 0;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[character]);
}

function safeClass(value, fallback = "") {
  const normalized = String(value || "").toLowerCase();
  return /^[a-z0-9_-]+$/.test(normalized) ? normalized : fallback;
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatBytes(value) {
  const number = finite(value);
  if (number === null) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = Math.max(0, number);
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  const digits = index >= 3 ? 2 : index > 0 ? 1 : 0;
  return `${size.toFixed(digits)} ${units[index]}`;
}

function formatRate(value) {
  const number = finite(value);
  return number === null ? "--" : `${formatBytes(number)}/s`;
}

function formatPercent(value) {
  const number = finite(value);
  return number === null ? "--" : `${clamp(number, 0, 100).toFixed(0)}%`;
}

function formatElapsed(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function jobElapsed(job, now = Date.now()) {
  const started = Date.parse(job.started_at || job.created_at || "");
  if (!Number.isFinite(started)) return "--:--:--";
  const finished = Date.parse(job.finished_at || "");
  return formatElapsed((Number.isFinite(finished) ? finished : now) - started);
}

function shortTime(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return "--:--:--";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 4400);
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || `${response.status} ${response.statusText}`, response.status);
  }
  if (response.status === 204) return null;
  return response.json();
}

function fileUploadHeaders(file) {
  return {
    "x-filename": encodeURIComponent(file.name),
    "Content-Type": file.type || "application/octet-stream",
  };
}

function componentLabel(component) {
  const labels = {
    audio_vae: "Audio VAE",
    video_vae: "Video VAE",
    text_encoder: "Qwen 文本编码器",
    fl2va_transformer: "FL2VA 主模型",
    ref2va_transformer: "Ref2VA 主模型",
    acceleration_lora: "加速 LoRA",
    tokenizer: "H3 Tokenizer",
    unknown: "未识别",
  };
  return labels[component] || component || "未识别";
}

function componentStatusLabel(status) {
  return {
    ready: "已验证",
    source_ready: "原文件就绪",
    download_required: "需要下载",
    dependency_required: "等待基座",
    incomplete: "文件不完整",
    missing: "未生成",
    invalid: "清单损坏",
    unvalidated: "未验证",
  }[status] || status || "未知";
}

function jobStatusLabel(status) {
  return {
    queued: "排队中",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
  }[status] || status || "未知";
}

function jobKindLabel(job) {
  const labels = {
    inference: "推理生成",
    super_resolution: "视频超分",
    download_export: "下载并自动切片",
    export: "本地模型切片",
    download: "模型下载",
  };
  return labels[job.kind] || job.kind || "任务";
}

function isActiveJob(job) {
  return job.status === "queued" || job.status === "running";
}

function setConnection(online, message = "服务正常") {
  const dot = $("#healthDot");
  dot.classList.toggle("online", online);
  dot.classList.toggle("error", !online);
  $("#healthText").textContent = message;
  $("#connectionBanner").classList.toggle("hidden", online);
  if (!online) $("#connectionBanner").textContent = `工作台连接异常：${message}。界面会继续自动重试。`;
}

function renderSystem() {
  if (!state.system) return;
  const system = state.system;
  const gpu = system.gpu || {};
  const cuda = Array.isArray(system.providers) && system.providers.includes("CUDAExecutionProvider");
  $("#workspacePath").textContent = system.workspace || "工作区未知";
  $("#sidebarGpu").textContent = gpu.name || "未检测到 GPU";
  $("#sidebarProvider").textContent = cuda ? "ONNX CUDA" : "ONNX CPU";
  $("#sidebarMemory").textContent = `可用内存 ${formatBytes(system.memory_available_bytes)}`;
  $("#providerDot").classList.toggle("online", cuda);
}

function presetProductReady(id) {
  return state.exportPresets.find((preset) => preset.id === id)?.product?.ready === true;
}

function mainProductReady() {
  return presetProductReady("fl2va_streaming");
}

function presetJob(preset) {
  return state.jobs.find((job) => job.kind === "download_export" && (
    job.model_id === preset.label
    || job.result?.preset === preset.id
  ));
}

function assetUrl(asset) {
  if (!asset) return "";
  const provided = asset.download_url || asset.url;
  if (typeof provided === "string" && /^https:\/\//i.test(provided)) return provided;
  if (!asset.repo_id || !asset.path) return "";
  const path = String(asset.path).split("/").map((part) => encodeURIComponent(part)).join("/");
  return `https://huggingface.co/${encodeURI(String(asset.repo_id))}/resolve/main/${path}?download=true`;
}

function assetRoleLabel(asset, index) {
  return {
    lora: "LoRA",
    silu_timestep_grid: "SiLU 网格",
    tokenizer: "Tokenizer",
  }[asset?.role] || (index === 0 ? "原模型" : "辅助文件");
}

function renderModelOverview() {
  const requiredChecks = [
    presetProductReady("tokenizer"),
    presetProductReady("qwen"),
    mainProductReady(),
    presetProductReady("video_vae"),
    presetProductReady("audio_vae"),
  ];
  const requiredReady = requiredChecks.filter(Boolean).length;
  const validated = state.exportPresets.filter((preset) => preset.product?.ready).length;
  const readySources = new Set(state.exportPresets.flatMap((preset) => preset.sources || [])
    .filter((source) => source.ready)
    .map((source) => `${source.repo_id}:${source.path}`)).size;
  const modelTasks = state.jobs.filter((job) => ["download_export", "export"].includes(job.kind) && isActiveJob(job)).length;
  $("#requiredModelMetric").textContent = state.loaded.presets ? `${requiredReady} / ${requiredChecks.length}` : "--";
  $("#validatedModelMetric").textContent = state.loaded.presets ? `${validated} / ${state.exportPresets.length}` : "--";
  $("#sourceModelMetric").textContent = state.loaded.presets ? `${readySources} 个` : "--";
  $("#modelTaskMetric").textContent = state.loaded.jobs ? `${modelTasks} 个` : "--";
  const freeBytes = finite(state.modelWorkspace?.disk?.free_bytes);
  $("#diskNotice").textContent = `原始权重受 MiniMax H3 Community License Agreement 约束。${freeBytes === null ? "任务启动前会检查所需空间。" : `当前工作盘可用 ${formatBytes(freeBytes)}，任务会再次检查源文件、切片或适配器产物所需空间。`}`;

  const profile = state.profiles[0];
  const status = $("#modelReadiness");
  status.classList.remove("warning", "error", "neutral");
  if (!state.loaded.presets || !state.loaded.profiles) {
    status.classList.add("neutral");
    status.querySelector("span:last-child").textContent = "正在检查模型";
  } else if (profile?.generation_ready && requiredReady === requiredChecks.length) {
    status.querySelector("span:last-child").textContent = "端到端链路已就绪";
  } else {
    status.classList.add("warning");
    const missing = Math.max(0, requiredChecks.length - requiredReady);
    status.querySelector("span:last-child").textContent = missing ? `缺少 ${missing} 个必需组件` : "主模型尚未就绪";
  }
}

function renderExportPresets() {
  const list = $("#exportPresetList");
  if (!state.loaded.presets) {
    list.innerHTML = '<div class="loading-state"><span class="spinner"></span>正在读取可用方案</div>';
    return;
  }
  if (state.exportPresets.length === 0) {
    list.innerHTML = '<div class="empty-state"><strong>没有可用的验证方案</strong><span>服务端尚未登记原模型与切片配置。</span></div>';
    return;
  }

  list.innerHTML = state.exportPresets.map((preset) => {
    const job = presetJob(preset);
    const active = job && isActiveJob(job);
    const runtimeAdapter = preset.product_type === "runtime_adapter";
    const dependencies = Array.isArray(preset.dependencies) ? preset.dependencies : [];
    const dependencyMissing = dependencies.some((dependency) => dependency.ready !== true);
    const status = active ? job.status : preset.status || (preset.product?.ready ? "ready" : "download_required");
    const statusLabel = active ? jobStatusLabel(job.status) : componentStatusLabel(status);
    const assets = Array.isArray(preset.sources) && preset.sources.length
      ? preset.sources.map((asset, index) => [assetRoleLabel(asset, index), asset])
      : [["原模型", preset.source], ["LoRA", preset.lora], ["辅助文件", preset.support]].filter(([, asset]) => asset);
    const sourceLines = assets.map(([label, asset]) => {
      const href = assetUrl(asset);
      const identity = `${asset.repo_id || "来源"} · ${asset.path || "文件"}`;
      const sourceLabel = asset.ready === true ? `${label} ✓` : label;
      return `<div class="source-line"><span>${escapeHtml(sourceLabel)}</span>${href
        ? `<a href="${escapeHtml(href)}" title="${escapeHtml(identity)}" target="_blank" rel="noreferrer">${escapeHtml(identity)}</a>`
        : `<span title="${escapeHtml(identity)}">${escapeHtml(identity)}</span>`}</div>`;
    }).join("");
    const dependencyLines = dependencies.map((dependency) => {
      const label = dependency.ready === true ? "依赖 ✓" : "依赖";
      const detail = `${dependency.label || dependency.id}${dependency.ready === true ? " 已验证" : " 尚未就绪"}`;
      return `<div class="source-line"><span>${escapeHtml(label)}</span><span title="${escapeHtml(dependency.path || detail)}">${escapeHtml(detail)}</span></div>`;
    }).join("");
    const sources = `${sourceLines}${dependencyLines}`;
    const progress = active ? clamp(finite(job.progress) ?? 0, 0, 1) : null;
    const buttonLabel = active
      ? jobStatusLabel(job.status)
      : runtimeAdapter
        ? status === "ready" ? "重新校验适配器" : status === "source_ready" ? "生成动态适配器" : "下载并生成适配器"
        : status === "ready" ? "重新检查并切片" : status === "source_ready" ? "开始自动切片" : "下载并自动切片";
    const sourceMetric = runtimeAdapter ? "适配文件" : "原文件";
    const productMetric = runtimeAdapter ? "拓扑产物" : "切片产物";
    const actionDetail = dependencyMissing
      ? `先完成 ${dependencies.find((dependency) => dependency.ready !== true)?.label || "依赖模型"}`
      : preset.product?.ready
        ? `当前产物 ${formatBytes(preset.product.size_bytes)}`
        : status === "source_ready" ? "源文件已缓存" : "HTTP Range 断点续传";
    return `<article class="preset-row">
      <div class="preset-identity">
        <div class="preset-title-line"><strong>${escapeHtml(preset.label)}</strong><span class="status-pill ${safeClass(status)}">${escapeHtml(statusLabel)}</span></div>
        <p>${escapeHtml(preset.description)}</p>
      </div>
      <div class="preset-source">${sources}</div>
      <dl class="preset-metrics">
        <div><dt>${sourceMetric}</dt><dd>${formatBytes(preset.download_size_bytes)}</dd></div>
        <div><dt>${productMetric}</dt><dd>${formatBytes(preset.output_size_bytes)}</dd></div>
        <div><dt>空间需求</dt><dd>${formatBytes(preset.required_space_bytes)}</dd></div>
      </dl>
      <div class="preset-actions">
        <button class="primary-button preset-export-button" type="button" data-preset="${escapeHtml(preset.id)}" ${active || dependencyMissing ? "disabled" : ""}>${escapeHtml(buttonLabel)}</button>
        <small>${escapeHtml(actionDetail)}</small>
      </div>
      ${progress !== null ? `<div class="pipeline-progress"><div class="progress-track"><div class="progress-bar" style="width:${(progress * 100).toFixed(1)}%"></div></div><small>${Math.round(progress * 100)}% · ${escapeHtml(job.message)}</small></div>` : ""}
    </article>`;
  }).join("");
}

function renderComponents() {
  const grid = $("#componentGrid");
  const empty = $("#componentsEmpty");
  if (!state.loaded.presets || state.exportPresets.length === 0) {
    grid.innerHTML = state.loaded.presets ? "" : '<div class="loading-state"><span class="spinner"></span>正在读取组件状态</div>';
    empty.classList.toggle("hidden", !state.loaded.presets);
    $("#componentCount").textContent = "--";
    return;
  }
  empty.classList.add("hidden");
  const ready = state.exportPresets.filter((preset) => preset.product?.ready).length;
  $("#componentCount").textContent = `${ready} / ${state.exportPresets.length} 已验证`;
  grid.innerHTML = state.exportPresets.map((preset) => {
    const status = preset.status || (preset.product?.ready ? "ready" : "download_required");
    const runtimeAdapter = preset.product_type === "runtime_adapter";
    const detail = status === "dependency_required" ? "需要先完成 FL2VA 流式基座"
      : preset.product?.ready
        ? runtimeAdapter ? "动态叠加拓扑与适配器清单已验证" : "产物清单与运行拓扑已验证"
        : status === "source_ready" ? runtimeAdapter ? "LoRA 与 SiLU 网格已就绪" : "原文件已就绪，等待切片"
          : runtimeAdapter ? "需要下载 LoRA 与 SiLU 网格" : "需要下载原文件";
    return `<article class="component-item">
      <strong title="${escapeHtml(preset.label)}">${escapeHtml(preset.label)}</strong>
      <span class="status-pill ${safeClass(status)}">${escapeHtml(componentStatusLabel(status))}</span>
      <p title="${escapeHtml(preset.product?.path || preset.output_dir || "")}">${escapeHtml(detail)} · ${formatBytes(preset.product?.size_bytes)}</p>
    </article>`;
  }).join("");
}

function renderModels() {
  const body = $("#modelsBody");
  $("#modelCount").textContent = `${state.models.length} 个模型资产`;
  $("#modelsEmpty").classList.toggle("hidden", state.models.length > 0 || !state.loaded.models);
  if (!state.loaded.models) {
    body.innerHTML = '<tr><td colspan="7"><div class="loading-state compact"><span class="spinner"></span>正在扫描原模型</div></td></tr>';
    return;
  }
  body.innerHTML = state.models.map((model, index) => {
    const product = model.record_type === "product";
    const supported = Boolean(model.export_supported);
    let scope = product ? '<span class="type-pill">已验证切片</span>' : '<span class="type-pill">Encoder + Decoder</span>';
    if (!product && model.component === "video_vae") {
      scope = `<select class="scope-select" data-model-index="${index}" aria-label="Video VAE 切片范围"><option value="0">Block 0 冒烟验证</option><option value="all">全部 36 Blocks</option></select>`;
    } else if (!product && ["fl2va_transformer", "ref2va_transformer"].includes(model.component)) {
      scope = `<select class="scope-select" data-model-index="${index}" aria-label="主模型切片范围"><option value="0">Block 0 冒烟验证</option><option value="all">全部 50 Blocks</option></select>`;
    } else if (!product && model.component === "text_encoder") {
      scope = `<span class="type-pill">全部 50 Layers</span>`;
    }
    const action = product
      ? '<span class="status-pill ready">已识别</span>'
      : `<button class="secondary-button table-action export-button" type="button" data-model-index="${index}" ${supported ? "" : "disabled"}>切片并验证</button>`;
    return `<tr>
      <td><span class="model-name" title="${escapeHtml(model.name)}">${escapeHtml(model.name)}</span><span class="model-path" title="${escapeHtml(model.id)}">${escapeHtml(model.id)}</span></td>
      <td><span class="type-pill">${escapeHtml(componentLabel(model.component))}</span></td>
      <td>${escapeHtml(model.dtype || "--")}</td>
      <td>${formatBytes(model.size_bytes)}</td>
      <td>${escapeHtml(model.tensor_count ?? "--")}</td>
      <td>${scope}</td>
      <td>${action}</td>
    </tr>`;
  }).join("");
}

function renderModelPage() {
  renderModelOverview();
  renderExportPresets();
  renderComponents();
  renderModels();
}

function videoVaeOutputFrames(latentFrames) {
  if (latentFrames === 1) return 1;
  const alignment = (5 - ((latentFrames + 3) % 5)) % 5;
  const paddedLength = latentFrames + 3 + alignment;
  let total = 0;
  let finalOverlap = 0;
  for (let index = 0; index < paddedLength / 5; index += 1) {
    const start = index * 5;
    const clipTokens = Math.max(0, Math.min(start + 7, paddedLength) - Math.min(start, paddedLength));
    const clipFrames = clipTokens * 4;
    total += Math.max(0, Math.min(20, clipFrames) - 3);
    finalOverlap = Math.max(0, Math.min(40, clipFrames) - 23);
  }
  total += finalOverlap;
  const beforeAlignment = paddedLength - alignment;
  for (let index = 0; index < alignment; index += 1) total -= (beforeAlignment + index) % 5 === 0 ? 1 : 4;
  return total;
}

function videoLatentFramesForOutput(outputFrames) {
  let latentFrames = 2;
  while (videoVaeOutputFrames(latentFrames) < outputFrames) latentFrames += 1;
  return latentFrames;
}

function temporalMode() {
  return $('input[name="temporalMode"]:checked')?.value || "segmented";
}

function conditioningMode() {
  return $('input[name="conditioningMode"]:checked')?.value || "text";
}

function activeFrameRoles(mode = conditioningMode()) {
  if (mode === "first") return ["start"];
  if (mode === "last") return ["end"];
  if (mode === "first_last") return ["start", "end"];
  return [];
}

function conditioningModeLabel(mode = conditioningMode()) {
  return { text: "纯文本", first: "首帧", last: "尾帧", first_last: "首尾帧" }[mode] || "纯文本";
}

function renderFrameConditions() {
  const mode = conditioningMode();
  const active = activeFrameRoles(mode);
  const inputs = $("#frameConditionInputs");
  inputs.classList.toggle("hidden", active.length === 0);

  for (const role of ["start", "end"]) {
    const item = state.frameImages[role];
    const enabled = active.includes(role);
    const panel = $(`#${role}FramePanel`);
    const preview = $(`#${role}FramePreview`);
    const placeholder = $(`#${role}FramePlaceholder`);
    const fileInput = $(`#${role}FrameFile`);
    const clearButton = $(`#${role}FrameClear`);
    const status = $(`#${role}FrameFileStatus`);
    panel.classList.toggle("hidden", !enabled);
    preview.classList.toggle("hidden", !item.previewUrl);
    placeholder.classList.toggle("hidden", Boolean(item.previewUrl));
    if (item.previewUrl && preview.src !== item.previewUrl) preview.src = item.previewUrl;
    if (!item.previewUrl) preview.removeAttribute("src");
    fileInput.disabled = item.uploading;
    clearButton.classList.toggle("hidden", !item.previewUrl && !item.path && !item.error);
    if (item.uploading) status.textContent = "正在上传";
    else if (item.error) status.textContent = "上传失败";
    else if (item.info) status.textContent = `${item.info.width} × ${item.info.height}`;
    else status.textContent = "等待输入";
    status.title = item.error || item.path || "";
    placeholder.textContent = item.uploading ? "正在上传图片" : item.error ? "图片不可用" : "未选择图片";
  }

  const ready = active.every((role) => state.frameImages[role].path && !state.frameImages[role].uploading);
  const uploading = active.some((role) => state.frameImages[role].uploading);
  const loaded = active.filter((role) => state.frameImages[role].path).length;
  $("#frameConditionStatus").textContent = active.length === 0
    ? "纯文本"
    : uploading
      ? "上传中"
      : ready
        ? `${conditioningModeLabel(mode)} · 已就绪`
        : `${loaded} / ${active.length} 张`;
  return { mode, active, ready, uploading };
}

function clearFrameImage(role) {
  const current = state.frameImages[role];
  if (current.previewUrl) URL.revokeObjectURL(current.previewUrl);
  state.frameImages[role] = {
    path: null,
    info: null,
    previewUrl: null,
    uploading: false,
    error: null,
    fileName: null,
    uploadId: ++state.frameUploadSerial,
  };
  $(`#${role}FrameFile`).value = "";
  renderProfiles();
}

async function uploadFrameImage(role, file) {
  if (!file) return;
  if (file.size > 32 * 1024 ** 2) {
    showToast("图片不能超过 32 MiB", true);
    return;
  }
  const previous = state.frameImages[role];
  if (previous.previewUrl) URL.revokeObjectURL(previous.previewUrl);
  const uploadId = ++state.frameUploadSerial;
  state.frameImages[role] = {
    path: null,
    info: null,
    previewUrl: URL.createObjectURL(file),
    uploading: true,
    error: null,
    fileName: file.name,
    uploadId,
  };
  renderProfiles();
  try {
    const payload = await api("/api/images/upload", {
      method: "POST",
      headers: fileUploadHeaders(file),
      body: file,
    });
    if (state.frameImages[role].uploadId !== uploadId) return;
    state.frameImages[role].path = payload.path;
    state.frameImages[role].info = payload.image;
    showToast(`${role === "start" ? "首帧" : "尾帧"}图片已上传`);
  } catch (error) {
    if (state.frameImages[role].uploadId !== uploadId) return;
    state.frameImages[role].error = error.message;
    showToast("图片上传失败：" + error.message, true);
  } finally {
    if (state.frameImages[role].uploadId === uploadId) {
      state.frameImages[role].uploading = false;
      renderProfiles();
    }
  }
}

function parsedTokenIds() {
  const raw = $("#tokenIdsInput").value.trim();
  if (!raw) return { values: [], valid: true };
  const parts = raw.split(",").map((value) => value.trim());
  if (parts.some((part) => !/^-?\d+$/.test(part))) return { values: [], valid: false, message: "Token IDs 必须是逗号分隔的整数" };
  const values = parts.map(Number);
  if (values.length > 192) return { values, valid: false, message: "Token IDs 不能超过 192 个" };
  if (values.some((value) => !Number.isSafeInteger(value))) return { values, valid: false, message: "Token ID 超出整数范围" };
  if (values.some((value) => value < 0)) return { values, valid: false, message: "Token ID 不能为负数" };
  return { values, valid: true };
}

function preflightRow(label, detail, stateName, trailing) {
  const icon = stateName === "passed" ? "✓" : stateName === "warning" ? "!" : "×";
  const className = stateName === "passed" ? "" : stateName;
  return `<div class="preflight-row ${className}"><span class="preflight-icon">${icon}</span><div><strong>${escapeHtml(label)}</strong><small title="${escapeHtml(detail)}">${escapeHtml(detail)}</small></div><span>${escapeHtml(trailing)}</span></div>`;
}

function renderProfiles() {
  const profile = state.profiles[0];
  const provider = $("#inferenceProvider");
  const prompt = $("#promptInput").value.trim();
  const tokens = parsedTokenIds();
  const frameCondition = renderFrameConditions();
  $("#promptCounter").textContent = `${$("#promptInput").value.length} / 4000`;

  if (!profile) {
    provider.className = "header-status neutral";
    provider.querySelector("span:last-child").textContent = state.loaded.profiles ? "没有可用配置" : "正在预检";
    $("#generateButton").disabled = true;
    $("#preflightList").innerHTML = state.loaded.profiles
      ? preflightRow("推理配置", "服务端没有返回生成配置", "failed", "阻塞")
      : '<div class="loading-state compact"><span class="spinner"></span>正在检查</div>';
    $("#profileGrid").innerHTML = "";
    return;
  }

  const width = clamp(finite($("#widthInput").value) || profile.output_width || 640, 128, 1024);
  const height = clamp(finite($("#heightInput").value) || profile.output_height || 360, 128, 1024);
  const duration = clamp(finite($("#durationInput").value) || profile.frames / profile.fps, 0.1, 15);
  const steps = clamp(finite($("#stepsInput").value) || 4, 1, 50);
  const paddedWidth = Math.ceil(width / 32) * 32;
  const paddedHeight = Math.ceil(height / 32) * 32;
  const targetFrames = Math.max(1, Math.round(duration * profile.fps));
  const mode = temporalMode();
  const segments = mode === "segmented" ? Math.ceil(targetFrames / profile.frames) : 1;
  const turboSteps = steps >= 4 && steps <= 8;
  const accelerationRequested = $("#accelerationLoraInput").checked;
  const baseReady = Boolean(profile.main_ready);
  const adapterReady = Boolean(profile.acceleration_ready);
  const accelerationActive = Boolean(accelerationRequested && baseReady && adapterReady && turboSteps);
  const latentFrames = mode === "native" ? videoLatentFramesForOutput(targetFrames) : profile.video_latent_frames;
  const audioTokens = mode === "native" ? Math.ceil(targetFrames * 40 / profile.fps) * 2 : profile.audio_tokens;
  const videoTokens = latentFrames * (paddedHeight / 32) * (paddedWidth / 32);
  const sequenceTokens = profile.text_tokens + audioTokens + videoTokens;
  const qkvCpuBytes = sequenceTokens * 56 * 384 * 2;
  const kvGpuBytes = sequenceTokens * 56 * 128 * 2 * 2;
  const queryChunk = Number($("#queryChunkSelect").value) || 512;
  const videoReady = presetProductReady("video_vae");
  const audioReady = presetProductReady("audio_vae");
  const contentReady = Boolean(prompt || tokens.values.length) && tokens.valid;
  const tokenizerReady = !prompt || Boolean(profile.tokenizer_ready);
  const mainReady = baseReady;
  const stepScheduleReady = !accelerationRequested || (turboSteps && accelerationActive);
  const dimensionsReady = width >= 128 && width <= 1024 && height >= 128 && height <= 1024;
  const componentsKnown = state.loaded.presets;
  const mediaReady = !componentsKnown || (videoReady === true && audioReady === true);
  const frameConditionReady = frameCondition.ready;
  const canStart = Boolean(profile.generation_ready && mainReady && stepScheduleReady && contentReady && tokenizerReady && dimensionsReady && mediaReady && frameConditionReady);

  provider.className = `header-status ${profile.cuda_provider_available ? "" : "warning"}`.trim();
  provider.querySelector("span:last-child").textContent = profile.cuda_provider_available ? "CUDA 推理" : "CPU 推理（较慢）";

  const checks = [
    ["执行后端", profile.cuda_provider_available ? "CUDAExecutionProvider 可用" : "将使用 CPUExecutionProvider", profile.cuda_provider_available ? "passed" : "warning", profile.cuda_provider_available ? "CUDA" : "CPU"],
    ["Qwen 文本编码器", profile.qwen_ready ? "验证产物可用" : "缺少或未通过验证", profile.qwen_ready ? "passed" : "failed", profile.qwen_ready ? "通过" : "阻塞"],
    ["FL2VA 流式基座", mainReady ? "50 Block 基座产物可用" : "缺少或未通过验证", mainReady ? "passed" : "failed", mainReady ? "通过" : "阻塞"],
    ["Turbo v4 动态 LoRA", !accelerationRequested ? "手动关闭，使用流式基座" : !turboSteps ? "已开启，但仅支持 4–8 步" : adapterReady ? "手动开启，将动态叠加" : "已开启，但运行时适配器未就绪", stepScheduleReady ? "passed" : "failed", accelerationActive ? "启用" : stepScheduleReady ? "未启用" : "阻塞"],
    ["Video / Audio VAE", !componentsKnown ? "组件状态尚未返回" : mediaReady ? "编码与解码组件均可用" : "缺少 Video VAE 或 Audio VAE", !componentsKnown ? "warning" : mediaReady ? "passed" : "failed", !componentsKnown ? "待检查" : mediaReady ? "通过" : "阻塞"],
    ["文本输入", contentReady && tokenizerReady ? (prompt ? "Tokenizer 与提示词可用" : `${tokens.values.length} 个 Token IDs`) : !tokens.valid ? tokens.message : prompt ? "Tokenizer 文件不完整" : "请输入提示词或 Token IDs", contentReady && tokenizerReady ? "passed" : "failed", contentReady && tokenizerReady ? "通过" : "阻塞"],
    ["帧条件", frameCondition.active.length === 0 ? "纯文本模式，不使用图片条件" : frameConditionReady ? `${conditioningModeLabel(frameCondition.mode)}条件已上传` : frameCondition.uploading ? "条件图片正在上传" : `还需 ${frameCondition.active.filter((role) => !state.frameImages[role].path).length} 张条件图片`, frameConditionReady ? "passed" : "failed", frameConditionReady ? conditioningModeLabel(frameCondition.mode) : "阻塞"],
    ["输出规格", dimensionsReady ? `${width} × ${height}，${targetFrames} 帧` : "宽高必须在 128–1024 之间", dimensionsReady ? "passed" : "failed", dimensionsReady ? "通过" : "阻塞"],
  ];
  $("#preflightList").innerHTML = checks.map((row) => preflightRow(...row)).join("");
  const passed = checks.filter((row) => row[2] === "passed").length;
  $("#preflightSummary").textContent = `${passed} / ${checks.length} 通过`;

  const rows = [
    ["输出", `${width} × ${height} / ${targetFrames} 帧`],
    ["预计时长", `${(targetFrames / profile.fps).toFixed(2)} 秒 @ ${profile.fps} FPS`],
    ["主模型", "FL2VA 流式基座"],
    ["运行时适配", accelerationActive ? "Turbo v4 动态 LoRA" : "未启用"],
    ["生成模式", conditioningModeLabel(frameCondition.mode)],
    ["时序", mode === "native" ? `${latentFrames} latent / 整体` : `${segments} 段 / ${segments * steps} 次去噪`],
    ["注意力", `${sequenceTokens} tokens / query ${queryChunk}`],
    ["CPU QKV 缓存", formatBytes(qkvCpuBytes)],
    ["GPU K/V 缓存", formatBytes(kvGpuBytes)],
  ];
  $("#profileGrid").innerHTML = rows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");

  const notices = [];
  if (!profile.cuda_provider_available) notices.push("当前 ONNX Runtime 没有 CUDAExecutionProvider，CPU 执行会非常缓慢。");
  if (accelerationRequested && !turboSteps) notices.push("Turbo v4 加速 LoRA 仅支持 4–8 步，请调整采样步数或关闭加速 LoRA。");
  if (accelerationRequested && turboSteps && !adapterReady) notices.push("已手动开启 Turbo v4 加速 LoRA，但适配器未就绪，请先在模型页生成适配器。");
  if (!baseReady && adapterReady) notices.push("动态 LoRA 不能独立运行，请先完成 FL2VA 流式基座。");
  if (mode === "segmented" && segments > 1) notices.push("分段模式尚未接入首帧续接，多段画面可能出现跳变。");
  if (frameCondition.mode === "first_last" && mode === "segmented" && segments > 1) notices.push("首尾条件分别作用于第一段和最后一段；整体长序列能提供更强的双端关联。");
  if (mode === "native" && targetFrames > profile.frames) notices.push("长序列 Video VAE 会使用 GPU 时间窗口解码。");
  if (steps < 4) notices.push("当前步数使用流式基座；低步数可能导致画面语义或音频数值不稳定。");
  const warning = $("#runtimeWarning");
  warning.classList.toggle("hidden", notices.length === 0);
  warning.textContent = notices.join(" ");

  $("#generateButton").disabled = !canStart;
  $("#submitSummary").textContent = canStart ? `${width} × ${height} · ${targetFrames} 帧 · ${steps} 步` : "预检尚未通过";
  $("#submitHint").textContent = canStart ? `${accelerationActive ? "基座 + Turbo v4 动态 LoRA" : "流式基座"} · ${mode === "native" ? "整体长序列" : "低显存分段"}` : "处理所有阻塞项后即可启动";
}

function updateResolutionPreset() {
  const value = $("#resolutionPreset").value;
  const custom = value === "custom";
  $("#widthInput").disabled = !custom;
  $("#heightInput").disabled = !custom;
  if (!custom) {
    const [width, height] = value.split("x").map(Number);
    $("#widthInput").value = String(width);
    $("#heightInput").value = String(height);
  }
  renderProfiles();
}

function renderSuperResolution() {
  const provider = $("#superResolutionProvider");
  const info = state.superVideo;
  const profile = state.profiles[0];
  const prompt = $("#superPromptInput").value.trim();
  const scale = Number($("#superScaleInput").value);
  const steps = Number($("#superStepsInput").value);
  const noise = Number($("#superNoiseInput").value);
  const processingMode = document.querySelector('input[name="superProcessingMode"]:checked')?.value || "segmented";
  $("#superNoiseValue").textContent = noise.toFixed(2);
  $("#superPromptCounter").textContent = $("#superPromptInput").value.length + " / 4000";

  if (!info) {
    provider.className = "header-status neutral";
    provider.querySelector("span:last-child").textContent = state.superUploadActive ? "正在上传视频" : "等待输入视频";
    $("#superVideoProfile").innerHTML = "<div><dt>输入</dt><dd>--</dd></div>";
    $("#superPreflightList").innerHTML = '<div class="loading-state compact">等待视频探测</div>';
    $("#superPreflightSummary").textContent = "--";
    $("#superGenerateButton").disabled = true;
    $("#superSubmitSummary").textContent = "等待输入视频";
    $("#superSubmitHint").textContent = "探测视频后显示输出计划";
    return;
  }
  if (!profile) {
    provider.className = "header-status neutral";
    provider.querySelector("span:last-child").textContent = "正在读取模型配置";
    $("#superGenerateButton").disabled = true;
    $("#superSubmitSummary").textContent = "正在读取模型配置";
    $("#superSubmitHint").textContent = "模型配置返回后显示启动预检";
    return;
  }

  const outputWidth = Math.round(info.width * scale);
  const outputHeight = Math.round(info.height * scale);
  const dimensionsReady = outputWidth <= 2048 && outputHeight <= 2048 && outputWidth * outputHeight <= 4194304;
  const turboSteps = steps >= 4 && steps <= 8;
  const accelerationRequested = $("#superAccelerationLoraInput").checked;
  const adapterReady = Boolean(profile && profile.acceleration_ready);
  const baseReady = Boolean(profile && profile.main_ready);
  const promptReady = Boolean(prompt || info.prompt);
  const videoReady = Boolean(profile && profile.video_vae_ready) && presetProductReady("video_vae");
  const tokenizerReady = Boolean(profile && profile.tokenizer_ready);
  const componentsKnown = state.loaded.profiles && state.loaded.presets;
  const contentReady = promptReady && tokenizerReady;
  const accelerationReady = !accelerationRequested || (turboSteps && adapterReady);
  const processingReady = processingMode !== "direct" || info.frames <= 360;
  const canStart = Boolean(
    componentsKnown && profile && baseReady && videoReady && contentReady && dimensionsReady && accelerationReady && processingReady,
  );
  const fps = finite(info.fps);
  const duration = fps ? info.frames / fps : info.duration_seconds;
  $("#superVideoProfile").innerHTML = [
    ["输入", info.width + " × " + info.height + " · " + info.frames + " 帧"],
    ["输出", outputWidth + " × " + outputHeight + " · " + info.frames + " 帧"],
    ["帧率", (fps === null ? "--" : fps.toFixed(3)) + " FPS"],
    ["时长", duration.toFixed(2) + " 秒"],
    ["音频", info.has_audio ? "保留原音轨" : "无音轨"],
    ["提示词", info.prompt ? (info.prompt_source || "metadata") : "需手动输入"],
  ].map(([key, value]) => "<div><dt>" + escapeHtml(key) + "</dt><dd title=\"" + escapeHtml(value) + "\">" + escapeHtml(value) + "</dd></div>").join("");

  provider.className = "header-status " + (profile.cuda_provider_available ? "" : "warning");
  provider.querySelector("span:last-child").textContent = profile.cuda_provider_available ? "CUDA 推理" : "CPU 推理（较慢）";
  const checks = [
    ["执行后端", profile.cuda_provider_available ? "CUDAExecutionProvider 可用" : "将使用 CPUExecutionProvider", profile.cuda_provider_available ? "passed" : "warning", profile.cuda_provider_available ? "CUDA" : "CPU"],
    ["Qwen 文本编码器", profile.qwen_ready ? "验证产物可用" : "缺少或未通过验证", profile.qwen_ready ? "passed" : "failed", profile.qwen_ready ? "通过" : "阻塞"],
    ["FL2VA 流式基座", baseReady ? "50 Block 基座产物可用" : "缺少或未通过验证", baseReady ? "passed" : "failed", baseReady ? "通过" : "阻塞"],
    ["Video VAE", videoReady ? "编码与解码产物可用" : "缺少或未通过验证", videoReady ? "passed" : "failed", videoReady ? "通过" : "阻塞"],
    ["条件提示词", contentReady ? (prompt ? "使用手动提示词" : "使用视频 metadata prompt") : tokenizerReady ? "未找到 prompt" : "Tokenizer 文件不完整", contentReady ? "passed" : "failed", contentReady ? "通过" : "阻塞"],
    ["输出规格", dimensionsReady ? outputWidth + " × " + outputHeight : "超过 2048 px 或 4 MP 限制", dimensionsReady ? "passed" : "failed", dimensionsReady ? "通过" : "阻塞"],
    ["处理方式", processingReady ? (processingMode === "direct" ? "完整视频直接推理" : "17 帧分片推理") : "直接推理最多支持 360 帧", processingReady ? "passed" : "failed", processingReady ? "通过" : "阻塞"],
    ["Turbo v4 LoRA", !accelerationRequested ? "手动关闭" : !turboSteps ? "仅支持 4–8 步" : adapterReady ? "手动开启，适配器可用" : "适配器未就绪", accelerationReady ? "passed" : "failed", accelerationReady && accelerationRequested ? "启用" : accelerationReady ? "关闭" : "阻塞"],
  ];
  $("#superPreflightList").innerHTML = checks.map((row) => preflightRow(...row)).join("");
  $("#superPreflightSummary").textContent = checks.filter((row) => row[2] === "passed").length + " / " + checks.length + " 通过";

  const segmentFrames = 17;
  const segments = Math.ceil(info.frames / segmentFrames);
  $("#superSubmitSummary").textContent = canStart ? outputWidth + " × " + outputHeight + " · " + info.frames + " 帧 · " + steps + " 步" : "预检尚未通过";
  $("#superSubmitHint").textContent = canStart
    ? (processingMode === "direct" ? "完整视频直接推理" : segments + " 段视频分片") + " · 噪声 " + noise.toFixed(2) + " · " + $("#superInterpolationInput").value
    : "处理所有阻塞项后即可启动";
  const warnings = [];
  if (outputWidth > 1024 || outputHeight > 1024) warnings.push("输出超过 1024 px，RTX 3050 需要较小 Query 分块并可能显著变慢。");
  if (outputWidth * outputHeight > 1048576) warnings.push("当前输出超过 1 MP，任务会把输入帧驻留主内存。");
  if (processingMode === "direct") warnings.push("直接推理会把完整视频放入一个 FL2VA 时间序列，显存和注意力开销显著增加。");
  if (!processingReady) warnings.push("直接推理最多支持 360 帧，请切换到视频分片。");
  if (accelerationRequested && !turboSteps) warnings.push("Turbo v4 加速 LoRA 仅支持 4–8 步。");
  if (accelerationRequested && turboSteps && !adapterReady) warnings.push("已开启 Turbo v4，但适配器尚未就绪。");
  const warning = $("#superRuntimeWarning");
  warning.classList.toggle("hidden", warnings.length === 0);
  warning.textContent = warnings.join(" ");
  $("#superGenerateButton").disabled = !canStart || state.superUploadActive;
}

async function probeSuperVideo() {
  const path = $("#superSourcePath").value.trim();
  if (!path) {
    showToast("请输入工作区内的视频路径，或先选择本机视频", true);
    return;
  }
  const button = $("#superProbeButton");
  button.disabled = true;
  try {
    const info = await api("/api/media/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    state.superVideo = info;
    if (!state.superPromptEdited && !$("#superPromptInput").value.trim() && info.prompt) {
      $("#superPromptInput").value = info.prompt;
    }
    $("#superPromptSource").textContent = info.prompt ? "来源：" + (info.prompt_source || "metadata") : "未找到 metadata prompt";
    $("#superVideoNotice").classList.add("hidden");
    renderSuperResolution();
  } catch (error) {
    state.superVideo = null;
    const notice = $("#superVideoNotice");
    notice.classList.remove("hidden");
    notice.textContent = error.message;
    renderSuperResolution();
    showToast("视频探测失败：" + error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function uploadSuperVideo(file) {
  if (!file) return;
  state.superUploadActive = true;
  renderSuperResolution();
  try {
    const payload = await api("/api/media/upload", {
      method: "POST",
      headers: fileUploadHeaders(file),
      body: file,
    });
    $("#superSourcePath").value = payload.path;
    state.superVideo = payload.video;
    if (!state.superPromptEdited && !$("#superPromptInput").value.trim() && payload.video.prompt) {
      $("#superPromptInput").value = payload.video.prompt;
    }
    $("#superPromptSource").textContent = payload.video.prompt ? "来源：" + (payload.video.prompt_source || "metadata") : "未找到 metadata prompt";
    showToast("视频已上传并完成元数据探测");
  } catch (error) {
    showToast("视频上传失败：" + error.message, true);
  } finally {
    state.superUploadActive = false;
    renderSuperResolution();
  }
}

async function startSuperResolution(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  if (!state.superVideo) {
    showToast("请先探测输入视频", true);
    return;
  }
  const prompt = $("#superPromptInput").value.trim();
  if (!prompt && !state.superVideo.prompt) {
    showToast("视频没有 metadata prompt，请手动输入提示词", true);
    return;
  }
  const button = $("#superGenerateButton");
  button.disabled = true;
  try {
    const job = await api("/api/jobs/super-resolution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_path: $("#superSourcePath").value.trim(),
        prompt: prompt || null,
        scale: Number($("#superScaleInput").value),
        interpolation: $("#superInterpolationInput").value,
        noise_strength: Number($("#superNoiseInput").value),
        processing_mode: document.querySelector('input[name="superProcessingMode"]:checked')?.value || "segmented",
        steps: Number($("#superStepsInput").value),
        use_acceleration_lora: $("#superAccelerationLoraInput").checked,
        seed: Number($("#superSeedInput").value),
        attention_query_chunk: Number($("#superQueryChunkSelect").value),
        l1_prefetch_shards: Number($("#superL1PrefetchSelect").value),
      }),
    });
    state.selectedJobId = job.id;
    showToast("超分任务已加入队列");
    await loadJobs();
    switchPage("tasks");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    renderSuperResolution();
  }
}

function activitySummary(job) {
  const activity = job.activity || {};
  const module = activity.module || jobKindLabel(job);
  const operation = activity.operation || job.message || jobStatusLabel(job.status);
  return `${module} · ${operation}`;
}

function rememberJobEvents(jobs) {
  for (const job of jobs) {
    const activity = job.activity || {};
    const prefetch = job.prefetch || {};
    const signature = JSON.stringify([
      job.status,
      Math.round((finite(job.progress) || 0) * 1000),
      job.message,
      activity.phase,
      activity.module,
      activity.operation,
      activity.current,
      activity.total,
      prefetch.operation,
    ]);
    if (state.jobSignatures.get(job.id) === signature) continue;
    state.jobSignatures.set(job.id, signature);
    const events = state.jobEvents.get(job.id) || [];
    let message = activitySummary(job);
    if (prefetch.operation) message += `；预取：${prefetch.operation}`;
    events.push({
      time: new Date().toISOString(),
      message,
      error: job.status === "failed",
    });
    state.jobEvents.set(job.id, events.slice(-80));
  }
}

function jobMatchesFilter(job) {
  if (state.jobFilter === "all") return true;
  if (state.jobFilter === "active") return isActiveJob(job);
  return job.status === state.jobFilter;
}

function renderTaskSummary() {
  const counts = { running: 0, queued: 0, completed: 0, failed: 0 };
  state.jobs.forEach((job) => { if (Object.hasOwn(counts, job.status)) counts[job.status] += 1; });
  $("#runningCount").textContent = String(counts.running);
  $("#queuedCount").textContent = String(counts.queued);
  $("#completedCount").textContent = String(counts.completed);
  $("#failedCount").textContent = String(counts.failed);
  $("#runningBadge").textContent = String(counts.running + counts.queued);
  $("#runningBadge").classList.toggle("hidden", counts.running + counts.queued === 0);
}

function renderJobList() {
  const jobs = state.jobs.filter(jobMatchesFilter);
  $("#jobCount").textContent = `${jobs.length} / ${state.jobs.length} 个任务`;
  $("#jobsEmpty").classList.toggle("hidden", jobs.length > 0 || !state.loaded.jobs);
  const list = $("#jobsList");
  list.classList.toggle("hidden", state.loaded.jobs && jobs.length === 0);
  $("#jobInspector").classList.toggle("hidden", state.loaded.jobs && jobs.length === 0);
  if (!state.loaded.jobs) {
    list.innerHTML = '<div class="loading-state"><span class="spinner"></span>正在读取任务</div>';
    return;
  }
  if (jobs.length === 0) {
    const filtered = state.jobs.length > 0;
    $("#jobsEmpty").innerHTML = filtered
      ? '<strong>没有匹配的任务</strong><span>切换筛选条件可查看其他任务。</span>'
      : '<strong>暂无任务</strong><span>从模型页或推理页启动一个任务后，进度会显示在这里。</span>';
    list.innerHTML = "";
    return;
  }
  if (!jobs.some((job) => job.id === state.selectedJobId)) {
    state.selectedJobId = jobs.find(isActiveJob)?.id || jobs[0].id;
  }
  list.innerHTML = jobs.map((job) => {
    const progress = clamp(finite(job.progress) || 0, 0, 1);
    const status = safeClass(job.status);
    return `<button class="job-list-item ${job.id === state.selectedJobId ? "selected" : ""}" type="button" data-job-id="${escapeHtml(job.id)}">
      <div class="job-list-top"><span class="job-kind" title="${escapeHtml(job.model_id)}">${escapeHtml(jobKindLabel(job))} · ${escapeHtml(job.model_id)}</span><span class="status-pill ${status}">${escapeHtml(jobStatusLabel(job.status))}</span></div>
      <div class="job-stage-line"><span title="${escapeHtml(activitySummary(job))}">${escapeHtml(activitySummary(job))}</span><strong class="job-progress-value">${Math.round(progress * 100)}%</strong></div>
      <div class="progress-track"><div class="progress-bar ${job.status === "failed" ? "failed" : ""}" style="width:${(progress * 100).toFixed(1)}%"></div></div>
      <div class="job-list-meta"><span class="job-id">${escapeHtml(job.id)}</span><span data-job-elapsed="${escapeHtml(job.id)}">耗时 ${jobElapsed(job)}</span></div>
    </button>`;
  }).join("");
}

function activityItems(job) {
  const activity = job.activity || {};
  const prefetch = job.prefetch || {};
  const performance = job.performance?.performance || {};
  const gpu = performance.gpu || {};
  const items = [];
  const push = (label, value) => { if (value !== null && value !== undefined && value !== "") items.push([label, value]); };
  push("当前模块", activity.module);
  push("当前操作", activity.operation);
  if (activity.current !== undefined && activity.total !== undefined) push("执行位置", `${activity.current} / ${activity.total}`);
  if (activity.sampling_step) push("采样步数", `${activity.sampling_step} / ${activity.sampling_steps}`);
  if (activity.segment) push("视频片段", `${activity.segment} / ${activity.segments}`);
  if (activity.shard) push("当前分片", `${activity.shard} / ${activity.shards}`);
  if (activity.provider) push("执行后端", String(activity.provider).replace("ExecutionProvider", ""));
  if (activity.elapsed_seconds !== undefined) push("本片耗时", `${Number(activity.elapsed_seconds).toFixed(2)} 秒`);
  if (prefetch.operation) push("预取状态", prefetch.operation);
  if (prefetch.l1_prefetch_hits !== undefined) push("L1 命中 / 等待", `${prefetch.l1_prefetch_hits} / ${prefetch.l1_prefetch_waits || 0}`);
  const execution = [];
  if (activity.qkv_chunk_tokens) execution.push(`QKV ${activity.qkv_chunk_tokens}`);
  if (activity.attention_output_chunk_tokens) execution.push(`Attention ${activity.attention_output_chunk_tokens}`);
  if (activity.mlp_chunk_tokens) execution.push(`MLP ${activity.mlp_chunk_tokens}`);
  if (activity.chunk_io_binding) execution.push("I/O Binding");
  if (execution.length) push("执行参数", execution.join(" · "));
  const cache = [];
  if (activity.l1_sessions) cache.push(`L1 ${activity.l1_sessions} 片`);
  if (activity.l2_budget_bytes) cache.push(`L2 ${formatBytes(activity.l2_staged_bytes)} / ${formatBytes(activity.l2_budget_bytes)}`);
  if (prefetch.prefetch_ahead !== undefined) cache.push(`向前 ${prefetch.prefetch_ahead} 片`);
  if (cache.length) push("缓存与预取", cache.join(" · "));
  if (finite(gpu.utilization_percent) !== null) push("GPU / CPU", `${formatPercent(gpu.utilization_percent)} / ${formatPercent(performance.system_cpu_percent)}`);
  if (finite(gpu.memory_used_mib) !== null) push("显存", `${formatBytes(gpu.memory_used_mib * 1024 ** 2)} 已用`);
  if (finite(performance.disk_read_bytes_per_second) !== null) push("磁盘读取", formatRate(performance.disk_read_bytes_per_second));
  return items;
}

function artifactLinks(job) {
  if (!["inference", "super_resolution"].includes(job.kind)) return "";
  const links = [];
  if (job.status === "completed") links.push(`<a class="link-button" href="/api/jobs/${encodeURIComponent(job.id)}/output">↓ 下载 MP4</a>`);
  if (job.result?.metadata) links.push(`<a class="link-button" href="/api/jobs/${encodeURIComponent(job.id)}/metadata">{} 元数据</a>`);
  if (job.performance_log) links.push(`<a class="link-button" href="/api/jobs/${encodeURIComponent(job.id)}/performance">↗ 性能日志</a>`);
  return links.join("");
}

function renderJobInspector() {
  const inspector = $("#jobInspector");
  const job = state.jobs.find((item) => item.id === state.selectedJobId);
  if (!job) {
    inspector.innerHTML = '<div class="inspector-empty"><span aria-hidden="true">≡</span><strong>选择一个任务</strong><p>查看当前阶段、缓存状态、运行日志和产物。</p></div>';
    return;
  }
  const progress = clamp(finite(job.progress) || 0, 0, 1);
  const items = activityItems(job);
  const events = state.jobEvents.get(job.id) || [];
  const links = artifactLinks(job);
  const eventHtml = events.length
    ? events.map((event) => `<div class="log-line ${event.error ? "error" : ""}"><span class="log-time">${escapeHtml(shortTime(event.time))}</span><span>${escapeHtml(event.message)}</span></div>`).join("")
    : '<div class="log-line"><span class="log-time">--:--:--</span><span>等待任务状态更新</span></div>';
  inspector.innerHTML = `<div class="inspector-content">
    <header class="inspector-header">
      <div class="inspector-title"><h4 title="${escapeHtml(job.model_id)}">${escapeHtml(job.model_id)}</h4><span class="status-pill ${safeClass(job.status)}">${escapeHtml(jobStatusLabel(job.status))}</span></div>
      <p>${escapeHtml(job.message || activitySummary(job))}</p>
      <div class="inspector-progress"><div class="progress-track"><div class="progress-bar ${job.status === "failed" ? "failed" : ""}" style="width:${(progress * 100).toFixed(1)}%"></div></div><strong class="job-progress-value">${Math.round(progress * 100)}%</strong></div>
    </header>
    <div class="inspector-meta">
      <div><span>任务类型</span><strong>${escapeHtml(jobKindLabel(job))}</strong></div>
      <div><span>任务 ID</span><strong title="${escapeHtml(job.id)}">${escapeHtml(job.id)}</strong></div>
      <div><span>累计耗时</span><strong data-job-elapsed="${escapeHtml(job.id)}">${jobElapsed(job)}</strong></div>
    </div>
    <section class="activity-panel"><h5 class="inspector-section-title">实时性能与执行参数</h5><div class="activity-grid">${items.length
      ? items.map(([label, value]) => `<div class="activity-item"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`).join("")
      : '<div class="activity-item"><span>状态</span><strong>等待执行信息</strong></div>'}</div></section>
    <section class="log-panel"><h5 class="inspector-section-title">任务日志</h5><div class="log-viewer" id="jobLogViewer">${eventHtml}</div>${job.error ? `<pre class="error-trace">${escapeHtml(job.error)}</pre>` : ""}</section>
    ${links ? `<section class="artifact-panel"><h5 class="inspector-section-title">任务产物</h5><div class="artifact-links">${links}</div></section>` : ""}
  </div>`;
  const log = $("#jobLogViewer");
  if (log) log.scrollTop = log.scrollHeight;
}

function renderJobs() {
  rememberJobEvents(state.jobs);
  renderTaskSummary();
  renderJobList();
  renderJobInspector();
  renderModelOverview();
  renderExportPresets();
  updateJobTimers();
  if (state.lastJobUpdate) $("#lastUpdated").textContent = `${shortTime(state.lastJobUpdate)} 更新`;
}

function updateJobTimers() {
  const now = Date.now();
  $$('[data-job-elapsed]').forEach((element) => {
    const job = state.jobs.find((item) => item.id === element.dataset.jobElapsed);
    if (!job) return;
    const value = jobElapsed(job, now);
    element.textContent = element.closest(".job-list-item") ? `耗时 ${value}` : value;
  });
}

function normalizeTelemetry(payload, source) {
  if (!payload || typeof payload !== "object") return null;
  const envelope = payload.sample && typeof payload.sample === "object" ? payload.sample : payload;
  const performance = envelope.performance && typeof envelope.performance === "object" ? envelope.performance : envelope;
  const gpuGroup = performance.gpu && typeof performance.gpu === "object"
    ? performance.gpu
    : envelope.gpu && typeof envelope.gpu === "object" ? envelope.gpu : {};
  const gpu = Array.isArray(gpuGroup.devices) && gpuGroup.devices.length ? gpuGroup.devices[0] : gpuGroup;
  const cpu = performance.cpu && typeof performance.cpu === "object" ? performance.cpu : {};
  const memory = performance.memory && typeof performance.memory === "object" ? performance.memory : {};
  const disk = performance.disk && typeof performance.disk === "object" ? performance.disk : {};
  const process = performance.process && typeof performance.process === "object" ? performance.process : {};
  const system = state.system || {};
  const vramUsedMib = finite(gpu.memory_used_mib ?? gpu.used_memory_mib ?? performance.gpu_memory_used_mib)
    ?? (finite(gpu.memory_used_bytes) !== null ? gpu.memory_used_bytes / 1024 ** 2 : null);
  const vramFreeMib = finite(gpu.memory_free_mib ?? gpu.free_memory_mib ?? performance.gpu_memory_free_mib)
    ?? (finite(gpu.memory_free_bytes) !== null ? gpu.memory_free_bytes / 1024 ** 2 : null);
  const vramTotalMib = finite(gpu.memory_total_mib ?? performance.gpu_memory_total_mib)
    ?? (finite(gpu.memory_total_bytes) !== null ? gpu.memory_total_bytes / 1024 ** 2 : null)
    ?? (vramUsedMib !== null && vramFreeMib !== null ? vramUsedMib + vramFreeMib : finite(system.gpu?.memory_bytes) !== null ? system.gpu.memory_bytes / 1024 ** 2 : null);
  const ramAvailable = finite(memory.available_bytes ?? performance.memory_available_bytes ?? envelope.memory_available_bytes);
  const ramTotal = finite(memory.total_bytes ?? performance.memory_total_bytes ?? envelope.memory_total_bytes ?? system.memory_total_bytes);
  const ramPercent = finite(memory.percent ?? performance.memory_percent ?? envelope.memory_percent)
    ?? (ramAvailable !== null && ramTotal ? (1 - ramAvailable / ramTotal) * 100 : null);
  const vramCapacityPercent = vramUsedMib !== null && vramTotalMib ? vramUsedMib / vramTotalMib * 100 : null;
  const sample = {
    timestamp: envelope.timestamp || payload.timestamp || new Date().toISOString(),
    sequence: envelope.sequence ?? payload.sequence ?? null,
    source,
    gpuUtil: finite(gpu.utilization_percent ?? gpu.utilization_gpu_percent ?? performance.gpu_utilization_percent),
    cpuUtil: finite(cpu.system_percent ?? performance.system_cpu_percent ?? envelope.system_cpu_percent ?? performance.cpu_percent),
    vramUsedMib,
    vramTotalMib,
    vramPercent: finite(gpu.memory_percent) ?? vramCapacityPercent,
    ramAvailable,
    ramTotal,
    ramPercent,
    diskRead: finite(disk.read_bytes_per_second ?? performance.disk_read_bytes_per_second ?? envelope.disk_read_bytes_per_second),
    diskWrite: finite(disk.write_bytes_per_second ?? performance.disk_write_bytes_per_second ?? envelope.disk_write_bytes_per_second),
    processRead: finite(process.read_bytes_per_second ?? disk.process_read_bytes_per_second ?? performance.process_read_bytes_per_second),
    powerWatts: finite(gpu.power_watts),
    temperatureC: finite(gpu.temperature_c),
  };
  const hasMetric = [sample.gpuUtil, sample.cpuUtil, sample.vramPercent, sample.ramPercent, sample.diskRead, sample.diskWrite].some((value) => value !== null);
  return hasMetric ? sample : null;
}

function addTelemetrySample(sample) {
  if (!sample) return;
  const last = state.telemetry.at(-1);
  const identity = sample.sequence !== null ? `${sample.source}:${sample.sequence}` : `${sample.source}:${sample.timestamp}`;
  const lastIdentity = last ? (last.sequence !== null ? `${last.source}:${last.sequence}` : `${last.source}:${last.timestamp}`) : null;
  if (identity === lastIdentity) return;
  state.telemetry.push(sample);
  if (state.telemetry.length > TELEMETRY_MAX_POINTS) state.telemetry.splice(0, state.telemetry.length - TELEMETRY_MAX_POINTS);
  state.telemetrySource = sample.source;
  renderTelemetry();
}

function latestJobTelemetry() {
  const job = state.jobs.find((item) => item.status === "running" && item.performance?.performance)
    || state.jobs.find((item) => item.performance?.performance);
  return job ? normalizeTelemetry(job.performance, "任务性能快照") : null;
}

function renderTelemetry() {
  const sample = state.telemetry.at(-1);
  const live = $("#telemetryLive");
  live.className = "live-indicator idle";
  if (sample) {
    const age = Date.now() - Date.parse(sample.timestamp);
    if (state.telemetryError) live.className = "live-indicator error";
    else if (!Number.isFinite(age) || age < 6000) live.className = "live-indicator live";
    live.querySelector("span").textContent = state.telemetryError ? "降级" : live.classList.contains("live") ? "实时" : "已暂停";
    $("#telemetrySource").textContent = state.telemetryError
      ? `${state.telemetryError} · 显示 ${shortTime(sample.timestamp)} 的最后采样`
      : `${sample.source} · ${shortTime(sample.timestamp)} · ${sample.powerWatts !== null ? `${sample.powerWatts.toFixed(0)} W` : "功耗 --"}${sample.temperatureC !== null ? ` · ${sample.temperatureC.toFixed(0)} °C` : ""}`;
    $("#utilizationValue").textContent = `${formatPercent(sample.gpuUtil)} GPU · ${formatPercent(sample.cpuUtil)} CPU`;
    $("#memoryValue").textContent = sample.vramUsedMib !== null
      ? `${formatBytes(sample.vramUsedMib * 1024 ** 2)} 显存`
      : `${formatPercent(sample.ramPercent)} 内存`;
    $("#diskValue").textContent = `${formatRate(sample.diskRead)} 读`;
  } else {
    live.className = state.telemetryError ? "live-indicator error" : "live-indicator idle";
    live.querySelector("span").textContent = state.telemetryError ? "异常" : state.telemetryEndpointAvailable === false ? "待机" : "连接中";
    $("#telemetrySource").textContent = state.telemetryError || (state.telemetryEndpointAvailable === false ? "等待工作台硬件采样或推理任务" : "正在连接硬件采样");
    $("#utilizationValue").textContent = "--";
    $("#memoryValue").textContent = "--";
    $("#diskValue").textContent = "--";
  }
  scheduleCharts();
}

function chartSeriesValues(key) {
  return state.telemetry.map((sample) => finite(sample[key]));
}

function drawLineChart(canvas, series, options = {}) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width);
  const height = Math.round(rect.height);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const padding = { top: 9, right: 8, bottom: 18, left: options.leftPadding || 30 };
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const plotHeight = Math.max(1, height - padding.top - padding.bottom);
  const allValues = series.flatMap((item) => item.values.filter((value) => value !== null));
  const maximum = options.maximum || Math.max(options.minimumMaximum || 1, ...allValues, 0);
  const axisMax = options.maximum || Math.max(options.minimumMaximum || 1, maximum * 1.12);

  context.strokeStyle = "#e3e7e4";
  context.fillStyle = "#8a948e";
  context.lineWidth = 1;
  context.font = "9px Consolas, monospace";
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let index = 0; index <= 2; index += 1) {
    const fraction = index / 2;
    const y = padding.top + plotHeight * fraction;
    context.beginPath();
    context.moveTo(padding.left, y + 0.5);
    context.lineTo(width - padding.right, y + 0.5);
    context.stroke();
    const labelValue = axisMax * (1 - fraction);
    const label = options.percent ? `${labelValue.toFixed(0)}` : options.formatAxis ? options.formatAxis(labelValue) : labelValue.toFixed(0);
    context.fillText(label, padding.left - 5, y);
  }

  const pointCount = Math.max(2, ...series.map((item) => item.values.length));
  for (const item of series) {
    context.strokeStyle = item.color;
    context.lineWidth = 1.7;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    let drawing = false;
    item.values.forEach((value, index) => {
      if (value === null) {
        drawing = false;
        return;
      }
      const x = padding.left + (pointCount <= 1 ? plotWidth : index / (pointCount - 1) * plotWidth);
      const y = padding.top + (1 - clamp(value / axisMax, 0, 1)) * plotHeight;
      if (!drawing) {
        context.moveTo(x, y);
        drawing = true;
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  }

  return axisMax;
}

function drawCharts() {
  chartFrame = null;
  const utilizationValues = [...chartSeriesValues("gpuUtil"), ...chartSeriesValues("cpuUtil")].filter((value) => value !== null);
  const memoryValues = [...chartSeriesValues("vramPercent"), ...chartSeriesValues("ramPercent")].filter((value) => value !== null);
  const diskValues = [...chartSeriesValues("diskRead"), ...chartSeriesValues("diskWrite")].filter((value) => value !== null);
  $("#utilizationEmpty").classList.toggle("hidden", utilizationValues.length > 0);
  $("#memoryEmpty").classList.toggle("hidden", memoryValues.length > 0);
  $("#diskEmpty").classList.toggle("hidden", diskValues.length > 0);
  drawLineChart($("#utilizationChart"), [
    { values: chartSeriesValues("gpuUtil"), color: "#16805c" },
    { values: chartSeriesValues("cpuUtil"), color: "#3178a6" },
  ], { maximum: 100, percent: true });
  drawLineChart($("#memoryChart"), [
    { values: chartSeriesValues("vramPercent"), color: "#8c6db0" },
    { values: chartSeriesValues("ramPercent"), color: "#c58631" },
  ], { maximum: 100, percent: true });
  const diskMaximum = drawLineChart($("#diskChart"), [
    { values: chartSeriesValues("diskRead"), color: "#3178a6" },
    { values: chartSeriesValues("diskWrite"), color: "#b65a50" },
  ], { minimumMaximum: 1024 ** 2, leftPadding: 46, formatAxis: (value) => formatBytes(value).replace(" ", "") });
  $("#diskScale").textContent = diskValues.length ? `峰值 ${formatRate(diskMaximum)}` : "自动量程";
}

function scheduleCharts() {
  if (chartFrame !== null) return;
  chartFrame = requestAnimationFrame(drawCharts);
}

async function pollTelemetry() {
  if (telemetryPollActive) return;
  if (telemetryLastReceivedAt && Date.now() - telemetryLastReceivedAt < 3500) return;
  telemetryPollActive = true;
  let sample = null;
  try {
    if (Date.now() >= state.telemetryRetryAt) {
      try {
        const payload = await api(TELEMETRY_ENDPOINT);
        state.telemetryEndpointAvailable = true;
        state.telemetryError = payload?.monitor_error || null;
        sample = normalizeTelemetry(payload, "硬件采样快照");
      } catch (error) {
        state.telemetryEndpointAvailable = false;
        state.telemetryError = `硬件快照不可用：${error.message}`;
        state.telemetryRetryAt = Date.now() + (error.status === 404 ? 30000 : 5000);
      }
    }
    addTelemetrySample(sample || latestJobTelemetry());
    if (!sample && !latestJobTelemetry()) renderTelemetry();
  } finally {
    telemetryPollActive = false;
  }
}

function connectTelemetryStream() {
  if (TELEMETRY_MODE === "snapshot") return;
  if (!("EventSource" in window)) return;
  if (telemetryEventSource) telemetryEventSource.close();
  telemetryEventSource = new EventSource(TELEMETRY_STREAM_ENDPOINT);
  telemetryEventSource.addEventListener("hardware", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.monitor_error) {
        state.telemetryEndpointAvailable = false;
        state.telemetryError = `硬件采样失败：${payload.monitor_error}`;
        renderTelemetry();
        return;
      }
      const sample = normalizeTelemetry(payload, "系统硬件流");
      if (!sample) return;
      telemetryLastReceivedAt = Date.now();
      state.telemetryEndpointAvailable = true;
      state.telemetryError = null;
      addTelemetrySample(sample);
    } catch (_) {
      state.telemetryEndpointAvailable = false;
    }
  });
  telemetryEventSource.addEventListener("open", () => {
    state.telemetryEndpointAvailable = true;
    state.telemetryError = null;
    renderTelemetry();
  });
  telemetryEventSource.addEventListener("error", () => {
    state.telemetryEndpointAvailable = false;
    state.telemetryError = "实时硬件连接中断，正在切换到快照采样";
    renderTelemetry();
  });
}

async function loadJobs({ announceErrors = false } = {}) {
  try {
    const jobs = await api("/api/jobs");
    state.jobs = Array.isArray(jobs) ? jobs : [];
    state.loaded.jobs = true;
    state.lastJobUpdate = new Date().toISOString();
    renderJobs();
  } catch (error) {
    if (announceErrors) showToast(`任务刷新失败：${error.message}`, true);
    throw error;
  }
}

async function refreshSystem() {
  if (systemPollActive) return;
  systemPollActive = true;
  try {
    state.system = await api("/api/system");
    state.loaded.system = true;
    renderSystem();
  } finally {
    systemPollActive = false;
  }
}

async function refreshAll({ quiet = false } = {}) {
  const button = $("#refreshButton");
  button.disabled = true;
  const requests = [
    ["system", "/api/system"],
    ["models", "/api/models"],
    ["presets", "/api/export-presets"],
    ["jobs", "/api/jobs"],
    ["profiles", "/api/profiles"],
  ];
  try {
    const results = await Promise.allSettled(requests.map(([, path]) => api(path)));
    const failures = [];
    results.forEach((result, index) => {
      const key = requests[index][0];
      if (result.status === "rejected") {
        failures.push(`${key}: ${result.reason.message}`);
        return;
      }
      state.loaded[key] = true;
      if (key === "system") state.system = result.value;
      if (key === "models") state.models = Array.isArray(result.value) ? result.value : [];
      if (key === "presets") {
        state.exportPresets = result.value?.presets || [];
        state.modelWorkspace = result.value || null;
      }
      if (key === "jobs") {
        state.jobs = Array.isArray(result.value) ? result.value : [];
        state.lastJobUpdate = new Date().toISOString();
      }
      if (key === "profiles") state.profiles = Array.isArray(result.value) ? result.value : [];
    });
    renderSystem();
    renderModelPage();
    renderProfiles();
    renderSuperResolution();
    renderJobs();
    setConnection(failures.length === 0, failures.length ? `${failures.length} 个接口不可用` : "服务正常");
    if (failures.length && !quiet) showToast(`部分数据刷新失败：${failures[0]}`, true);
  } finally {
    button.disabled = false;
  }
}

async function startPresetExport(presetId, button) {
  const preset = state.exportPresets.find((item) => item.id === presetId);
  if (!preset) return;
  button.disabled = true;
  try {
    const job = await api("/api/jobs/download-export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: presetId }),
    });
    state.selectedJobId = job.id;
    showToast(`${preset.label} 已进入${preset.product_type === "runtime_adapter" ? "下载与适配器构建" : "下载与切片"}队列`);
    await loadJobs();
    switchPage("tasks");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function startExport(index, button) {
  const model = state.models[index];
  if (!model) return;
  const selector = $(`.scope-select[data-model-index="${index}"]`);
  button.disabled = true;
  try {
    const job = await api("/api/jobs/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: model.id, video_blocks: selector?.value || "all" }),
    });
    state.selectedJobId = job.id;
    showToast(`${model.name} 已进入切片验证队列`);
    await loadJobs();
    switchPage("tasks");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function startInference(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  const prompt = $("#promptInput").value.trim();
  const tokens = parsedTokenIds();
  const frameCondition = renderFrameConditions();
  if (!tokens.valid) {
    showToast(tokens.message, true);
    return;
  }
  if (!prompt && tokens.values.length === 0) {
    showToast("请输入提示词或 Token IDs", true);
    return;
  }
  if (!frameCondition.ready) {
    showToast("请先上传当前模式需要的条件图片", true);
    return;
  }
  const button = $("#generateButton");
  button.disabled = true;
  try {
    const job = await api("/api/jobs/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt || null,
        token_ids: prompt ? null : tokens.values,
        steps: Number($("#stepsInput").value),
        use_acceleration_lora: $("#accelerationLoraInput").checked,
        seed: Number($("#seedInput").value),
        width: Number($("#widthInput").value),
        height: Number($("#heightInput").value),
        duration_seconds: Number($("#durationInput").value),
        temporal_mode: temporalMode(),
        conditioning_mode: frameCondition.mode,
        start_image_path: frameCondition.active.includes("start") ? state.frameImages.start.path : null,
        end_image_path: frameCondition.active.includes("end") ? state.frameImages.end.path : null,
        attention_query_chunk: Number($("#queryChunkSelect").value),
        l1_prefetch_shards: Number($("#l1PrefetchSelect").value),
      }),
    });
    state.selectedJobId = job.id;
    showToast("推理任务已加入队列");
    await loadJobs();
    switchPage("tasks");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    renderProfiles();
  }
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#menuButton").setAttribute("aria-expanded", "false");
}

function switchPage(page) {
  if (!["models", "inference", "super-resolution", "tasks"].includes(page)) page = "models";
  state.activePage = page;
  $$(".nav-item").forEach((item) => {
    const active = item.dataset.page === page;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  $$("[data-page-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.pagePanel === page));
  if (window.location.hash !== `#${page}`) window.history.replaceState(null, "", `#${page}`);
  closeSidebar();
  if (page === "tasks") scheduleCharts();
  $("#mainContent").focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "auto" });
}

function setJobFilter(filter) {
  state.jobFilter = filter;
  $$("[data-job-filter]").forEach((button) => button.classList.toggle("active", button.dataset.jobFilter === filter));
  renderJobList();
  renderJobInspector();
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
  $("#menuButton").addEventListener("click", () => {
    const open = $("#sidebar").classList.toggle("open");
    $("#menuButton").setAttribute("aria-expanded", String(open));
  });
  $("#sidebarScrim").addEventListener("click", closeSidebar);
  $("#refreshButton").addEventListener("click", () => refreshAll());
  $("#checkModelsButton").addEventListener("click", () => refreshAll());
  $("#refreshTasksButton").addEventListener("click", () => loadJobs({ announceErrors: true }));
  $("#inferenceForm").addEventListener("submit", startInference);
  $$("input[name=\"conditioningMode\"]").forEach((input) => input.addEventListener("change", renderProfiles));
  $("#startFrameFile").addEventListener("change", (event) => uploadFrameImage("start", event.target.files[0]));
  $("#endFrameFile").addEventListener("change", (event) => uploadFrameImage("end", event.target.files[0]));
  $("#startFrameClear").addEventListener("click", () => clearFrameImage("start"));
  $("#endFrameClear").addEventListener("click", () => clearFrameImage("end"));
  $("#superResolutionForm").addEventListener("submit", startSuperResolution);
  $("#superProbeButton").addEventListener("click", probeSuperVideo);
  $("#superSourcePath").addEventListener("change", () => {
    state.superVideo = null;
    renderSuperResolution();
  });
  $("#superVideoFile").addEventListener("change", (event) => uploadSuperVideo(event.target.files[0]));
  $("#superPromptInput").addEventListener("input", () => {
    state.superPromptEdited = true;
    renderSuperResolution();
  });
  ["#superScaleInput", "#superInterpolationInput", "#superStepsInput", "#superNoiseInput", "#superSeedInput", "#superQueryChunkSelect", "#superL1PrefetchSelect"]
    .forEach((selector) => $(selector).addEventListener("input", renderSuperResolution));
  $("#superAccelerationLoraInput").addEventListener("change", renderSuperResolution);
  $$('input[name="superProcessingMode"]').forEach((input) => input.addEventListener("change", renderSuperResolution));
  $("#resolutionPreset").addEventListener("change", updateResolutionPreset);
  ["#promptInput", "#tokenIdsInput", "#durationInput", "#stepsInput", "#seedInput", "#queryChunkSelect", "#l1PrefetchSelect"]
    .forEach((selector) => $(selector).addEventListener("input", renderProfiles));
  ["#widthInput", "#heightInput"].forEach((selector) => $(selector).addEventListener("input", renderProfiles));
  $$('input[name="temporalMode"]').forEach((input) => input.addEventListener("change", renderProfiles));
  $("#accelerationLoraInput").addEventListener("change", renderProfiles);
  $("#exportPresetList").addEventListener("click", (event) => {
    const button = event.target.closest(".preset-export-button");
    if (button) startPresetExport(button.dataset.preset, button);
  });
  $("#modelsBody").addEventListener("click", (event) => {
    const button = event.target.closest(".export-button");
    if (button) startExport(Number(button.dataset.modelIndex), button);
  });
  $("#jobsList").addEventListener("click", (event) => {
    const item = event.target.closest("[data-job-id]");
    if (!item) return;
    state.selectedJobId = item.dataset.jobId;
    renderJobList();
    renderJobInspector();
  });
  $$("[data-job-filter]").forEach((button) => button.addEventListener("click", () => setJobFilter(button.dataset.jobFilter)));
  window.addEventListener("resize", scheduleCharts, { passive: true });
  window.addEventListener("hashchange", () => switchPage(window.location.hash.slice(1)));
  window.addEventListener("beforeunload", () => {
    telemetryEventSource?.close();
    for (const item of Object.values(state.frameImages)) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    }
  });
}

async function initialize() {
  bindEvents();
  updateResolutionPreset();
  const initialPage = window.location.hash.slice(1);
  switchPage(["models", "inference", "super-resolution", "tasks"].includes(initialPage) ? initialPage : "models");
  await refreshAll({ quiet: true });
  connectTelemetryStream();
  pollTelemetry();
}

initialize();

setInterval(async () => {
  if (jobsPollActive) return;
  jobsPollActive = true;
  try {
    await loadJobs();
  } catch (_) {
    // The next poll retries transient service failures.
  } finally {
    jobsPollActive = false;
  }
}, JOB_POLL_INTERVAL_MS);

setInterval(pollTelemetry, TELEMETRY_POLL_INTERVAL_MS);
setInterval(updateJobTimers, 1000);
setInterval(async () => {
  try {
    await refreshSystem();
  } catch (_) {
    // System status is supplementary; the full refresh remains available.
  }
}, 15000);
