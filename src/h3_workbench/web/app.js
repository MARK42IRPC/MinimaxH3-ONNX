const state = { system: null, models: [], components: [], jobs: [], profiles: [] };
let toastTimer = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function formatBytes(value) {
  if (!Number.isFinite(value)) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index > 2 ? 2 : 1)} ${units[index]}`;
}

function formatElapsed(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function jobElapsed(job, now = Date.now()) {
  const started = Date.parse(job.started_at || job.created_at);
  if (!Number.isFinite(started)) return "--:--:--";
  const finished = Date.parse(job.finished_at || "");
  const end = Number.isFinite(finished) ? finished : now;
  return formatElapsed(end - started);
}

function performanceDetails(job) {
  const record = job.performance || {};
  const perf = record.performance || {};
  if (record.monitor_error) return [`监控异常 ${record.monitor_error}`];
  if (!Object.keys(perf).length) return [];
  const gpu = perf.gpu || {};
  const details = [];
  if (Number.isFinite(gpu.utilization_percent)) details.push(`GPU ${gpu.utilization_percent.toFixed(0)}%`);
  if (Number.isFinite(gpu.memory_used_mib)) {
    const totalMib = gpu.memory_used_mib + (gpu.memory_free_mib || 0);
    details.push(`显存 ${(gpu.memory_used_mib / 1024).toFixed(2)}/${(totalMib / 1024).toFixed(2)} GiB`);
  }
  if (Number.isFinite(perf.system_cpu_percent)) details.push(`CPU ${perf.system_cpu_percent.toFixed(0)}%`);
  if (Number.isFinite(perf.process_cpu_percent)) details.push(`进程 ${perf.process_cpu_percent.toFixed(0)}%`);
  if (Number.isFinite(perf.memory_available_bytes)) details.push(`可用内存 ${formatBytes(perf.memory_available_bytes)}`);
  if (Number.isFinite(perf.disk_read_bytes_per_second)) details.push(`磁盘读 ${formatBytes(perf.disk_read_bytes_per_second)}/s`);
  if (Number.isFinite(perf.process_read_bytes_per_second)) details.push(`进程读 ${formatBytes(perf.process_read_bytes_per_second)}/s`);
  if (Number.isFinite(gpu.power_watts)) details.push(`${gpu.power_watts.toFixed(0)} W`);
  return details;
}

function activityGroup(label, title, details, kind = "") {
  if (!title && details.length === 0) return "";
  return `<div class="activity-group ${kind}">
    <div class="activity-label">${label}</div>
    <div class="activity-content">
      ${title ? `<strong>${title}</strong>` : ""}
      ${details.map((detail) => `<span>${detail}</span>`).join("")}
    </div>
  </div>`;
}

function updateJobTimers() {
  const now = Date.now();
  $$("[data-job-elapsed]").forEach((element) => {
    const job = state.jobs.find((item) => item.id === element.dataset.jobElapsed);
    if (job) element.textContent = `耗时 ${jobElapsed(job, now)}`;
  });
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
  let latentFrames = 1;
  while (videoVaeOutputFrames(latentFrames) < outputFrames) latentFrames += 1;
  return latentFrames;
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 4200);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function componentLabel(component) {
  const labels = {
    audio_vae: "Audio VAE",
    video_vae: "Video VAE",
    text_encoder: "Text Encoder",
    fl2va_transformer: "FL2VA",
    ref2va_transformer: "Ref2VA",
    acceleration_lora: "加速 LoRA",
    unknown: "Unknown",
  };
  return labels[component] || component;
}

function componentStatusLabel(status) {
  return { ready: "完整", incomplete: "不完整", missing: "未安装", invalid: "清单损坏", unvalidated: "未验证" }[status] || status;
}

function renderDownloadComponents() {
  const list = $("#downloadList");
  if (!list) return;
  list.innerHTML = state.components.map((item) => {
    const checked = item.status !== "ready" ? "checked" : "";
    const disabled = item.status === "ready" ? "disabled" : "";
    const size = item.size_bytes ? formatBytes(item.size_bytes) : "待下载";
    const missing = item.missing_files?.length ? `缺少 ${item.missing_files.length} 个文件` : "文件清单正常";
    return `<label class="download-item ${item.status}"><input type="checkbox" value="${item.id}" ${checked} ${disabled}><span class="download-item-main"><strong>${item.label}</strong><small>${item.role} · ${size} · ${missing}</small></span><span class="download-status">${componentStatusLabel(item.status)}</span></label>`;
  }).join("");
  const incomplete = state.components.filter((item) => item.status !== "ready").length;
  $("#downloadSummary").textContent = `${incomplete} 个组件需要处理 / ${state.components.length} 个已登记`;
}

function renderSystem() {
  const system = state.system;
  if (!system) return;
  $("#workspacePath").textContent = system.workspace;
  $("#gpuName").textContent = system.gpu.name || "CPU only";
  $("#gpuMemory").textContent = system.gpu.memory_bytes ? formatBytes(system.gpu.memory_bytes) : "--";
  $("#ramAvailable").textContent = formatBytes(system.memory_available_bytes);
  $("#providerName").textContent = system.providers.includes("CUDAExecutionProvider") ? "ONNX CUDA" : "ONNX CPU";
  $("#healthDot").classList.add("online");
  $("#healthText").textContent = "服务正常";

  const rows = [
    ["操作系统", system.platform],
    ["Python", system.python],
    ["PyTorch", `${system.torch} / CUDA ${system.gpu.cuda || "N/A"}`],
    ["ONNX Runtime", system.onnxruntime],
    ["Execution Providers", system.providers.join(", ")],
    ["系统内存", `${formatBytes(system.memory_available_bytes)} 可用 / ${formatBytes(system.memory_total_bytes)}`],
    ["分页文件", `${formatBytes(system.pagefile_used_bytes)} 已用 / ${formatBytes(system.pagefile_total_bytes)}`],
    ["L2 分片缓存", `${formatBytes(system.l2_cache_bytes)} / 向前预取 ${system.prefetch_shards} 个图`],
    ["GPU", system.gpu.name || "不可用"],
    ["工作区", system.workspace],
  ];
  $("#systemGrid").innerHTML = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
}

function renderModels() {
  const body = $("#modelsBody");
  $("#modelCount").textContent = `${state.models.length} 个组件`;
  $("#modelsEmpty").classList.toggle("hidden", state.models.length > 0);
  body.innerHTML = state.models.map((model, index) => {
    const supported = model.export_supported;
    let scope = "Encoder + Decoder";
    if (model.component === "video_vae") {
      scope = `<select id="scope-${index}" aria-label="Video VAE 导出范围"><option value="0">Block 0 冒烟验证</option><option value="all">全部 36 Blocks</option></select>`;
    } else if (["fl2va_transformer", "ref2va_transformer"].includes(model.component)) {
      scope = `<select id="scope-${index}" aria-label="主模型导出范围"><option value="0">Block 0 冒烟验证</option><option value="all">全部 50 Blocks</option></select>`;
    } else if (model.component === "text_encoder") {
      scope = `<select id="scope-${index}" aria-label="文本编码器导出范围"><option value="0">Layer 0 冒烟验证</option><option value="all">全部 50 Layers</option></select>`;
    }
    return `<tr>
      <td><span class="model-name" title="${model.name}">${model.name}</span><span class="model-path">${model.id}</span></td>
      <td><span class="tag ${supported ? "" : "unsupported"}">${componentLabel(model.component)}</span></td>
      <td>${model.dtype}</td>
      <td>${formatBytes(model.size_bytes)}</td>
      <td>${model.tensor_count}</td>
      <td>${scope}</td>
      <td><button class="command-button export-button" data-index="${index}" ${supported ? "" : "disabled"}>导出并验证</button></td>
    </tr>`;
  }).join("");

  $$(".export-button").forEach((button) => button.addEventListener("click", () => startExport(Number(button.dataset.index))));
}

function renderJobs() {
  $("#jobCount").textContent = `${state.jobs.length} 个任务`;
  $("#jobsEmpty").classList.toggle("hidden", state.jobs.length > 0);
  $("#jobsList").classList.toggle("hidden", state.jobs.length === 0);
  $("#jobsList").innerHTML = state.jobs.map((job) => {
    const activity = job.activity || {};
    const prefetch = job.prefetch || {};
    const performance = performanceDetails(job);
    const positionDetails = [];
    const executionDetails = [];
    const cacheDetails = [];
    if (activity.sampling_step) positionDetails.push(`步数 ${activity.sampling_step}/${activity.sampling_steps}`);
    if (activity.segment) positionDetails.push(`片段 ${activity.segment}/${activity.segments}`);
    if (activity.current !== undefined && activity.total !== undefined) {
      const unit = activity.module === "Qwen" ? "Layer" : activity.module === "Video VAE" ? "Block" : "Block";
      positionDetails.push(`${unit} ${activity.current}/${activity.total}`);
    }
    if (activity.shard) positionDetails.push(`分片 ${activity.shard}/${activity.shards}`);
    if (activity.tile) positionDetails.push(`Tile ${activity.tile}/${activity.tiles}`);
    if (activity.provider) positionDetails.push(activity.provider.replace("ExecutionProvider", ""));
    if (activity.elapsed_seconds !== undefined) positionDetails.push(`本片 ${Number(activity.elapsed_seconds).toFixed(2)}s`);
    if (activity.qkv_chunk_tokens) executionDetails.push(`QKV ${activity.qkv_chunk_tokens}`);
    if (activity.qkv_buffer_dtype) executionDetails.push(`QKV Buffer ${activity.qkv_buffer_dtype.toUpperCase()}`);
    if (activity.attention_output_chunk_tokens) executionDetails.push(`Attention Out ${activity.attention_output_chunk_tokens}`);
    if (activity.mlp_chunk_tokens) executionDetails.push(`MLP ${activity.mlp_chunk_tokens}`);
    if (activity.chunk_io_binding) executionDetails.push("I/O Binding");
    if (activity.attention_buffer_dtype) executionDetails.push(`Attended ${activity.attention_buffer_dtype.toUpperCase()}`);
    if (activity.l1_sessions) cacheDetails.push(`L1 ${activity.l1_sessions} 分片 / ${formatBytes(activity.l1_weight_bytes)}`);
    if (activity.l2_budget_bytes) {
      cacheDetails.push(`L2 ${formatBytes(activity.l2_staged_bytes)} / ${formatBytes(activity.l2_budget_bytes)}`);
      cacheDetails.push(`就绪 ${activity.l2_ready}/${activity.l2_entries}`);
      cacheDetails.push(`命中 ${activity.l2_hits} / 等待 ${activity.l2_waits}`);
    }
    if (activity.vram_free_bytes) cacheDetails.push(`规划显存 ${formatBytes(activity.vram_free_bytes)}`);
    if (activity.batch_size) cacheDetails.push(`加载批次 ${activity.batch_index || 1}/${activity.batch_size}`);
    if (activity.host_prefetch_budget_bytes !== undefined && activity.host_prefetch_budget_bytes !== null) {
      cacheDetails.push(`提交余量 ${formatBytes(activity.host_prefetch_budget_bytes)}`);
    }
    const prefetchDetails = [];
    if (prefetch.prefetch_layer) prefetchDetails.push(`Layer ${prefetch.prefetch_layer}/${prefetch.prefetch_total}`);
    if (prefetch.shard) prefetchDetails.push(`分片 ${prefetch.shard}/${prefetch.shards}`);
    if (prefetch.prefetch_ahead !== undefined) prefetchDetails.push(`向前 ${prefetch.prefetch_ahead} 片`);
    if (prefetch.l1_prefetch_hits !== undefined) {
      prefetchDetails.push(`L1 命中 ${prefetch.l1_prefetch_hits} / 等待 ${prefetch.l1_prefetch_waits}`);
    }
    if (prefetch.wait_seconds !== undefined) prefetchDetails.push(`等待 ${Number(prefetch.wait_seconds).toFixed(2)}s`);
    const cacheTitle = prefetch.operation ? `预取: ${prefetch.operation}` : cacheDetails.length ? "缓存状态" : "";
    return `<article class="job-row">
    <div><div class="job-title" title="${job.model_id}">${job.model_id}</div><div class="job-message">${job.message}</div><div class="job-elapsed" data-job-elapsed="${job.id}">耗时 ${jobElapsed(job)}</div></div>
    <span class="status ${job.status}">${job.status}</span>
    <div class="progress-track" aria-label="${Math.round(job.progress * 100)}%"><div class="progress-bar" style="width:${job.progress * 100}%"></div></div>
    <div class="job-actions">
      <strong class="job-progress-value">${Math.round(job.progress * 100)}%</strong>
      <div class="job-links">
        ${job.kind === "inference" && job.status === "completed" ? `<a href="/api/jobs/${job.id}/output">下载 MP4</a>` : ""}
        ${job.kind === "inference" && job.result?.metadata ? `<a href="/api/jobs/${job.id}/metadata">元数据</a>` : ""}
        ${job.kind === "inference" && job.performance_log ? `<a href="/api/jobs/${job.id}/performance">性能日志</a>` : ""}
      </div>
    </div>
    <div class="job-activity">
      ${activityGroup("当前计算", `${activity.module || "Queue"} · ${activity.operation || "Waiting"}`, positionDetails, "compute")}
      ${activityGroup("执行参数", "", executionDetails, "execution")}
      ${activityGroup("缓存与预取", cacheTitle, [...cacheDetails, ...prefetchDetails], "cache")}
      ${activityGroup("实时性能", "", performance, "performance")}
    </div>
  </article>`;
  }).join("");
  updateJobTimers();
}

function renderProfiles() {
  const profile = state.profiles[0];
  if (!profile) return;
  $("#inferenceProvider").textContent = profile.cuda_provider_available ? "CUDA" : "CPU";
  const width = Math.max(128, Math.min(1024, Number($("#widthInput").value) || profile.output_width));
  const height = Math.max(128, Math.min(1024, Number($("#heightInput").value) || profile.output_height));
  const paddedWidth = Math.ceil(width / 32) * 32;
  const paddedHeight = Math.ceil(height / 32) * 32;
  const duration = Math.max(0.1, Math.min(15, Number($("#durationInput").value) || profile.frames / profile.fps));
  const targetFrames = Math.max(1, Math.round(duration * profile.fps));
  const temporalMode = $("#temporalMode").value;
  const segments = temporalMode === "segmented" ? Math.ceil(targetFrames / profile.frames) : 1;
  const steps = Math.max(1, Math.min(50, Number($("#stepsInput").value) || 4));
  const accelerationActive = profile.acceleration_ready && steps >= 4 && steps <= 8;
  const latentFrames = temporalMode === "native" ? videoLatentFramesForOutput(targetFrames) : profile.video_latent_frames;
  const audioTokens = temporalMode === "native" ? Math.ceil(targetFrames * 40 / profile.fps) * 2 : profile.audio_tokens;
  const videoTokens = latentFrames * (paddedHeight / 32) * (paddedWidth / 32);
  const sequenceTokens = profile.text_tokens + audioTokens + videoTokens;
  const qkvCpuBytes = sequenceTokens * 56 * 384 * 2;
  const kvGpuBytes = sequenceTokens * 56 * 128 * 2 * 2;
  const queryChunk = Number($("#queryChunkSelect").value) || 128;
  const rows = [
    ["输出", `${width}×${height} / ${targetFrames} 帧 / ${(targetFrames / profile.fps).toFixed(2)}s`],
    ["主模型", accelerationActive ? "Turbo v4 加速版" : "基础版"],
    ["时序计划", temporalMode === "native" ? `${latentFrames} latent / 整体` : `${segments} 段 / ${segments * steps} 次去噪`],
    ["注意力序列", `${sequenceTokens} tokens / query ${queryChunk}`],
    ["CPU QKV 缓存 (FP16)", formatBytes(qkvCpuBytes)],
    ["GPU K/V 缓存", formatBytes(kvGpuBytes)],
  ];
  $("#profileGrid").innerHTML = rows.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
  const warning = $("#runtimeWarning");
  const notices = [];
  if (!profile.cuda_provider_available) notices.push("当前 ONNX Runtime 没有 CUDAExecutionProvider，将使用 CPU 执行。");
  if (!profile.acceleration_ready && steps >= 4 && steps <= 8) notices.push("加速 LoRA 主模型变体尚未完成，将使用基础模型。");
  if (!profile.main_ready && profile.acceleration_ready && (steps < 4 || steps > 8)) notices.push("当前仅安装 Turbo v4 主模型，请使用 4-8 步；其他步数需要基础主模型。");
  if (!profile.tokenizer_ready) notices.push("Tokenizer 尚未落盘，当前仅接受 Token IDs。");
  if (temporalMode === "segmented" && segments > 1) notices.push("分段模式会独立生成 17 帧片段；当前尚未接入首帧续接，段间画面可能跳变。");
  if (temporalMode === "native" && targetFrames > profile.frames) notices.push("整体模式保持完整时间序列；长时间 Video VAE 将使用 GPU 时间窗口解码。");
  if (steps < 5 && !accelerationActive) notices.push("基础主模型低于 5 步时音频支路可能数值失稳；异常时任务会保留视频并使用静音轨。");
  warning.classList.toggle("hidden", notices.length === 0);
  warning.textContent = notices.join(" ");
  $("#generateButton").disabled = !profile.generation_ready;
}

async function startInference(event) {
  event.preventDefault();
  const prompt = $("#promptInput").value.trim();
  const tokenIds = $("#tokenIdsInput").value.split(",").map((value) => Number(value.trim())).filter(Number.isInteger);
  const profile = state.profiles[0];
  try {
    if (!prompt && tokenIds.length === 0) throw new Error("请输入 Prompt 或 Token IDs");
    if (prompt && profile && !profile.tokenizer_ready) throw new Error("H3 Tokenizer 文件不完整");
    await api("/api/jobs/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt || null,
        token_ids: prompt ? null : tokenIds,
        steps: Number($("#stepsInput").value),
        seed: Number($("#seedInput").value),
        width: Number($("#widthInput").value),
        height: Number($("#heightInput").value),
        duration_seconds: Number($("#durationInput").value),
        temporal_mode: $("#temporalMode").value,
        attention_query_chunk: Number($("#queryChunkSelect").value),
        l1_prefetch_shards: Number($("#l1PrefetchSelect").value),
      }),
    });
    showToast("推理任务已加入队列");
    switchTab("jobs");
    await loadJobs();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function startExport(index) {
  const model = state.models[index];
  const selector = $(`#scope-${index}`);
  const videoBlocks = selector ? selector.value : "all";
  try {
    await api("/api/jobs/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: model.id, video_blocks: videoBlocks }),
    });
    showToast(`${model.name} 已加入导出队列`);
    switchTab("jobs");
    await loadJobs();
  } catch (error) {
    showToast(error.message, true);
  }
}

function switchTab(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.remove("active"));
  $(`#${name}Panel`).classList.add("active");
  if (window.location.hash !== `#${name}`) window.history.replaceState(null, "", `#${name}`);
}

async function loadJobs() {
  state.jobs = await api("/api/jobs");
  renderJobs();
}

async function refresh() {
  $("#refreshButton").disabled = true;
  try {
    const [system, models, components, jobs, profiles] = await Promise.all([api("/api/system"), api("/api/models"), api("/api/model-components"), api("/api/jobs"), api("/api/profiles")]);
    state.system = system;
    state.models = models;
    state.components = components.components || [];
    state.jobs = jobs;
    state.profiles = profiles;
    renderSystem();
    renderModels();
    renderDownloadComponents();
    renderJobs();
    renderProfiles();
  } catch (error) {
    $("#healthText").textContent = "连接失败";
    showToast(error.message, true);
  } finally {
    $("#refreshButton").disabled = false;
  }
}

async function downloadSelectedModels() {
  const ids = $$("#downloadList input[type=checkbox]:checked").map((input) => input.value);
  if (!ids.length) { showToast("请选择至少一个未安装组件", true); return; }
  const button = $("#downloadModelsButton");
  button.disabled = true;
  try {
    await api("/api/jobs/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ components: ids }) });
    showToast(`已加入 ${ids.length} 个组件的下载队列`);
    switchTab("jobs");
    await loadJobs();
  } catch (error) { showToast(error.message, true); } finally { button.disabled = false; }
}

$$('.nav-item').forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
$("#refreshButton").addEventListener("click", refresh);
$("#checkModelsButton").addEventListener("click", refresh);
$("#downloadModelsButton").addEventListener("click", downloadSelectedModels);
$("#inferenceForm").addEventListener("submit", startInference);
$("#widthInput").addEventListener("input", renderProfiles);
$("#heightInput").addEventListener("input", renderProfiles);
$("#durationInput").addEventListener("input", renderProfiles);
$("#stepsInput").addEventListener("input", renderProfiles);
$("#temporalMode").addEventListener("change", renderProfiles);
$("#queryChunkSelect").addEventListener("change", renderProfiles);
const initialTab = window.location.hash.slice(1);
if (["models", "inference", "jobs", "system"].includes(initialTab)) switchTab(initialTab);
refresh();
let jobsPollActive = false;
setInterval(async () => {
  if (jobsPollActive) return;
  jobsPollActive = true;
  try {
    await loadJobs();
  } catch (_) {
    // The next poll recovers transient API failures and stale terminal states.
  } finally {
    jobsPollActive = false;
  }
}, 2000);
setInterval(updateJobTimers, 1000);
