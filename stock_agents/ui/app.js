"use strict";

const state = {
  ohlcvPayload: null,
  candles: [],
  trainingResult: null,
  autonomousResult: null,
  financialDocumentFile: null,
};

const colors = {
  blue: "#4a8cff",
  cyan: "#28d7c0",
  gold: "#f4b860",
  violet: "#aa83ff",
  grid: "rgba(175, 207, 222, 0.12)",
  muted: "#78919c",
  text: "#dcecef",
};

const byId = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  bindUpload();
  bindDocuments();
  bindTraining();
  bindPrediction();
  bindReview();
  bindUtilityActions();
});

async function checkHealth() {
  try {
    await api("/health");
    byId("health-dot").className = "status-dot ok";
    byId("health-text").textContent = "API online";
  } catch (error) {
    byId("health-dot").className = "status-dot error";
    byId("health-text").textContent = "API unavailable";
  }
}

function bindDocuments() {
  const input = byId("financial-document-file");
  const zone = byId("document-drop-zone");
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) {
      selectFinancialDocument(input.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
      selectFinancialDocument(file);
    }
  });

  byId("document-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const status = byId("document-index-status");
    const button = byId("document-index-button");
    if (!state.financialDocumentFile) {
      setStatus(status, "Choose a financial document first.", "error");
      return;
    }
    setBusy(button, true, "Indexing...");
    setStatus(status, "Extracting text, chunking, and indexing the document...");
    try {
      const publishedDate = byId("document-date").value;
      const body = {
        stock: documentStock(),
        filename: state.financialDocumentFile.name,
        document_type: byId("document-type").value,
        title: optionalText("document-title"),
        published_at: publishedDate ? `${publishedDate}T00:00:00+05:30` : null,
        content_base64: await fileToBase64(state.financialDocumentFile),
      };
      const result = await api("/api/v1/rag/documents/ingest", body);
      setStatus(
        status,
        result.status === "duplicate"
          ? "This document was already indexed for the stock."
          : `Indexed ${result.document.chunk_count} document chunks.`,
        "success",
      );
      await loadDocumentList();
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Index document");
    }
  });

  byId("refresh-documents-button").addEventListener("click", loadDocumentList);
  byId("rag-query-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = byId("rag-query-button");
    const status = byId("rag-query-status");
    setBusy(button, true, "Retrieving...");
    setStatus(status, "Searching indexed financial evidence...");
    try {
      const result = await api("/api/v1/rag/documents/query", {
        stock: documentStock(),
        query: byId("rag-query").value.trim(),
        top_k: 6,
      });
      renderRagResult(result);
      setStatus(status, "Retrieval complete.", "success");
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Retrieve evidence");
    }
  });

  byId("predict-symbol").addEventListener("change", syncDocumentStock);
  byId("predict-name").addEventListener("change", syncDocumentStock);
}

function selectFinancialDocument(file) {
  const status = byId("document-file-status");
  if (file.size > 25 * 1024 * 1024) {
    state.financialDocumentFile = null;
    setStatus(status, "Documents larger than 25 MB are not accepted.", "error");
    return;
  }
  state.financialDocumentFile = file;
  byId("document-drop-zone").querySelector("strong").textContent = file.name;
  if (!byId("document-title").value) {
    byId("document-title").value = file.name.replace(/\.[^.]+$/, "");
  }
  setStatus(status, `${file.name} selected (${formatFileSize(file.size)}).`, "success");
}

async function loadDocumentList() {
  const container = byId("document-list");
  container.innerHTML = "<div class=\"mini-empty\">Loading indexed documents...</div>";
  try {
    const result = await api("/api/v1/rag/documents/list", { stock: documentStock() });
    renderDocumentList(result.documents || []);
  } catch (error) {
    container.innerHTML = "";
    const message = document.createElement("div");
    message.className = "mini-empty";
    message.textContent = error.message;
    container.append(message);
  }
}

function renderDocumentList(documents) {
  const container = byId("document-list");
  container.innerHTML = "";
  if (!documents.length) {
    container.innerHTML = "<div class=\"mini-empty\">No documents indexed for this stock.</div>";
    return;
  }
  documents.forEach((documentRow) => {
    const row = document.createElement("article");
    row.className = "document-row";
    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = documentRow.title || documentRow.filename;
    const metadata = document.createElement("span");
    metadata.textContent = `${String(documentRow.document_type || "other").replaceAll("_", " ")} | ${formatDate(documentRow.published_at || documentRow.ingested_at)}`;
    const filename = document.createElement("small");
    filename.textContent = documentRow.filename || "";
    details.append(title, metadata, filename);
    const count = document.createElement("span");
    count.className = "document-count";
    count.textContent = `${documentRow.chunk_count || 0} chunks`;
    row.append(details, count);
    container.append(row);
  });
}

function renderRagResult(result) {
  byId("rag-summary").hidden = false;
  byId("rag-document-count").textContent = formatInteger(result.document_count);
  byId("rag-chunk-count").textContent = formatInteger(result.retrieved_count);
  byId("rag-signal-score").textContent = result.signal_score == null
    ? "-"
    : signedNumber(result.signal_score, 3);
  byId("rag-confidence").textContent = percentFromRatio(result.confidence);
  byId("rag-summary-text").textContent = result.summary || "";

  const container = byId("rag-citations");
  container.innerHTML = "";
  (result.citations || []).forEach((citation) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    const title = document.createElement("strong");
    title.textContent = citation.title || "Untitled document";
    const excerpt = document.createElement("p");
    excerpt.textContent = citation.excerpt || "";
    const metadata = document.createElement("span");
    metadata.textContent = `${citation.document_type || "other"} | relevance ${formatNumber(citation.relevance_score, 3)} | ${citation.chunk_id}`;
    card.append(title, excerpt, metadata);
    container.append(card);
  });
  if (!(result.citations || []).length) {
    container.innerHTML = "<div class=\"mini-empty\">No matching evidence found.</div>";
  }
}

function documentStock() {
  return {
    symbol: byId("document-symbol").value.trim(),
    name: byId("document-company").value.trim(),
  };
}

function syncDocumentStock() {
  byId("document-symbol").value = byId("predict-symbol").value;
  byId("document-company").value = byId("predict-name").value;
}

function bindUpload() {
  const input = byId("ohlcv-file");
  const zone = byId("drop-zone");

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) {
      loadOhlcvFile(input.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    });
  });

  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
      loadOhlcvFile(file);
    }
  });
}

async function loadOhlcvFile(file) {
  const status = byId("file-status");
  status.className = "inline-status";
  status.textContent = `Reading ${file.name}...`;

  if (file.size > 25 * 1024 * 1024) {
    setStatus(status, "Files larger than 25 MB are not accepted by this UI.", "error");
    return;
  }

  try {
    const payload = JSON.parse(await file.text());
    const candles = extractCandles(payload);
    validateCandles(candles);
    candles.sort((left, right) => new Date(left[0]) - new Date(right[0]));
    state.ohlcvPayload = payload;
    state.candles = candles;
    renderDataset(file.name, candles);
    setStatus(status, `${file.name} loaded and validated.`, "success");
  } catch (error) {
    state.ohlcvPayload = null;
    state.candles = [];
    resetDataset();
    setStatus(status, error.message || "Could not parse the JSON file.", "error");
  }
}

function extractCandles(payload) {
  if (payload && !Array.isArray(payload) && Array.isArray(payload.candles)) {
    return payload.candles;
  }
  if (payload && !Array.isArray(payload) && Array.isArray(payload.data)) {
    for (const item of payload.data) {
      if (item && item.data && Array.isArray(item.data.candles) && item.data.candles.length) {
        return item.data.candles;
      }
    }
  }
  if (Array.isArray(payload) && payload.length && Array.isArray(payload[0])) {
    return payload;
  }
  if (
    Array.isArray(payload)
    && payload.length
    && payload[0]
    && Array.isArray(payload[0].candles)
  ) {
    return payload[0].candles;
  }
  throw new Error("Unsupported JSON format. Expected an OHLCV candle array or Upstox-style payload.");
}

function validateCandles(candles) {
  if (!candles.length) {
    throw new Error("The file does not contain any candles.");
  }
  candles.forEach((candle, index) => {
    if (!Array.isArray(candle) || candle.length < 6) {
      throw new Error(`Candle ${index + 1} must contain timestamp, OHLC, and volume.`);
    }
    if (Number.isNaN(new Date(candle[0]).getTime())) {
      throw new Error(`Candle ${index + 1} has an invalid timestamp.`);
    }
    for (let field = 1; field <= 5; field += 1) {
      if (!Number.isFinite(Number(candle[field]))) {
        throw new Error(`Candle ${index + 1} contains a non-numeric OHLCV value.`);
      }
    }
  });
}

function renderDataset(fileName, candles) {
  const first = candles[0];
  const last = candles[candles.length - 1];
  byId("dataset-state").textContent = "Ready";
  byId("dataset-state").className = "pill";
  byId("dataset-count").textContent = formatInteger(candles.length);
  byId("dataset-first").textContent = formatDate(first[0]);
  byId("dataset-last").textContent = formatDate(last[0]);
  byId("dataset-close").textContent = formatNumber(last[4]);
  byId("drop-zone").querySelector("strong").textContent = fileName;
  byId("predict-reference").placeholder = formatNumber(last[4]);
}

function resetDataset() {
  byId("dataset-state").textContent = "Waiting";
  byId("dataset-state").className = "pill muted";
  ["dataset-count", "dataset-first", "dataset-last", "dataset-close"].forEach((id) => {
    byId(id).textContent = "-";
  });
}

function bindTraining() {
  byId("training-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!requireDataset(byId("training-status"))) {
      return;
    }

    const button = byId("train-button");
    const status = byId("training-status");
    setBusy(button, true, "Training...");
    setStatus(status, "TensorFlow training is running. Keep this page open.");

    const body = {
      ohlcv_data: state.ohlcvPayload,
      model_dir: byId("train-model-dir").value.trim(),
      stock_name: byId("train-stock").value.trim(),
      window_size: numericValue("train-window"),
      horizon: numericValue("train-horizon"),
      target_field: byId("train-target").value,
      epochs: numericValue("train-epochs"),
      batch_size: numericValue("train-batch"),
      train_ratio: numericValue("train-ratio"),
      validation_split: 0.1,
    };

    try {
      const result = await api("/api/v1/model/train", body);
      state.trainingResult = result;
      renderTrainingResults(result);
      byId("predict-model-dir").value = body.model_dir;
      setStatus(status, `Training complete. Model saved to ${body.model_dir}.`, "success");
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Start training");
    }
  });

  byId("load-results-button").addEventListener("click", async () => {
    const status = byId("training-status");
    const button = byId("load-results-button");
    const modelDir = byId("train-model-dir").value.trim();
    if (!modelDir) {
      setStatus(status, "Enter a model directory first.", "error");
      return;
    }
    setBusy(button, true, "Loading...");
    setStatus(status, "Loading saved model artifacts...");
    try {
      const result = await api("/api/v1/model/results", { model_dir: modelDir });
      state.trainingResult = result;
      renderTrainingResults(result);
      byId("predict-model-dir").value = modelDir;
      setStatus(status, "Saved training results loaded.", "success");
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Load saved results");
    }
  });

  byId("test-sample-select").addEventListener("change", () => {
    renderTestSample(Number(byId("test-sample-select").value));
  });

  byId("train-model-dir").addEventListener("change", () => {
    byId("predict-model-dir").value = byId("train-model-dir").value;
  });
}

function renderTrainingResults(result) {
  byId("training-empty").hidden = true;
  byId("training-results").hidden = false;

  const metrics = result.metrics || {};
  const metadata = result.metadata || {};
  byId("metric-mae").textContent = formatNumber(metrics.mae, 3);
  byId("metric-rmse").textContent = formatNumber(metrics.rmse, 3);
  byId("metric-mape").textContent = metrics.mape == null ? "-" : `${formatNumber(metrics.mape, 2)}%`;
  byId("metric-samples").textContent = formatInteger(result.test_samples ?? metadata.test_samples);

  requestAnimationFrame(() => {
    drawLineChart(
      byId("history-chart"),
      [
        { name: "Train", values: result.history?.loss || [], color: colors.blue },
        { name: "Validation", values: result.history?.val_loss || [], color: colors.gold },
      ],
      (result.history?.loss || []).map((_, index) => String(index + 1)),
    );
  });

  const predictions = result.test_predictions || [];
  const select = byId("test-sample-select");
  select.innerHTML = "";
  predictions.forEach((row, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Window ${index + 1} - after ${row.context_end_date}`;
    select.append(option);
  });

  if (predictions.length) {
    select.value = String(predictions.length - 1);
    renderTestSample(predictions.length - 1);
  } else {
    byId("comparison-table").innerHTML = "<tr><td colspan=\"5\">No test predictions found.</td></tr>";
    clearCanvas(byId("comparison-chart"));
  }
}

function renderTestSample(index) {
  const predictions = state.trainingResult?.test_predictions || [];
  const row = predictions[index];
  if (!row) {
    return;
  }

  requestAnimationFrame(() => {
    drawLineChart(
      byId("comparison-chart"),
      [
        { name: "Actual", values: row.actual, color: colors.gold },
        { name: "Predicted", values: row.predicted, color: colors.blue },
      ],
      row.forecast_dates,
    );
  });

  const tbody = byId("comparison-table");
  tbody.innerHTML = "";
  row.forecast_dates.forEach((date, itemIndex) => {
    const actual = Number(row.actual[itemIndex]);
    const predicted = Number(row.predicted[itemIndex]);
    const error = predicted - actual;
    const errorPct = actual === 0 ? null : (error / Math.abs(actual)) * 100;
    const tr = document.createElement("tr");
    [
      formatDate(date),
      formatNumber(actual),
      formatNumber(predicted),
      signedNumber(error),
      errorPct == null ? "-" : `${signedNumber(errorPct)}%`,
    ].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    });
    tbody.append(tr);
  });
}

function bindPrediction() {
  byId("prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!requireDataset(byId("prediction-status"))) {
      return;
    }

    const button = byId("predict-button");
    const status = byId("prediction-status");
    setBusy(button, true, "Agents working...");
    setStatus(status, "Running memory, technical, context, fundamental, document RAG, and decision agents...");

    const referenceInput = byId("predict-reference").value;
    const body = {
      stock: {
        symbol: byId("predict-symbol").value.trim(),
        name: byId("predict-name").value.trim(),
        aliases: commaList("predict-aliases"),
        sector: optionalText("predict-sector"),
        sector_keywords: commaList("predict-keywords"),
        fundamental_symbol: optionalText("predict-fundamental-symbol"),
        nse_symbol: optionalText("predict-nse"),
      },
      ohlcv_data: state.ohlcvPayload,
      model_dir: byId("predict-model-dir").value.trim(),
      reference_price: referenceInput ? Number(referenceInput) : null,
      include_fundamentals: byId("include-fundamentals").checked,
      include_financial_documents: byId("include-documents").checked,
      rag_query: optionalText("predict-rag-query"),
      rag_top_k: 5,
      auto_store_prediction: byId("auto-store").checked,
      options: {
        hours_before_previous_close: 2,
      },
    };

    try {
      const result = await api("/api/v1/predictions/autonomous", body);
      state.autonomousResult = result;
      renderAutonomousResult(result);
      setStatus(status, "Autonomous prediction complete.", "success");
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Run autonomous prediction");
    }
  });
}

function renderAutonomousResult(result) {
  byId("prediction-empty").hidden = true;
  byId("prediction-results").hidden = false;

  const decision = result.decision || {};
  byId("decision-direction").textContent = decision.predicted_direction || "-";
  byId("decision-direction").style.color = directionColor(decision.predicted_direction);
  byId("decision-change").textContent = percentValue(decision.predicted_change_pct);
  byId("decision-confidence").textContent = percentFromRatio(decision.confidence);
  byId("decision-agreement").textContent = percentFromRatio(decision.agreement);
  byId("coverage-pill").textContent = `${percentFromRatio(decision.signal_coverage)} coverage`;

  renderAgentCards(result.agent_outputs || {});
  renderPredictionCitations(result.agent_outputs?.financial_documents?.payload?.citations || []);
  renderContributions(decision.contributions || {});
  renderRationale(decision.rationale || []);
  renderForecast(result.agent_outputs?.technical?.payload || {});

  const stored = result.autonomous_actions?.store_prediction || {};
  const predictionId = stored.prediction_id || "Not stored";
  byId("prediction-id").textContent = predictionId;
  if (stored.prediction_id) {
    byId("review-prediction-id").value = stored.prediction_id;
    const technicalReference = result.agent_outputs?.technical?.payload?.reference_value;
    if (technicalReference) {
      byId("review-previous-close").value = Number(technicalReference).toFixed(2);
    }
  }
}

function renderPredictionCitations(citations) {
  const section = byId("prediction-document-evidence");
  const container = byId("prediction-citation-list");
  container.innerHTML = "";
  section.hidden = citations.length === 0;
  citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    const title = document.createElement("strong");
    title.textContent = citation.title || "Untitled document";
    const excerpt = document.createElement("p");
    excerpt.textContent = citation.excerpt || "";
    const metadata = document.createElement("span");
    metadata.textContent = `${citation.document_type || "other"} | ${citation.chunk_id || "chunk"} | relevance ${formatNumber(citation.relevance_score, 3)}`;
    card.append(title, excerpt, metadata);
    container.append(card);
  });
}

function renderAgentCards(outputs) {
  const container = byId("agent-grid");
  container.innerHTML = "";
  Object.entries(outputs).forEach(([name, output]) => {
    const card = document.createElement("article");
    card.className = "agent-card";

    const header = document.createElement("div");
    header.className = "agent-card-header";
    const title = document.createElement("h4");
    title.textContent = name.replaceAll("_", " ");
    const status = document.createElement("span");
    status.className = `agent-status ${output.status || ""}`;
    status.textContent = output.status || "unknown";
    header.append(title, status);

    const score = document.createElement("div");
    score.className = "agent-score";
    const scoreValue = document.createElement("strong");
    scoreValue.textContent = output.score == null ? "-" : signedNumber(output.score, 3);
    const confidence = document.createElement("span");
    confidence.textContent = `${percentFromRatio(output.confidence)} confidence`;
    score.append(scoreValue, confidence);

    const evidence = document.createElement("p");
    evidence.textContent = output.evidence?.[0] || output.error || "No evidence returned.";
    card.append(header, score, evidence);
    container.append(card);
  });
}

function renderContributions(contributions) {
  const container = byId("contribution-list");
  container.innerHTML = "";
  Object.entries(contributions).forEach(([name, contribution]) => {
    const row = document.createElement("div");
    row.className = "contribution-row";
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    const track = document.createElement("div");
    track.className = "weight-track";
    const fill = document.createElement("div");
    fill.className = "weight-fill";
    fill.style.width = `${Math.max(0, Math.min(100, contribution.final_weight * 100))}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = percentFromRatio(contribution.final_weight);
    row.append(label, track, value);
    container.append(row);
  });
}

function renderRationale(items) {
  const list = byId("decision-rationale");
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.append(li);
  });
}

function renderForecast(payload) {
  const dates = payload.forecast_dates || [];
  const values = payload.predicted_values || [];
  const labels = [...dates];
  const points = [...values];
  if (payload.reference_value != null) {
    labels.unshift(payload.reference_date || "Reference");
    points.unshift(payload.reference_value);
  }
  requestAnimationFrame(() => {
    drawLineChart(
      byId("forecast-chart"),
      [{ name: "Forecast", values: points, color: colors.cyan }],
      labels,
    );
  });
}

function bindReview() {
  byId("review-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = byId("review-button");
    const status = byId("review-status");
    setBusy(button, true, "Comparing...");
    setStatus(status, "Comparing prediction with the actual market result...");

    const date = byId("review-date").value;
    const body = {
      prediction_id: byId("review-prediction-id").value.trim(),
      previous_close: numericValue("review-previous-close"),
      actual_close: numericValue("review-actual-close"),
      actual_for_date: date ? `${date}T15:30:00+05:30` : null,
      auto_fetch_context: byId("review-context").checked,
    };

    try {
      const result = await api("/api/v1/feedback/predictions/review", body);
      renderReviewResult(result);
      setStatus(status, "Comparison saved and agent memory updated.", "success");
    } catch (error) {
      setStatus(status, error.message, "error");
    } finally {
      setBusy(button, false, "Compare and update memory");
    }
  });
}

function renderReviewResult(result) {
  byId("review-empty").hidden = true;
  byId("review-results").hidden = false;
  const review = result.review || {};
  const memory = result.memory_guidance || {};
  const correct = Boolean(review.direction_correct);

  byId("review-outcome").className = `review-outcome ${correct ? "" : "incorrect"}`;
  byId("review-outcome-label").textContent = "Direction result";
  byId("review-outcome-title").textContent = correct ? "Correct prediction" : "Incorrect prediction";
  byId("review-outcome-title").style.color = correct ? colors.cyan : "#ff6d78";
  byId("review-predicted").textContent = percentValue(review.predicted_change_pct);
  byId("review-actual").textContent = percentValue(review.actual_change_pct);
  byId("review-error").textContent = percentValue(review.price_error_pct);
  byId("review-accuracy").textContent = memory.accuracy == null ? "-" : percentFromRatio(memory.accuracy);

  renderReviewBars(review.predicted_change_pct || 0, review.actual_change_pct || 0);
  const diagnosis = byId("review-diagnosis");
  diagnosis.innerHTML = "";
  (review.diagnosis?.reasons || []).forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    diagnosis.append(li);
  });
}

function renderReviewBars(predicted, actual) {
  const container = byId("review-bars");
  container.innerHTML = "";
  const maxMagnitude = Math.max(Math.abs(predicted), Math.abs(actual), 0.5);
  [
    { name: "Predicted", value: predicted, className: "" },
    { name: "Actual", value: actual, className: "actual" },
  ].forEach((item) => {
    const row = document.createElement("div");
    row.className = `comparison-bar ${item.className}`;
    const label = document.createElement("span");
    label.textContent = item.name;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    const width = (Math.abs(item.value) / maxMagnitude) * 48;
    fill.style.width = `${width}%`;
    fill.style.left = item.value >= 0 ? "50%" : `${50 - width}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = percentValue(item.value);
    row.append(label, track, value);
    container.append(row);
  });
}

function bindUtilityActions() {
  byId("copy-prediction-id").addEventListener("click", async () => {
    const value = byId("prediction-id").textContent;
    if (!value || value === "Not stored") {
      return;
    }
    await navigator.clipboard.writeText(value);
    byId("copy-prediction-id").textContent = "Copied";
    window.setTimeout(() => {
      byId("copy-prediction-id").textContent = "Copy";
    }, 1200);
  });

  window.addEventListener("resize", debounce(() => {
    if (state.trainingResult) {
      const history = state.trainingResult.history || {};
      drawLineChart(
        byId("history-chart"),
        [
          { name: "Train", values: history.loss || [], color: colors.blue },
          { name: "Validation", values: history.val_loss || [], color: colors.gold },
        ],
        (history.loss || []).map((_, index) => String(index + 1)),
      );
      renderTestSample(Number(byId("test-sample-select").value || 0));
    }
    if (state.autonomousResult) {
      renderForecast(state.autonomousResult.agent_outputs?.technical?.payload || {});
    }
  }, 180));
}

async function api(path, body = undefined) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok) {
    const detail = payload?.detail;
    if (typeof detail === "string") {
      throw new Error(detail);
    }
    if (Array.isArray(detail)) {
      throw new Error(detail.map((item) => item.msg || JSON.stringify(item)).join("; "));
    }
    throw new Error(`Request failed with status ${response.status}.`);
  }
  return payload;
}

function drawLineChart(canvas, series, labels) {
  const usable = series.filter((item) => Array.isArray(item.values) && item.values.length);
  if (!usable.length) {
    clearCanvas(canvas);
    return;
  }

  const width = Math.max(320, canvas.parentElement.clientWidth - 40);
  const height = Number(canvas.getAttribute("height")) || 260;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);

  const padding = { top: 20, right: 18, bottom: 38, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = usable.flatMap((item) => item.values.map(Number)).filter(Number.isFinite);
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    minimum -= 1;
    maximum += 1;
  }
  const margin = (maximum - minimum) * 0.08;
  minimum -= margin;
  maximum += margin;

  context.clearRect(0, 0, width, height);
  context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.lineWidth = 1;

  for (let line = 0; line <= 4; line += 1) {
    const y = padding.top + (plotHeight * line) / 4;
    const value = maximum - ((maximum - minimum) * line) / 4;
    context.strokeStyle = colors.grid;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillStyle = colors.muted;
    context.textAlign = "right";
    context.fillText(formatNumber(value, 2), padding.left - 10, y + 3);
  }

  const pointCount = Math.max(...usable.map((item) => item.values.length));
  usable.forEach((item) => {
    context.strokeStyle = item.color;
    context.lineWidth = 2.2;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    item.values.forEach((rawValue, index) => {
      const value = Number(rawValue);
      const x = padding.left + (pointCount === 1 ? plotWidth / 2 : (plotWidth * index) / (pointCount - 1));
      const y = padding.top + ((maximum - value) / (maximum - minimum)) * plotHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  });

  const labelCount = labels.length;
  const tickIndexes = [...new Set([0, Math.floor((labelCount - 1) / 2), labelCount - 1])].filter(
    (index) => index >= 0,
  );
  tickIndexes.forEach((index) => {
    const x = padding.left + (labelCount <= 1 ? plotWidth / 2 : (plotWidth * index) / (labelCount - 1));
    context.fillStyle = colors.muted;
    context.textAlign = index === 0 ? "left" : index === labelCount - 1 ? "right" : "center";
    context.fillText(shortLabel(labels[index]), x, height - 12);
  });
}

function clearCanvas(canvas) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function requireDataset(statusElement) {
  if (!state.ohlcvPayload) {
    setStatus(statusElement, "Upload and validate an OHLCV JSON file first.", "error");
    document.querySelector("#data").scrollIntoView({ behavior: "smooth" });
    return false;
  }
  return true;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.className = `${element.classList.contains("inline-status") ? "inline-status" : "operation-status"} ${type}`;
}

function numericValue(id) {
  return Number(byId(id).value);
}

function optionalText(id) {
  const value = byId(id).value.trim();
  return value || null;
}

function commaList(id) {
  return byId(id).value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = String(reader.result || "");
      resolve(value.includes(",") ? value.split(",", 2)[1] : value);
    };
    reader.onerror = () => reject(new Error("Could not read the selected document."));
    reader.readAsDataURL(file);
  });
}

function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${formatNumber(bytes / 1024, 1)} KB`;
  }
  return `${formatNumber(bytes / (1024 * 1024), 1)} MB`;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: Math.min(digits, 2),
  });
}

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString() : "-";
}

function signedNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${number >= 0 ? "+" : ""}${formatNumber(number, digits)}`;
}

function percentValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${signedNumber(number, 2)}%` : "-";
}

function percentFromRatio(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number * 100, 1)}%` : "-";
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value).slice(0, 10)
    : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function shortLabel(value) {
  if (value == null) {
    return "";
  }
  const text = String(value);
  return text.length > 12 ? text.slice(0, 10) : text;
}

function directionColor(direction) {
  if (direction === "up") {
    return colors.cyan;
  }
  if (direction === "down") {
    return "#ff6d78";
  }
  return colors.gold;
}

function debounce(callback, wait) {
  let timeout;
  return (...args) => {
    window.clearTimeout(timeout);
    timeout = window.setTimeout(() => callback(...args), wait);
  };
}
