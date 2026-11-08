// Small progressive enhancements. Every page works without this file.
(function () {
  "use strict";

  // ---- Tooltips: elements with data-tip-value / data-tip-label (and optional data-tip-rows JSON) ----
  const tooltip = document.getElementById("tooltip");

  function fillTooltip(el) {
    tooltip.replaceChildren();
    const rows = el.dataset.tipRows ? JSON.parse(el.dataset.tipRows) : null;
    if (el.dataset.tipTitle) {
      const title = document.createElement("div");
      title.className = "tooltip__label";
      title.textContent = el.dataset.tipTitle;
      tooltip.append(title);
    }
    if (rows) {
      rows.forEach(function (row) {
        const line = document.createElement("div");
        line.className = "tooltip__row";
        const key = document.createElement("span");
        key.className = "tooltip__key" + (row.deemph ? " tooltip__key--deemph" : "");
        const value = document.createElement("span");
        value.className = "tooltip__value";
        value.textContent = row.value;
        const label = document.createElement("span");
        label.className = "tooltip__label";
        label.textContent = row.label;
        line.append(key, value, label);
        tooltip.append(line);
      });
    } else {
      const value = document.createElement("div");
      value.className = "tooltip__value";
      value.textContent = el.dataset.tipValue || "";
      tooltip.append(value);
    }
    if (el.dataset.tipNote) {
      const note = document.createElement("div");
      note.className = "tooltip__label";
      note.textContent = el.dataset.tipNote;
      tooltip.append(note);
    }
  }

  function place(x, y) {
    const pad = 12;
    const rect = tooltip.getBoundingClientRect();
    let left = x + pad;
    let top = y + pad;
    if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
    if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
    tooltip.style.left = Math.max(8, left) + "px";
    tooltip.style.top = Math.max(8, top) + "px";
  }

  if (tooltip) {
    document.querySelectorAll("[data-tip-value], [data-tip-rows]").forEach(function (el) {
      el.addEventListener("pointerenter", function () { fillTooltip(el); tooltip.hidden = false; });
      el.addEventListener("pointermove", function (e) { place(e.clientX, e.clientY); });
      el.addEventListener("pointerleave", function () { tooltip.hidden = true; });
      el.addEventListener("focus", function () {
        fillTooltip(el);
        tooltip.hidden = false;
        const r = el.getBoundingClientRect();
        place(r.left + r.width / 2, r.top + r.height / 2);
      });
      el.addEventListener("blur", function () { tooltip.hidden = true; });
    });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") tooltip.hidden = true; });
  }

  // ---- Confirm destructive actions ----
  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      if (!window.confirm(form.dataset.confirm)) e.preventDefault();
    });
  });

  // ---- Busy state on long-running submits (evaluation runs synchronously) ----
  document.querySelectorAll("form[data-busy-label]").forEach(function (form) {
    form.addEventListener("submit", function () {
      const button = form.querySelector("button[type=submit]");
      if (!button) return;
      window.setTimeout(function () {
        button.setAttribute("aria-busy", "true");
        button.disabled = true;
        button.textContent = form.dataset.busyLabel;
      }, 0);
    });
  });

  // ---- Evaluation form: show only the settings of the chosen retriever ----
  const retrieverInputs = document.querySelectorAll("input[name=retriever_type]");
  const panels = document.querySelectorAll("[data-retriever-panel]");
  function syncPanels() {
    const checked = document.querySelector("input[name=retriever_type]:checked");
    const kind = checked ? checked.value : "";
    panels.forEach(function (panel) {
      const kinds = panel.dataset.retrieverPanel.split(" ");
      panel.hidden = kinds.indexOf(kind) === -1;
    });
  }
  if (retrieverInputs.length) {
    retrieverInputs.forEach(function (input) { input.addEventListener("change", syncPanels); });
    syncPanels();
  }

  // ---- Warn when a built-in retriever is chosen for a dataset without a chunk corpus ----
  const datasetSelect = document.querySelector("select[name=dataset]");
  const noCorpus = document.getElementById("no-corpus-warning");
  if (datasetSelect && noCorpus) {
    const without = JSON.parse(noCorpus.dataset.datasets || "[]").map(String);
    const sync = function () {
      const checked = document.querySelector("input[name=retriever_type]:checked");
      const builtin = checked && ["bm25", "tfidf", "fusion"].indexOf(checked.value) !== -1;
      noCorpus.hidden = !(builtin && without.indexOf(datasetSelect.value) !== -1);
    };
    datasetSelect.addEventListener("change", sync);
    retrieverInputs.forEach(function (input) { input.addEventListener("change", sync); });
    sync();
  }
})();
