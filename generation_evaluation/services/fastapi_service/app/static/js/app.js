// Small enhancements; every page works without this file.
(function () {
  "use strict";

  // While an evaluation runs (it can take minutes), lock the button and show the elapsed time.
  document.querySelectorAll("form[data-busy-form]").forEach(function (form) {
    form.addEventListener("submit", function () {
      const button = form.querySelector("button[type=submit]");
      const note = form.querySelector(".busy-note");
      if (button) {
        button.setAttribute("aria-busy", "true");
        if (button.dataset.busyLabel) button.textContent = button.dataset.busyLabel;
        // Disable after the submit has been sent, so the form data still goes out.
        setTimeout(function () { button.disabled = true; }, 0);
      }
      if (note) {
        const started = Date.now();
        note.hidden = false;
        note.textContent = "0 s";
        setInterval(function () {
          note.textContent = Math.round((Date.now() - started) / 1000) + " s";
        }, 1000);
      }
    });
  });

  // A framework picker only matters when its metric is ticked.
  document.querySelectorAll(".metric-choice").forEach(function (row) {
    const box = row.querySelector("input[type=checkbox]");
    const select = row.querySelector("select");
    if (!box || !select) return;
    const sync = function () { select.disabled = !box.checked; };
    box.addEventListener("change", sync);
    sync();
  });
})();
