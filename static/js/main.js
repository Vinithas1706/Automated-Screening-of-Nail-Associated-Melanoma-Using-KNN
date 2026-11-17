// Home page: drop a photo -> validate -> upload automatically.
(function () {
  const form = document.getElementById("upload-form");
  if (form) {
    const input = document.getElementById("image");
    const zone = document.getElementById("drop-zone");
    const errBox = document.getElementById("client-error");
    const busy = document.getElementById("uploading");
    const MAX = parseInt(form.dataset.maxMb, 10) * 1024 * 1024;
    const OK_TYPES = ["image/jpeg", "image/png"];
    const OK_EXT = /\.(jpe?g|png)$/i;

    function showError(msg) {
      errBox.textContent = msg;
      errBox.classList.remove("d-none");
      input.value = "";
    }
    function handleFile(file) {
      errBox.classList.add("d-none");
      if (!file) return;
      if (!OK_EXT.test(file.name) || (file.type && !OK_TYPES.includes(file.type))) {
        return showError("Unsupported file type. Please use a JPG, JPEG or PNG image.");
      }
      if (file.size > MAX) {
        return showError("File is too large. Maximum size is " + form.dataset.maxMb + " MB.");
      }
      busy.classList.remove("d-none");
      zone.style.pointerEvents = "none";
      form.submit();
    }
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => handleFile(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        handleFile(input.files[0]);
      }
    });
  }

  // Processing page: reveal the 5 steps one after another.
  const pipe = document.getElementById("pipeline");
  if (pipe) {
    const steps = Array.from(pipe.querySelectorAll("[data-step]"));
    const doneRow = document.getElementById("done-row");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduce) {
      document.documentElement.classList.add("js-anim");
      steps.forEach((s) => s.classList.add("wait"));
      let i = 0;
      const next = () => {
        if (i >= steps.length) { doneRow.classList.add("show"); return; }
        const s = steps[i++];
        s.classList.replace("wait", "run");
        setTimeout(() => {
          s.classList.replace("run", "done");
          s.scrollIntoView({ behavior: "smooth", block: "nearest" });
          setTimeout(next, 250);
        }, 800);
      };
      setTimeout(next, 300);
    }
  }

  // Before/after slider
  const cmp = document.getElementById("compare");
  if (cmp) {
    const range = cmp.querySelector("input[type=range]");
    const top = cmp.querySelector(".top");
    const handle = cmp.querySelector(".handle");
    const set = () => {
      top.style.clipPath = "inset(0 " + (100 - range.value) + "% 0 0)";
      handle.style.left = range.value + "%";
    };
    range.addEventListener("input", set);
    set();
  }
})();
