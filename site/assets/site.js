// Progressive enhancement only: the page works without this file.
document.documentElement.classList.add("js");

document.addEventListener("DOMContentLoaded", () => {
  // Accessible tabs (WAI-ARIA tabs pattern) for the client examples.
  for (const box of document.querySelectorAll("[data-tabs]")) {
    const tabs = [...box.querySelectorAll('[role="tab"]')];
    const select = (tab, focus) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute("aria-selected", on);
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      }
      if (focus) tab.focus();
    };
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab, false));
      tab.addEventListener("keydown", (e) => {
        const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
        if (next === undefined) return;
        e.preventDefault();
        select(tabs[(next + tabs.length) % tabs.length], true);
      });
    });
  }

  // Copy the visible code block next to a copy button.
  for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async () => {
      const box = button.closest(".code");
      const panel = box.querySelector('[role="tabpanel"]:not([hidden]) pre') || box.querySelector("pre");
      try {
        await navigator.clipboard.writeText(panel.innerText.trim());
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select and copy";
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1800);
    });
  }

  // Close the mobile menu after choosing a link.
  const menu = document.querySelector(".menu");
  if (menu) menu.addEventListener("click", (e) => { if (e.target.closest("a")) menu.open = false; });
});
