#!/usr/bin/env python3
"""Browser-based quality checks for the static Statim marketing site."""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import os
import posixpath
import re
import sys
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import unquote, urljoin, urlsplit


SITE_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/statim/"
WIDTHS = (360, 390, 430, 768, 1024, 1280, 1440)
HOME_LIMIT = 1_500_000


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, page: str, check: str, detail: str) -> None:
        self.failures.append(f"{page} [{check}] {detail}")

    def note(self, message: str) -> None:
        self.notes.append(message)


class IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name.lower() == "id" and value is not None:
                self.ids.add(value)


class PrefixHandler(http.server.SimpleHTTPRequestHandler):
    """Serve SITE_ROOT, but expose it only beneath /statim/."""

    def __init__(self, *args: object, directory: str, **kwargs: object) -> None:
        self.site_directory = Path(directory).resolve()
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, _format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path == PREFIX.rstrip("/"):
            self.send_response(301)
            self.send_header("Location", PREFIX)
            self.end_headers()
            return
        if not urlsplit(self.path).path.startswith(PREFIX):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        if not urlsplit(self.path).path.startswith(PREFIX):
            self.send_error(404)
            return
        super().do_HEAD()

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        relative = request_path[len(PREFIX) :] if request_path.startswith(PREFIX) else ""
        parts = [part for part in PurePosixPath(relative).parts if part not in ("", ".", "..")]
        return os.fspath(self.site_directory.joinpath(*parts))


@contextlib.contextmanager
def local_server() -> Iterator[str]:
    handler = functools.partial(PrefixHandler, directory=os.fspath(SITE_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{PREFIX}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def discover_pages(only: str | None, report: Report) -> list[Path]:
    pages = sorted(
        path
        for path in SITE_ROOT.rglob("*.html")
        if path.relative_to(SITE_ROOT).parts[0] != "tests"
    )
    if only is None:
        return pages

    value = unquote(urlsplit(only).path).lstrip("/")
    if value.startswith(PREFIX.lstrip("/")):
        value = value[len(PREFIX.lstrip("/")) :]
    value = posixpath.normpath(value or "index.html")
    if value.startswith("../") or value == ".." or PurePosixPath(value).is_absolute():
        report.fail("(suite)", "arguments", f"--only escapes site/: {only}")
        return []
    candidate = (SITE_ROOT / value).resolve()
    if SITE_ROOT.resolve() not in candidate.parents or candidate.suffix.lower() != ".html":
        report.fail("(suite)", "arguments", f"--only is not an HTML page in site/: {only}")
        return []
    if candidate not in pages:
        report.fail("(suite)", "arguments", f"--only page does not exist: {value}")
        return []
    return [candidate]


def describe_bytes(size: int) -> str:
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1000:.1f} kB"
    return f"{size / 1_000_000:.2f} MB"


OVERFLOW_JS = r"""
() => {
  const viewport = document.documentElement.clientWidth;
  const visible = (el) => {
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' &&
      Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
  };
  const selector = (el) => {
    if (el.id) return `${el.tagName.toLowerCase()}#${CSS.escape(el.id)}`;
    let result = el.tagName.toLowerCase();
    if (typeof el.className === 'string' && el.className.trim()) {
      result += '.' + el.className.trim().split(/\s+/).slice(0, 2).map(CSS.escape).join('.');
    }
    return result;
  };
  const results = [];
  for (const el of [document.documentElement, document.body, ...document.body.querySelectorAll('*')]) {
    if (!visible(el)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.left >= -0.5 && rect.right <= viewport + 0.5) continue;

    let scroller = false;
    let clipped = false;
    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
      const overflow = getComputedStyle(parent).overflowX;
      if (overflow === 'auto' || overflow === 'scroll') scroller = true;
      if (overflow === 'hidden' || overflow === 'clip') clipped = true;
    }
    if (scroller) continue;

    const directText = [...el.childNodes].some(
      node => node.nodeType === Node.TEXT_NODE && node.textContent.trim()
    );
    const interactive = el.matches(
      'a[href], button, input, select, textarea, [role="button"], [tabindex]'
    );
    if (clipped && !directText && !interactive) continue;

    results.push({
      element: selector(el),
      left: Math.round(rect.left * 10) / 10,
      right: Math.round(rect.right * 10) / 10,
      text: (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 80)
    });
  }
  return results.slice(0, 30);
}
"""


SEMANTICS_JS = r"""
() => ({
  lang: document.documentElement.getAttribute('lang') || '',
  title: document.title.trim(),
  description: (document.querySelector('meta[name="description" i]')?.content || '').trim(),
  viewport: (document.querySelector('meta[name="viewport" i]')?.content || '').trim(),
  headings: [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(
    h => ({level: Number(h.tagName[1]), text: h.textContent.trim().replace(/\s+/g, ' ').slice(0, 80)})
  ),
  images: [...document.images].map((img, index) => ({
    index,
    src: img.getAttribute('src') || img.currentSrc || '(no src)',
    altPresent: img.hasAttribute('alt'),
    alt: img.getAttribute('alt'),
    ariaHidden: img.getAttribute('aria-hidden') === 'true',
    role: (img.getAttribute('role') || '').toLowerCase()
  })),
  links: [...document.querySelectorAll('a[href]')].map(a => a.getAttribute('href')),
  ids: [...document.querySelectorAll('[id]')].map(el => el.id)
})
"""


IMAGES_JS = r"""
async () => {
  const failed = [];
  for (const [index, img] of [...document.images].entries()) {
    img.scrollIntoView({block: 'center', inline: 'nearest'});
    if (!img.complete || img.naturalWidth === 0) {
      await Promise.race([
        new Promise(resolve => {
          img.addEventListener('load', resolve, {once: true});
          img.addEventListener('error', resolve, {once: true});
        }),
        new Promise(resolve => setTimeout(resolve, 3000))
      ]);
    }
    if (!img.complete || img.naturalWidth === 0) {
      failed.push({index, src: img.getAttribute('src') || img.currentSrc || '(no src)'});
    }
  }
  window.scrollTo(0, 0);
  return failed;
}
"""


CONTRAST_JS = r"""
() => {
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = colorCanvas.height = 1;
  const colorContext = colorCanvas.getContext('2d', {willReadFrequently: true});
  const colorCache = new Map();
  const parse = (value) => {
    if (colorCache.has(value)) return colorCache.get(value);
    let result = null;
    try {
      colorContext.clearRect(0, 0, 1, 1);
      colorContext.fillStyle = value;
      colorContext.fillRect(0, 0, 1, 1);
      result = [...colorContext.getImageData(0, 0, 1, 1).data].map((part, index) =>
        index === 3 ? part / 255 : part
      );
    } catch (_) {
      result = null;
    }
    colorCache.set(value, result);
    return result;
  };
  const luminance = (rgb) => {
    const linear = rgb.slice(0, 3).map(v => {
      v /= 255;
      return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  };
  const selector = (el) => el.id
    ? `${el.tagName.toLowerCase()}#${CSS.escape(el.id)}`
    : el.tagName.toLowerCase() + (typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).map(CSS.escape).join('.') : '');
  const visible = (el) => {
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
  };
  const failures = [];
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: node => node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
  });
  while (walker.nextNode()) {
    const el = walker.currentNode.parentElement;
    if (!el || seen.has(el) || !visible(el)) continue;
    seen.add(el);
    const range = document.createRange();
    range.selectNodeContents(walker.currentNode);
    if (![...range.getClientRects()].some(r => r.width > 0 && r.height > 0)) continue;

    let background = null;
    const translucentLayers = [];
    let unknown = false;
    for (let current = el; current; current = current.parentElement) {
      const style = getComputedStyle(current);
      if (style.backgroundImage !== 'none') {
        unknown = true;
        break;
      }
      const color = parse(style.backgroundColor);
      if (color && color[3] >= 0.999) {
        background = color;
        break;
      }
      if (color && color[3] > 0) translucentLayers.push(color);
    }
    if (unknown || !background) continue;
    for (const layer of translucentLayers.reverse()) {
      const alpha = layer[3];
      background = [
        layer[0] * alpha + background[0] * (1 - alpha),
        layer[1] * alpha + background[1] * (1 - alpha),
        layer[2] * alpha + background[2] * (1 - alpha),
        1
      ];
    }

    const style = getComputedStyle(el);
    const foreground = parse(style.color);
    if (!foreground) continue;
    const alpha = foreground[3];
    const rendered = [
      foreground[0] * alpha + background[0] * (1 - alpha),
      foreground[1] * alpha + background[1] * (1 - alpha),
      foreground[2] * alpha + background[2] * (1 - alpha)
    ];
    const l1 = luminance(rendered);
    const l2 = luminance(background);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    const fontSize = parseFloat(style.fontSize);
    const weight = Number(style.fontWeight) || (style.fontWeight === 'bold' ? 700 : 400);
    const large = fontSize >= 24 || (fontSize >= 18.66 && weight >= 700);
    const minimum = large ? 3 : 4.5;
    if (ratio + 0.001 < minimum) {
      failures.push({
        element: selector(el),
        ratio: Math.round(ratio * 100) / 100,
        minimum,
        color: style.color,
        background: `rgb(${background.slice(0, 3).join(', ')})`,
        text: walker.currentNode.textContent.trim().replace(/\s+/g, ' ').slice(0, 80)
      });
    }
  }
  return failures;
}
"""


REDUCED_MOTION_JS = r"""
() => [...document.getAnimations({subtree: true})]
  .filter(animation => {
    const cssAnimation = typeof CSSAnimation !== 'undefined' && animation instanceof CSSAnimation;
    const cssTransition = typeof CSSTransition !== 'undefined' && animation instanceof CSSTransition;
    const duration = Number(animation.effect?.getComputedTiming().duration) || 0;
    return (cssAnimation || cssTransition) &&
      (animation.playState === 'running' || animation.pending) && duration > 10;
  })
  .map(animation => ({
    type: (typeof CSSTransition !== 'undefined' && animation instanceof CSSTransition) ? 'transition' : 'animation',
    name: animation.animationName || animation.transitionProperty || '(unnamed)',
    duration: animation.effect?.getComputedTiming().duration,
    element: animation.effect?.target?.tagName?.toLowerCase() || '(unknown)'
  }))
"""


def install_event_checks(page: object, page_name: str, report: Report, seen: set[str]) -> None:
    def record(kind: str, detail: str) -> None:
        key = f"{page_name}|{kind}|{detail}"
        if key not in seen:
            seen.add(key)
            report.fail(page_name, kind, detail)

    def on_console(message: object) -> None:
        if getattr(message, "type", "") == "error":
            record("console", getattr(message, "text", str(message)))

    def on_page_error(error: object) -> None:
        record("page error", str(error))

    def on_request_failed(request: object) -> None:
        failure = getattr(request, "failure", None)
        detail = getattr(failure, "error_text", None) if failure is not None else None
        record("request", f"{getattr(request, 'url', '(unknown)')}: {detail or 'network error'}")

    def on_response(response: object) -> None:
        status = getattr(response, "status", 0)
        if status >= 400:
            record("request", f"HTTP {status}: {getattr(response, 'url', '(unknown)')}")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)


def check_semantics(data: dict[str, object], page_name: str, report: Report) -> None:
    if not str(data["lang"]).strip():
        report.fail(page_name, "metadata", "<html lang> is missing or empty")
    if not data["title"]:
        report.fail(page_name, "metadata", "<title> is missing or empty")
    if not data["description"]:
        report.fail(page_name, "metadata", "meta description is missing or empty")
    if not data["viewport"]:
        report.fail(page_name, "metadata", "viewport meta is missing or empty")

    headings = data["headings"]
    h1_count = sum(item["level"] == 1 for item in headings)
    if h1_count != 1:
        report.fail(page_name, "headings", f"expected exactly one <h1>, found {h1_count}")
    previous = 0
    for heading in headings:
        level = heading["level"]
        if level > previous + 1:
            report.fail(
                page_name,
                "headings",
                f"skips from h{previous} to h{level} at {heading['text']!r}",
            )
        previous = level

    for image in data["images"]:
        alt = image["alt"]
        if not image["altPresent"]:
            report.fail(page_name, "images", f"image {image['src']!r} has no alt attribute")
        elif not str(alt or "").strip() and not (
            image["ariaHidden"] or image["role"] in ("presentation", "none")
        ):
            report.fail(
                page_name,
                "images",
                f"decorative image {image['src']!r} needs aria-hidden=\"true\" or role=\"presentation\"",
            )


def check_keyboard(page: object, page_name: str, report: Report) -> None:
    targets = page.eval_on_selector_all(
        "a, button",
        """elements => elements.filter(el => {
          const s = getComputedStyle(el); const r = el.getBoundingClientRect();
          return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) !== 0 &&
            r.width > 0 && r.height > 0 && !(el instanceof HTMLButtonElement && el.disabled) &&
            // content of a closed <details> is not rendered and rightly not focusable
            !(el.checkVisibility && !el.checkVisibility()) &&
            // WAI-ARIA tabs use a roving tabindex: only the selected tab is in the Tab order,
            // the others are reached with the arrow keys.
            !(el.getAttribute('role') === 'tab' && el.getAttribute('aria-selected') !== 'true' &&
              el.closest('[role="tablist"]')?.querySelector('[role="tab"][aria-selected="true"]'));
        }).map((el, index) => {
          el.dataset.qaFocusTarget = String(index);
          const s = getComputedStyle(el);
          return {
            description: `${el.tagName.toLowerCase()} \"${(el.innerText || el.getAttribute('aria-label') || el.getAttribute('href') || '').trim().replace(/\\s+/g, ' ').slice(0, 70)}\"`,
            before: [s.outlineStyle, s.outlineWidth, s.outlineColor, s.outlineOffset, s.boxShadow]
          };
        })""",
    )
    if not targets:
        return

    page.evaluate("document.activeElement instanceof HTMLElement && document.activeElement.blur()")
    reached: set[int] = set()
    focus_failures: set[int] = set()
    element_count = page.locator("*").count()
    for _ in range(max(20, element_count * 2 + 10)):
        page.keyboard.press("Tab")
        state = page.evaluate(
            """() => {
              const el = document.activeElement;
              if (!(el instanceof HTMLElement) || el.dataset.qaFocusTarget === undefined) return null;
              const s = getComputedStyle(el);
              return {
                index: Number(el.dataset.qaFocusTarget),
                style: [s.outlineStyle, s.outlineWidth, s.outlineColor, s.outlineOffset, s.boxShadow]
              };
            }"""
        )
        if state is None:
            continue
        index = state["index"]
        if index in reached:
            if len(reached) < len(targets):
                break
            continue
        reached.add(index)
        focused = state["style"]
        before = targets[index]["before"]
        outline_visible = focused[0] not in ("none", "hidden") and float(focused[1].removesuffix("px") or 0) > 0
        shadow_visible = focused[4] != "none" and focused[4] != before[4]
        if focused == before or not (outline_visible or shadow_visible):
            focus_failures.add(index)
        if len(reached) == len(targets):
            break

    for index, target in enumerate(targets):
        if index not in reached:
            report.fail(page_name, "keyboard", f"not reachable with Tab: {target['description']}")
        elif index in focus_failures:
            report.fail(page_name, "keyboard", f"no visible focus indicator: {target['description']}")
    page.eval_on_selector_all("[data-qa-focus-target]", "els => els.forEach(el => delete el.dataset.qaFocusTarget)")


def path_for_internal_url(url: str, base_url: str) -> Path | None:
    parsed = urlsplit(url)
    base = urlsplit(base_url)
    base_path = base.path if base.path.endswith("/") else base.path + "/"
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        return None
    decoded_path = unquote(parsed.path)
    if not decoded_path.startswith(base_path):
        return Path("__outside_site__")
    relative = decoded_path[len(base_path) :]
    if not relative or relative.endswith("/"):
        relative += "index.html"
    return Path(*PurePosixPath(relative).parts)


def external_link_error(raw: str, absolute: str) -> str | None:
    if any(ord(char) < 32 for char in raw):
        return "contains a control character"
    parsed = urlsplit(absolute)
    if parsed.scheme in ("http", "https"):
        return None if parsed.netloc else "HTTP(S) URL has no host"
    if parsed.scheme == "mailto":
        address = unquote(parsed.path).split(",", 1)[0]
        return None if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address) else "invalid mailto address"
    if parsed.scheme == "tel":
        return None if re.fullmatch(r"[+0-9().\s-]+", unquote(parsed.path)) else "invalid telephone URL"
    return f"unsupported or invalid URL scheme {parsed.scheme!r}"


def read_ids(path: Path) -> set[str]:
    parser = IdParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.ids


def check_links(
    links_by_page: dict[str, list[str]], base_url: str, report: Report
) -> None:
    ids_cache: dict[Path, set[str]] = {}
    base = urlsplit(base_url)
    for page_name, links in links_by_page.items():
        source_url = urljoin(base_url, page_name)
        for raw in dict.fromkeys(links):
            raw = raw.strip()
            if not raw:
                continue
            try:
                absolute = urljoin(source_url, raw)
                parsed = urlsplit(absolute)
                raw_scheme = urlsplit(raw).scheme
            except ValueError as error:
                report.fail(page_name, "links", f"{raw!r} is malformed: {error}")
                continue
            is_same_origin = (parsed.scheme, parsed.netloc) == (base.scheme, base.netloc)
            if not is_same_origin or raw_scheme in ("mailto", "tel"):
                error = external_link_error(raw, absolute)
                if error:
                    report.fail(page_name, "links", f"{raw!r}: {error}")
                continue

            relative = path_for_internal_url(absolute, base_url)
            if relative == Path("__outside_site__"):
                report.fail(page_name, "links", f"{raw!r} points outside the site prefix")
                continue
            if relative is None:
                continue
            target = SITE_ROOT.joinpath(relative)
            try:
                target.relative_to(SITE_ROOT)
            except ValueError:
                report.fail(page_name, "links", f"{raw!r} escapes site/")
                continue
            if not target.is_file():
                report.fail(page_name, "links", f"{raw!r} targets missing file {relative.as_posix()!r}")
                continue
            if parsed.fragment:
                if target.suffix.lower() != ".html":
                    report.fail(page_name, "links", f"{raw!r} has an anchor on a non-HTML file")
                    continue
                resolved = target.resolve()
                if resolved not in ids_cache:
                    ids_cache[resolved] = read_ids(resolved)
                anchor = unquote(parsed.fragment)
                if anchor not in ids_cache[resolved]:
                    report.fail(page_name, "links", f"{raw!r} targets missing id {anchor!r}")


def wait_after_load(page: object) -> None:
    page.wait_for_load_state("load", timeout=15_000)
    page.wait_for_timeout(250)


def audit_page(browser: object, path: Path, base_url: str, report: Report, seen: set[str]) -> tuple[list[str], int]:
    page_name = path.relative_to(SITE_ROOT).as_posix()
    url = urljoin(base_url, page_name)
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    install_event_checks(page, page_name, report, seen)
    session = context.new_cdp_session(page)
    session.send("Network.enable")
    session.send("Network.setCacheDisabled", {"cacheDisabled": True})
    transferred = 0

    def add_bytes(event: dict[str, object]) -> None:
        nonlocal transferred
        transferred += int(event.get("encodedDataLength", 0))

    session.on("Network.loadingFinished", add_bytes)
    links: list[str] = []
    try:
        response = page.goto(url, wait_until="load", timeout=15_000)
        if response is None or response.status >= 400:
            report.fail(page_name, "load", f"could not load {url}")
            return links, transferred
        wait_after_load(page)

        overflows = page.evaluate(OVERFLOW_JS)
        for item in overflows:
            report.fail(
                page_name,
                "overflow 1440px",
                f"{item['element']} spans x={item['left']}..{item['right']} ({item['text']!r})",
            )

        data = page.evaluate(SEMANTICS_JS)
        check_semantics(data, page_name, report)
        links = data["links"]

        for failed in page.evaluate(IMAGES_JS):
            report.fail(page_name, "images", f"did not load: {failed['src']!r}")

        for item in page.evaluate(CONTRAST_JS):
            report.fail(
                page_name,
                "contrast",
                f"{item['element']} is {item['ratio']}:1 (needs {item['minimum']}:1), "
                f"{item['color']} on {item['background']}: {item['text']!r}",
            )

        check_keyboard(page, page_name, report)
        page.wait_for_timeout(250)
        weight = transferred

        for width in WIDTHS:
            if width == 1440:
                continue
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(url, wait_until="load", timeout=15_000)
            wait_after_load(page)
            for item in page.evaluate(OVERFLOW_JS):
                report.fail(
                    page_name,
                    f"overflow {width}px",
                    f"{item['element']} spans x={item['left']}..{item['right']} ({item['text']!r})",
                )
        return links, weight
    except Exception as error:  # Playwright errors should become readable suite failures.
        report.fail(page_name, "runner", f"{type(error).__name__}: {error}")
        return links, transferred
    finally:
        context.close()


def check_reduced_motion(browser: object, path: Path, base_url: str, report: Report, seen: set[str]) -> None:
    page_name = path.relative_to(SITE_ROOT).as_posix()
    context = browser.new_context(
        viewport={"width": 1440, "height": 900}, reduced_motion="reduce"
    )
    page = context.new_page()
    install_event_checks(page, page_name, report, seen)
    try:
        page.goto(urljoin(base_url, page_name), wait_until="load", timeout=15_000)
        page.wait_for_timeout(20)
        for item in page.evaluate(REDUCED_MOTION_JS):
            report.fail(
                page_name,
                "reduced motion",
                f"running CSS {item['type']} {item['name']!r} lasts {item['duration']}ms on {item['element']}",
            )
    except Exception as error:
        report.fail(page_name, "reduced motion", f"{type(error).__name__}: {error}")
    finally:
        context.close()


def run_browser_checks(pages: list[Path], base_url: str, report: Report) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        report.fail(
            "(suite)",
            "setup",
            "Playwright is not installed; run: python -m pip install playwright==1.63.0",
        )
        return

    links_by_page: dict[str, list[str]] = {}
    seen_events: set[str] = set()
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:
            report.fail(
                "(suite)",
                "setup",
                f"Chromium could not start ({error}); run: python -m playwright install chromium",
            )
            return
        try:
            for index, path in enumerate(pages, start=1):
                page_name = path.relative_to(SITE_ROOT).as_posix()
                print(f"[{index}/{len(pages)}] {page_name}", flush=True)
                links, weight = audit_page(browser, path, base_url, report, seen_events)
                links_by_page[page_name] = links
                report.note(f"{page_name}: {describe_bytes(weight)} transferred")
                if page_name == "index.html" and weight > HOME_LIMIT:
                    report.fail(
                        page_name,
                        "page weight",
                        f"{describe_bytes(weight)} exceeds the 1.5 MB home-page limit",
                    )
                check_reduced_motion(browser, path, base_url, report, seen_events)
        finally:
            browser.close()

    check_links(links_by_page, base_url, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        help="base URL of an existing server (for example http://localhost:8000/statim/)",
    )
    parser.add_argument(
        "--only",
        metavar="PAGE",
        help="check one site-relative HTML page (for example docs/index.html)",
    )
    return parser.parse_args()


def print_report(report: Report, page_count: int) -> int:
    print("\nSite QA report")
    print("==============")
    print(f"Pages checked: {page_count}")
    if report.notes:
        print("\nPage weights:")
        for note in report.notes:
            print(f"  - {note}")
    if report.failures:
        print(f"\nFailures ({len(report.failures)}):")
        for failure in report.failures:
            print(f"  - {failure}")
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


def main() -> int:
    args = parse_args()
    report = Report()
    pages = discover_pages(args.only, report)
    if not pages:
        if not report.failures:
            report.fail("(suite)", "discovery", "no HTML pages found under site/ (site/tests/ is excluded)")
        return print_report(report, 0)

    if args.url:
        base_url = args.url if args.url.endswith("/") else args.url + "/"
        run_browser_checks(pages, base_url, report)
    else:
        with local_server() as base_url:
            print(f"Serving {SITE_ROOT} at {base_url}")
            run_browser_checks(pages, base_url, report)
    return print_report(report, len(pages))


if __name__ == "__main__":
    raise SystemExit(main())
