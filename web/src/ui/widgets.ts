/**
 * Minimal form widgets. Every input calls the shared change handler, and the
 * app recomputes everything that depends on the inputs (like Streamlit's rerun).
 */

let onChange: () => void = () => {};
export const setChangeHandler = (f: () => void): void => {
  onChange = f;
};

type Child = Node | string | null | undefined | false;

/** Create an element: h("div.cls1.cls2", {attrs}, ...children). */
export function h<K extends keyof HTMLElementTagNameMap>(
  tagAndClasses: K | `${K}.${string}`, attrs: Record<string, string | boolean | number> = {}, ...children: Child[]
): HTMLElementTagNameMap[K] {
  const [tag, ...classes] = tagAndClasses.split(".") as [K, ...string[]];
  const el = document.createElement(tag);
  if (classes.length) el.className = classes.join(" ");
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false) continue;
    if (k === "html") el.innerHTML = String(v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}

export const row = (...cols: Child[]): HTMLDivElement => h("div.row", {}, ...cols);
export const col = (...items: Child[]): HTMLDivElement => h("div.col", {}, ...items);

function field(label: string, control: HTMLElement, help?: string): HTMLElement {
  return h("label.field", help ? { title: help } : {}, h("span.label", {}, label, help ? h("span.help", {}, " ⓘ") : null), control);
}

// ------------------------------------------------------------------ number

export interface NumberWidget {
  el: HTMLElement;
  input: HTMLInputElement;
  value: number;
  set(v: number): void;
  disabled: boolean;
  /** Follow a computed default until the user edits; reset when `key` changes. */
  follow(key: string, v: number): void;
}

export function numberInput(label: string, value: number, opts: {
  min?: number; max?: number; step?: number | "any"; help?: string; digits?: number;
} = {}): NumberWidget {
  const { min = -Infinity, max = Infinity, step = "any", help, digits } = opts;
  const show = (v: number) => (digits !== undefined ? String(Number(v.toPrecision(digits))) : String(v));
  const input = h("input", { type: "number", step: String(step), value: show(value) });
  if (Number.isFinite(min)) input.min = String(min);
  if (Number.isFinite(max)) input.max = String(max);
  let last = value;
  let edited = false;
  let key: string | null = null;
  input.addEventListener("input", () => {
    edited = true;
    onChange();
  });
  input.addEventListener("change", () => {
    input.value = show(w.value); // show the clamped value
  });
  const w: NumberWidget = {
    el: field(label, input, help),
    input,
    get value() {
      const v = Number(input.value);
      if (input.value.trim() === "" || !Number.isFinite(v)) return last;
      last = Math.min(Math.max(v, min), max);
      return last;
    },
    set(v: number) {
      last = v;
      input.value = show(v);
    },
    get disabled() {
      return input.disabled;
    },
    set disabled(b: boolean) {
      input.disabled = b;
      w.el.classList.toggle("disabled", b);
    },
    follow(k: string, v: number) {
      if (k !== key) {
        key = k;
        edited = false;
      }
      if (!edited && Number(input.value) !== v) w.set(v);
    },
  };
  return w;
}

// ------------------------------------------------------------------ text

export interface TextWidget {
  el: HTMLElement;
  value: string;
  follow(key: string, v: string): void;
}

export function textInput(label: string, value: string, help?: string): TextWidget {
  const input = h("input", { type: "text", value, spellcheck: "false" });
  let edited = false;
  let key: string | null = null;
  input.addEventListener("input", () => {
    edited = true;
    onChange();
  });
  return {
    el: field(label, input, help),
    get value() {
      return input.value;
    },
    follow(k: string, v: string) {
      if (k !== key) {
        key = k;
        edited = false;
      }
      if (!edited && input.value !== v) input.value = v;
    },
  };
}

// ------------------------------------------------------------------ select

export interface Option<T extends string> {
  value: T;
  label: string;
}

export interface SelectWidget<T extends string> {
  el: HTMLElement;
  value: T;
  /** Replace the options; keep the current value if still present, else use `fallback`. */
  setOptions(options: Option<T>[], fallback?: T): void;
  /** Reset to `v` when `key` changes. */
  resetOn(key: string, v: T): void;
}

export function select<T extends string>(label: string, options: Option<T>[], value?: T, help?: string): SelectWidget<T> {
  const sel = h("select");
  let key: string | null = null;
  sel.addEventListener("change", () => onChange());
  const w: SelectWidget<T> = {
    el: field(label, sel, help),
    get value() {
      return sel.value as T;
    },
    set value(v: T) {
      sel.value = v;
    },
    setOptions(opts, fallback) {
      const cur = sel.value;
      const same = sel.options.length === opts.length &&
        opts.every((o, i) => sel.options[i].value === o.value && sel.options[i].text === o.label);
      if (!same) {
        sel.replaceChildren(...opts.map((o) => h("option", { value: o.value }, o.label)));
      }
      if (opts.some((o) => o.value === cur)) sel.value = cur;
      else if (fallback !== undefined) sel.value = fallback;
    },
    resetOn(k, v) {
      if (k !== key) {
        key = k;
        sel.value = v;
      }
    },
  };
  w.setOptions(options, value ?? options[0]?.value);
  if (value !== undefined) sel.value = value;
  return w;
}

// ------------------------------------------------------------------ radio

export interface RadioWidget<T extends string> {
  el: HTMLElement;
  value: T;
  setOptions(options: Option<T>[], fallback: T): void;
}

let radioCount = 0;

export function radio<T extends string>(label: string, options: Option<T>[], value: T, horizontal = true): RadioWidget<T> {
  const name = `radio${radioCount++}`;
  const box = h("div", { class: horizontal ? "radios horizontal" : "radios" });
  let current = value;
  const build = (opts: Option<T>[]) => {
    box.replaceChildren(...opts.map((o) => {
      const inp = h("input", { type: "radio", name, value: o.value });
      inp.checked = o.value === current;
      inp.addEventListener("change", () => {
        current = o.value;
        onChange();
      });
      return h("label.radio", {}, inp, " ", o.label);
    }));
  };
  build(options);
  let opts = options;
  return {
    el: h("div.field", {}, label ? h("span.label", {}, label) : null, box),
    get value() {
      return current;
    },
    set value(v: T) {
      current = v;
      build(opts);
    },
    setOptions(o, fallback) {
      const changed = o.length !== opts.length || o.some((x, i) => x.value !== opts[i].value);
      opts = o;
      if (!o.some((x) => x.value === current)) current = fallback;
      if (changed) build(o);
    },
  };
}

// ------------------------------------------------------------------ checkbox

export interface CheckWidget {
  el: HTMLElement;
  value: boolean;
}

export function checkbox(label: string, value: boolean, help?: string): CheckWidget {
  const inp = h("input", { type: "checkbox" });
  inp.checked = value;
  inp.addEventListener("change", () => onChange());
  return {
    el: h("label.check", help ? { title: help } : {}, inp, " ", label),
    get value() {
      return inp.checked;
    },
    set value(v: boolean) {
      inp.checked = v;
    },
  };
}

// ------------------------------------------------------------------ files

export interface LoadedFile {
  name: string;
  text: string;
}

export interface FileWidget {
  el: HTMLElement;
  files: LoadedFile[];
}

export function fileInput(label: string, multiple = false, help?: string): FileWidget {
  const inp = h("input", { type: "file", multiple });
  const clear = h("button.small", { type: "button" }, "Clear");
  clear.hidden = true;
  const w: FileWidget = { el: field(label, h("div.filebox", {}, inp, clear), help), files: [] };
  inp.addEventListener("change", async () => {
    w.files = await Promise.all([...(inp.files ?? [])].map(async (f) => ({ name: f.name, text: await f.text() })));
    clear.hidden = !w.files.length;
    onChange();
  });
  clear.addEventListener("click", () => {
    inp.value = "";
    w.files = [];
    clear.hidden = true;
    onChange();
  });
  return w;
}

// ------------------------------------------------------------------ output pieces

export interface Metric {
  label: string;
  value: string;
  help?: string;
  delta?: string;
}

export function metrics(items: Metric[]): HTMLElement {
  return h("div.metrics", {}, ...items.map((m) =>
    h("div.metric", m.help ? { title: m.help } : {},
      h("div.metric-label", {}, m.label, m.help ? h("span.help", {}, " ⓘ") : null),
      h("div.metric-value", {}, m.value),
      m.delta ? h("div", { class: `metric-delta ${m.delta.startsWith("-") ? "neg" : "pos"}` },
                  m.delta.startsWith("-") ? `↓ ${m.delta.slice(1)}` : `↑ ${m.delta}`) : null)));
}

export const warningBox = (text: string): HTMLElement => h("div.alert.warning", {}, "⚠️ ", text);
export const errorBox = (text: string): HTMLElement => h("div.alert.error", {}, text);
export const infoBox = (text: string): HTMLElement => h("div.alert.info", {}, text);
export const successBox = (text: string): HTMLElement => h("div.alert.success", {}, text);

/** Text with **bold** segments. */
export function rich(text: string, cls = "caption"): HTMLElement {
  const el = h("p", { class: cls });
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**")) el.append(h("strong", {}, part.slice(2, -2)));
    else if (part) el.append(part);
  });
  return el;
}

export function expander(title: string | (() => string), open = false, ...children: Child[]): HTMLDetailsElement & { setTitle(t: string): void } {
  const summary = h("summary", {}, typeof title === "string" ? title : title());
  const d = h("details.expander", open ? { open: true } : {}, summary, h("div.expander-body", {}, ...children)) as HTMLDetailsElement & { setTitle(t: string): void };
  d.setTitle = (t: string) => {
    summary.textContent = t;
  };
  return d;
}

export function downloadButton(label: string, content: string, filename: string, mime = "text/csv"): HTMLElement {
  const btn = h("button", { type: "button" }, label);
  btn.addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([content], { type: mime }));
    const a = h("a", { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  return btn;
}

/** A slot whose content is replaced on each recompute. */
export class Slot {
  readonly el = h("div.slot");
  set(...children: Child[]): void {
    this.el.replaceChildren(...children.filter((c): c is Node | string => c !== null && c !== undefined && c !== false));
  }
}
