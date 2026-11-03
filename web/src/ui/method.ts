/** The Method tab: equations and limitations, rendered with KaTeX. */
import katex from "katex";

const m = (tex: string) => katex.renderToString(tex, { throwOnError: false });
const M = (tex: string) => katex.renderToString(tex, { throwOnError: false, displayMode: true });

export function methodHtml(): string {
  return `
<h3>What is calculated</h3>

<p><strong>Incident flux.</strong> The built-in I₀ is the sum of the reference-pixel columns of the lab
spectrometer's scans, one measured window per element edge (V to Kr, 5.3 to 14.6 keV). By default the windows
are joined into one smoothed curve (log I₀, 100 eV bin medians, gaps interpolated, 300 eV Gaussian): a broad
estimate within about ±15 % of each window. Rates are divided by the reference-scan dwell and the detector gain
to give effective counts per second.</p>

<p><strong>Cross sections.</strong> Mass attenuation coefficients μ/ρ(E) from the Elam tables, summed over
element mass fractions. The edge step of the absorber is its photoabsorption above the edge minus a log-log
extrapolation of the pre-edge, so it decays with energy like a real atomic background.</p>

<p><strong>Transmission.</strong> For sample mass ${m("m_s")} and matrix mass ${m("m_m")} on area ${m("A")}:
${m("\\mu t(E) = [m_s(\\mu/\\rho)_s + m_m(\\mu/\\rho)_m]/A")} and
${m("\\Delta\\mu t = m_{abs}\\,\\Delta(\\mu/\\rho)_{abs}/A")}.
Counts are ${m("N_0 = I_0(E)\\,t_0")} and ${m("N = I_0(E)\\,t\\,e^{-\\mu t}")}, and</p>
${M("\\sigma^2(\\mu t) = \\frac{\\mathrm{var}(N)}{N^2} + \\frac{\\mathrm{var}(N_0)}{N_0^2},\\qquad \\mathrm{var}(S) = n_f\\sigma_r^2 + D t + S + (\\varepsilon S)^2")}
<p>in effective counts. The matrix only adds absorption, so the planner uses the least matrix the pellet rule
allows and scans the sample mass for the best edge-step SNR (${m("\\Delta\\mu t/\\sigma")} at E₀+50 eV) or the
lowest rms noise of ${m("k^w\\chi(k)")}, where ${m("\\sigma_\\chi = \\sigma(\\mu t)/\\Delta\\mu t(E)")}. Caps on
Δμt and μt guard against thickness effects and starved detector counts.</p>

<p><strong>Fluorescence.</strong> Uniform pellet, angles θ<sub>in</sub>/θ<sub>out</sub> from the surface:
${m("I_f/I_0 = \\frac{\\Omega}{4\\pi}\\eta Y \\frac{a/\\sin\\theta_{in}}{X}(1-e^{-X})")} with
${m("X = \\tau/\\sin\\theta_{in} + \\tau_f/\\sin\\theta_{out}")}. The EXAFS amplitude retention
${m("R = \\partial\\ln I_f/\\partial\\ln a")} measures self-absorption.</p>

<p><strong>Counting time.</strong> For a target rms σ(kʷχ), two options are compared at the planned masses.
Averaging N independent scan pairs divides every noise term by √N. Multiplying every dwell by f reduces shot,
dark and read noise roughly as 1/√f, but not source instability, which sets a floor (≈ √2·ε/Δμt per point in
transmission). If the target is below that floor, repeat scans instead. A fixed overhead per point is charged
once per point in every scan.</p>

<p><strong>Limitations.</strong> Tabulated cross sections have no white line or EXAFS: load a reference spectrum
for peak values. Pellet inhomogeneity, harmonics, and particle-size effects are not modeled; the edge-step cap is
the practical guard.</p>

<p class="caption">All calculations run in your browser; uploaded files are never sent anywhere.</p>`;
}
