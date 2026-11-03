# massSigray web

Browser version of the massSigray XAS sample planner, a TypeScript port of the frozen
Python/Streamlit app (git tag `python-final`). Everything runs in the visitor's browser:
there is no server, and uploaded files never leave the computer.

## Develop

    npm run dev        # live-reloading dev server at http://localhost:5173
    npm test           # unit tests + comparison against the frozen Python results
    npm run typecheck

## Deploy

    npm run build      # type-checks, then writes the static site to dist/

Upload the **contents** of `dist/` to any folder on the web server, for example
`rodsmithlab.ca/masssigray/`. Paths are relative, so any folder works. The bundle is a
plain (non-module) script, so `dist/index.html` also opens directly from disk
(`file://`), which is handy for a quick check or for sharing the folder.

Tabs can be linked directly: `.../masssigray/#fluorescence`, `#foil`, `#calibration`, `#method`.

## Layout

| Path | Contents |
|---|---|
| `src/core/` | Physics, one module per Python module (`materials`, `xray`, `flux`, `detector`, `scan`, `transmission`, `fluorescence`, `timing`, `simulate`), plus `xraydb.ts` (Elam tables, formula parser) and `num.ts` (numpy/scipy replacements: interp, nnls, brentq, polyfit, ...) |
| `src/ui/` | Widgets, Plotly charts, the Method text |
| `src/main.ts` | The app: sidebar, sample inputs, tabs |
| `src/data/` | Generated data: `xraydb.json` and the instrument I0 tables |
| `tests/golden.json` | Reference results from the frozen Python code |

## Regenerating data (Python, from the repo root)

    python scripts/build_i0_reference.py source/*.csv   # after new spectrometer exports
    python scripts/export_web_data.py                   # copies I0 tables + xraydb tables into src/data
    python scripts/make_golden.py                       # only if the reference cases change
