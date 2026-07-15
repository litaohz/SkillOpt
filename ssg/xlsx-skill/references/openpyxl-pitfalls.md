# openpyxl / pandas pitfalls

Referenced by the **Warning** under Library Selection in `SKILL.md`.

- `pandas.to_excel()` **silently destroys existing formulas and named ranges**.
  When writing back to a workbook that contains formulas, load and save with
  `openpyxl` instead.
- `openpyxl` writes formulas but does **not** evaluate them — the cached value is
  stale until the file is reopened in Excel. If you need the computed value, load
  a second copy with `data_only=True`.
- Preserve number formats and styles: read/modify/write cell-by-cell with
  `openpyxl` rather than round-tripping the sheet through a DataFrame.
- Named ranges and defined names live on the workbook, not the worksheet — do not
  drop them when rewriting.
