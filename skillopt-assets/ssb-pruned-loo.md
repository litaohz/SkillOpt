# Spreadsheet Manipulation Skill (xlsx)
This skill guides agents in manipulating Excel (.xlsx) spreadsheets using Python.
**Primary libraries**: `openpyxl` (structure-preserving read/write), `pandas` (data transformation).
Never use any other third-party libraries.
3. **Execute** `python solution.py` and verify the output file was created.
4. **Confirm** the target cells/range contain the expected values.
---
| Use case | Library |
|----------|---------|
| Preserve formulas, formatting, named ranges | `openpyxl` |
| Bulk data transformation, aggregation, sorting | `pandas` → write back with `openpyxl` |
| Simple cell read/write | `openpyxl` |
When the user provides an existing or broken formula, use it as a semantic specification: honor its referenced lookup ranges, criteria ranges, return ranges, aggregation intent, and error-handling behavior, then write the resulting values rather than guessing different source columns or leaving unevaluated formulas.
---
# --- perform manipulation ---
---
## Output Requirements
- Do not hardcode row counts or column letters — iterate over actual rows in the workbook.
- Preserve sheets and cells not mentioned in the instruction.
## Matching and Target Range Hygiene
- Create small helper functions for comparisons and numeric parsing. Normalize text by trimming, collapsing repeated spaces/NBSPs, and casefolding; when names or labels have punctuation/spacing inconsistencies, consider punctuation-insensitive keys. Parse numeric text after removing commas/currency symbols while preserving signs and decimal points; skip `None`/blank and booleans for numeric tests, and handle placeholders such as `"-"`, `"$"`, `"$0"`, blanks, and numeric zero deliberately.
- Normalize date keys deliberately: handle `datetime`/`date` objects, Excel serial numbers, and date-like strings, then compare at the granularity implied by the task, such as exact date, month, month/year, fiscal period, or year. For workday/date-window logic, compute the range in Python and exclude weekends/holidays as specified.
- For outputs that depend on other rows or lookup grids, make a first pass to build normalized dictionaries/groups/range structures, then a second pass to write results. Avoid nested full-sheet scans per row; split delimited tokens and ignore empty tokens, and treat error literals such as `#N/A` as meaningful sentinel values when the task refers to them.
- For filtered lists, summaries, and aggregations, first collect all source records/results in memory, preserving the required order, then write from the first output row and clear leftover cells below the new results in the target columns. When adding rows, copy style/alignment/number format from an existing template row when appropriate; when deleting rows, delete from bottom to top to avoid row-index shifts.
<!-- SLOW_UPDATE_START -->
When the user asks for a formula, macro, VBA code, or a fix to an Excel formula, still deliver the completed workbook state: compute the intended results in Python and write literal final values into the requested cells. Do not write formula strings unless the task explicitly says the output must contain live formulas.
After writing, reload or inspect the saved workbook and verify that every requested/evaluated target cell contains a non-formula literal where a value is expected. If a target cell is still `None` unexpectedly, fix the script before finishing.
Use existing formulas in the workbook as examples/specifications, not as output. If a cell contains a reference formula such as `=A25` or an INDEX/MATCH/SUMIFS pattern, parse what source cells/ranges/criteria it refers to, compute those results yourself, and overwrite the destination with the referenced or calculated value.
For blank-sensitive formula tasks, compute the branch explicitly: if the driving source cell is truly blank, write `None`; otherwise write the actual result such as `0`, `1`, a category label, or a lookup value. Never rely on `IF(...,"",...)` formulas to be recalculated later.
For schedule/calendar fill tasks, build a cycle-day-to-periods mapping from the schedule/template area first, then fill the daily rows across all requested class columns based on each row’s cycle day. Preserve repeated/double periods exactly as shown by the template; do not leave formulas in the schedule cells.
When a target range includes special rows such as `Total`, `Grand Total`, `min`, `max`, constraints, headers, or blank separators, do not apply ordinary row logic blindly to those rows. Compute totals as aggregates when indicated, and leave constraint/header/blank cells untouched unless explicitly requested.
