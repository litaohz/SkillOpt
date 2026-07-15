---
name: xlsx-skill
description: Manipulate Excel (.xlsx) spreadsheets with Python (openpyxl / pandas).
allowed-tools: [bash, python]
---

# Spreadsheet Manipulation Skill (xlsx)

## Overview

This skill guides agents in manipulating Excel (.xlsx) spreadsheets using Python.

**Primary libraries**: `openpyxl` (structure-preserving read/write), `pandas`
(data transformation). Never use any other third-party libraries.

## Common Workflow

1. **Explore** the input file: run `python scripts/inspect.py` to list sheets,
   inspect headers, and check dimensions.
2. **Write `solution.py`** using `scripts/template.py` as the skeleton — it
   defines `INPUT_PATH` and `OUTPUT_PATH` at the top.
3. Choose the library according to the Library Selection table.
4. **Execute** `python solution.py` and verify the output file was created.
5. **Confirm** the target cells/range contain the expected values.

## Library Selection

| Use case | Library |
|----------|---------|
| Preserve formulas, formatting, named ranges | `openpyxl` |
| Bulk data transformation, aggregation, sorting | `pandas` → write back with `openpyxl` |
| Simple cell read/write | `openpyxl` |

### Warning

`pandas.to_excel()` silently destroys existing formulas and named ranges. When
writing back to a spreadsheet that contains formulas, always use
`openpyxl.save()`. See [openpyxl pitfalls](references/openpyxl-pitfalls.md) for
the full list.

## Worked Example

Following the Common Workflow: first run `scripts/inspect.py` to discover the
sheet and headers, then adapt `scripts/template.py`, load with `openpyxl`, edit
the target range, and write to `OUTPUT_PATH`.

## Output Requirements

- Save the result to `OUTPUT_PATH`.
- Do not hardcode row counts or column letters — iterate over actual rows in the
  workbook.
- Preserve sheets and cells not mentioned in the instruction.
