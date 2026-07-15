"""Skeleton for solution.py — copy this and fill in the manipulation.

INPUT_PATH / OUTPUT_PATH are defined at the top; the Output Requirements section
of SKILL.md refers to OUTPUT_PATH defined here.
"""
import openpyxl
import pandas as pd  # noqa: F401  (available for bulk transforms)

INPUT_PATH = "..."   # set to the actual input path
OUTPUT_PATH = "..."  # set to the actual output path

wb = openpyxl.load_workbook(INPUT_PATH)
ws = wb.active  # or wb["SheetName"]

# --- perform manipulation ---

wb.save(OUTPUT_PATH)
