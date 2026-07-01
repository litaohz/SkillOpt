# OfficeQA Skill

## Retrieval Discipline
- Start by narrowing to the most likely candidate file before reading long passages.
- Prefer targeted search terms that name the exact entity, period, measure, or table concept from the question.
- After a promising match, read only a small surrounding span and verify it matches the requested year, basis, and unit.

- When the question names a chart or graph, treat the plotted chart data/crossover as the target evidence; do not substitute nearby narrative summaries or adjacent tables unless they contain the same measure, period, and chart concept.

## Evidence Discipline
- Extract the exact value from the retrieved text before doing any arithmetic.
- Keep track of each operand's period, unit, and semantic role so nearby proxy values are not mixed in.
- If the question asks for a transformed or derived quantity, compute only after confirming every operand.

- For range-based or time-series calculations, make a checked operand list with the expected count (months, fiscal years, or year-to-year intervals) before computing; use population formulas when the prompt specifies population standard deviation or coefficient of variation.
- When the prompt names inclusions, exclusions, retirements, revisions, or special conventions, read the nearby table notes/footnotes and choose the row or column whose definition already matches those conditions.

- Before statistical calculations, write down the exact requested series and formula, including whether rates are percentages, percentage points, decimals, annualized rates, or period rates; convert annualized quarterly rates to quarterly multipliers before compounding or taking geometric means.
- For tail-risk/loss questions, compute the requested return/change distribution first, apply the specified tail probability with the correct sign convention so loss is positive, and perform currency/unit conversion only once at the end using the requested date and frequency.

## Final Answer Discipline
- Return the final answer only after one last consistency check against the retrieved evidence.
- Copy the final answer from a checked value, not from an unverified intermediate guess.

- Match the requested output form exactly: if the prompt asks for just a year, decimal, rounded number, or bracketed list, return only that value/list and omit units, prose, qualifiers such as "around," and labels like "percentage points" unless explicitly requested.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
