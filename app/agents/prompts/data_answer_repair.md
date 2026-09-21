# One data-reference repair

The previous claims contained invalid references. Regenerate DataAnswerDraft
with its claims list only. Use exact JSON values and types from the supplied
generation_block and valid cell/statistic/row-count locations. Data facts require
supporting data_refs. Do not repeat unsupported numbers or return a prose summary.
Return an empty claims list if the evidence cannot support an answer.
