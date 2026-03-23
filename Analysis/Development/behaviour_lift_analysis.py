# ============================================
# FIXED STEP 2.4 — BEHAVIOUR LIFT
# ============================================

successful = comparison.loc[1]
unsuccessful = comparison.loc[0]

lift = ((successful - unsuccessful) / unsuccessful) * 100

print("Behaviour lift (%) of successful users vs non-successful:")
print(lift)