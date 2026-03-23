
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 / 5 / 6 SIGNAL PULL
# Pull key metrics from upstream analysis blocks (retention_distribution_1,
# activation_threshold_discovery, feature_impact_simulation)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 4: Retention signal lifts (top10_retention_signals)
# event_name → retention_lift (vs. baseline), total_frequency = user_reach proxy
_signals = top10_retention_signals[['event_name', 'retention_lift', 'total_frequency']].copy()
_signals.columns = ['event_name', 'retention_lift', 'user_reach']

# ── Step 5: Activation threshold insights (event_bucket_df, etype_bucket_df)
_low_event_users = int(event_bucket_df.loc[event_bucket_df['bucket'] == '0–10', 'users'].values[0])  # 3666

# Unique event-type threshold: users with 10+ types retain at 6.94% vs 0.96% for 1-3
_etype_max_rate = etype_bucket_df['retention_rate'].max()   # 6.94% at 10+ types
_etype_min_rate = etype_bucket_df['retention_rate'].min()   # 0.96% at 1-3 types
_etype_low_users = int(etype_bucket_df.loc[etype_bucket_df['bucket'] == '1–3', 'users'].values[0])  # 3021

# ── Step 5: Activation funnel — sign-in → canvas_create drop-off
_funnel_canvas_drop = int(activation_funnel.loc[activation_funnel['stage'] == 'canvas_create', 'drop_off_users'].values[0])  # 787
_funnel_canvas_cvr = float(activation_funnel.loc[activation_funnel['stage'] == 'canvas_create', 'conversion_rate'].values[0])  # 31.3

# ── Step 6: Feature impact simulation
_sim_agent = float(feature_impact_simulation.loc[feature_impact_simulation['scenario'] == 'C', 'retention_lift'].values[0])   # 2.37
_sim_canvas = float(feature_impact_simulation.loc[feature_impact_simulation['scenario'] == 'B', 'retention_lift'].values[0])  # 1.50

# ── AI interaction signal: top retention signal
_ai_open_lift = float(_signals.loc[_signals['event_name'] == 'agent_open', 'retention_lift'].values[0])   # 7.13
_ai_open_reach = int(_signals.loc[_signals['event_name'] == 'agent_open', 'user_reach'].values[0])         # 244
_ai_accept_lift = float(_signals.loc[_signals['event_name'] == 'agent_accept_suggestion', 'retention_lift'].values[0])  # 4.10
_ai_accept_reach = int(_signals.loc[_signals['event_name'] == 'agent_accept_suggestion', 'user_reach'].values[0])       # 305

# ── Summary stats from load_dataset (hardcoded from canvas metadata for reference)
_activation_rate = 10.3       # % of users who complete onboarding
_agent_adoption_rate = 52.9   # % of users who have used the agent
_baseline_retention_pct = retention_rate_pct  # 0.75% from retention_distribution_1

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE IMPACT SCORES: impact_score = retention_lift × user_reach
# ─────────────────────────────────────────────────────────────────────────────

_candidates = {
    # ── ONBOARDING: push users past 10-event activation threshold
    # 3,666 users at <10 events with 0.98% retention; 120+ events = 7.96% (8x lift)
    'Onboarding: Push Users Past 10-Event Activation': {
        'area': 'Onboarding',
        'retention_lift': round(
            event_bucket_df.loc[event_bucket_df['bucket'] == '120+', 'retention_rate'].values[0]
            - event_bucket_df.loc[event_bucket_df['bucket'] == '0–10', 'retention_rate'].values[0], 2),
        'user_reach': _low_event_users,
        'reasoning': (
            f"3,666 users (76.8% of cohort) are stuck at <10 events with only 0.98% retention. "
            f"Users with 120+ events retain at 7.96% — an 8x lift. Guided onboarding nudges "
            f"(checklists, progress indicators, tooltips) can move users past this critical threshold."
        ),
    },
    # ── FEATURE DISCOVERY: event-type breadth predicts retention (6.94% vs 0.96%)
    'Feature Discovery: Increase Event-Type Breadth to 10+': {
        'area': 'Feature Discovery',
        'retention_lift': round(_etype_max_rate - _etype_min_rate, 2),
        'user_reach': _etype_low_users,
        'reasoning': (
            f"Users exploring only 1–3 feature types retain at 0.96% (3,021 users). "
            f"Those reaching 10+ types retain at {_etype_max_rate}% — a 7x lift. "
            f"'What's new' highlights, contextual suggestions, and feature discovery tours "
            f"can drive breadth exploration and unlock higher-retention user behaviour."
        ),
    },
    # ── AI INTERACTION: agent_open is the #1 retention signal (7.13x lift)
    # Simulation C: +25% agent interaction → +2.37pp retention
    'AI Interaction: Promote Agent Usage (agent_open)': {
        'area': 'AI Interaction',
        'retention_lift': _ai_open_lift,
        'user_reach': _ai_open_reach,
        'reasoning': (
            f"agent_open is the strongest retention signal at {_ai_open_lift:.2f}x lift. "
            f"Simulation C confirms: +25% agent interaction → +{_sim_agent}pp retention. "
            f"Default AI panels, first-run AI demos, and prominent agent entry points "
            f"would accelerate AI adoption among the {100 - _agent_adoption_rate:.1f}% of users yet to discover it."
        ),
    },
    # ── CANVAS CREATION: 68.7% funnel drop-off; 4.67x retention lift
    # Simulation B: +30% canvas creation → +1.50pp retention
    'Onboarding: Reduce Canvas Creation Drop-off (68.7%)': {
        'area': 'Onboarding',
        'retention_lift': float(_signals.loc[_signals['event_name'] == 'canvas_create', 'retention_lift'].values[0]),
        'user_reach': _funnel_canvas_drop,
        'reasoning': (
            f"68.7% of sign-in users ({_funnel_canvas_drop} users) never create a canvas — "
            f"the critical first activation step. canvas_create has a "
            f"{float(_signals.loc[_signals['event_name'] == 'canvas_create', 'retention_lift'].values[0]):.2f}x retention lift. "
            f"Simulation B: +30% canvas creation → +{_sim_canvas}pp retention. "
            f"Templates, guided wizards, and pre-built examples can cut this drop-off significantly."
        ),
    },
    # ── AI ACCEPT: agent_accept_suggestion — 4.10x lift, deepens AI engagement
    'AI Interaction: Drive Agent Suggestion Acceptance': {
        'area': 'AI Interaction',
        'retention_lift': _ai_accept_lift,
        'user_reach': _ai_accept_reach,
        'reasoning': (
            f"agent_accept_suggestion carries a {_ai_accept_lift:.2f}x retention lift across "
            f"{_ai_accept_reach} users. Users who accept AI suggestions are the highest-retention "
            f"archetype. Improving suggestion quality, relevance ranking, and acceptance UX "
            f"converts passive AI browsers into active collaborators."
        ),
    },
    # ── WORKFLOW COMPLETION: run_block is #3 retention signal (4.42x lift)
    'Workflow Completion: Reduce run_block Friction': {
        'area': 'Workflow Completion',
        'retention_lift': float(_signals.loc[_signals['event_name'] == 'run_block', 'retention_lift'].values[0]),
        'user_reach': int(_signals.loc[_signals['event_name'] == 'run_block', 'user_reach'].values[0]),
        'reasoning': (
            f"run_block has a {float(_signals.loc[_signals['event_name'] == 'run_block', 'retention_lift'].values[0]):.2f}x retention lift. "
            f"Top retained workflow patterns all centre on run_block sequences. "
            f"Reducing friction (faster execution, clearer errors, one-click re-run) "
            f"turns explorers into committed workflow builders — driving the core value loop."
        ),
    },
}

# ── Compute impact_score = retention_lift × user_reach
for _key, _rec in _candidates.items():
    _rec['impact_score'] = round(_rec['retention_lift'] * _rec['user_reach'], 1)

# ── Build DataFrame and rank by impact_score descending
_rec_rows = []
for _title, _rec in _candidates.items():
    _rec_rows.append({
        'recommendation': _title,
        'area': _rec['area'],
        'reasoning': _rec['reasoning'],
        'expected_retention_lift': _rec['retention_lift'],
        'user_reach': _rec['user_reach'],
        'impact_score': _rec['impact_score'],
    })

_rec_df = pd.DataFrame(_rec_rows).sort_values('impact_score', ascending=False).reset_index(drop=True)
_rec_df.index = _rec_df.index + 1  # 1-based rank

# ── Top 5 only — required output
top5_recommendations = _rec_df.head(5).copy()

# ── Required column output: recommendation | reasoning | expected_retention_lift | impact_score
_display_df = top5_recommendations[['recommendation', 'reasoning', 'expected_retention_lift', 'impact_score']].copy()
_display_df.index.name = 'rank'

# ─────────────────────────────────────────────────────────────────────────────
# PRINT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 90)
print("  TOP 5 DATA-DRIVEN PRODUCT RECOMMENDATIONS")
print("  impact_score = expected_retention_lift × user_reach")
print("=" * 90)

for _rank, _row in _display_df.iterrows():
    print(f"\n{'─'*90}")
    print(f"  #{_rank}  {_row['recommendation']}")
    print(f"{'─'*90}")
    # Word-wrap reasoning at 85 chars
    _words = _row['reasoning'].split()
    _line, _wrapped = '', []
    for _w in _words:
        if len(_line) + len(_w) + 1 > 85:
            _wrapped.append(_line.strip())
            _line = _w + ' '
        else:
            _line += _w + ' '
    if _line:
        _wrapped.append(_line.strip())
    for _wl in _wrapped:
        print(f"  {_wl}")
    print(f"\n  Expected Retention Lift : {_row['expected_retention_lift']:.2f}x")
    print(f"  Impact Score            : {_row['impact_score']:,.0f}")

print(f"\n{'=' * 90}")
print("\nSummary table (rank | recommendation | retention_lift | impact_score):")
_summary = top5_recommendations[['recommendation', 'area', 'expected_retention_lift', 'user_reach', 'impact_score']].copy()
_summary.index = range(1, len(_summary) + 1)
_summary.index.name = 'rank'
print(_summary.to_string())

print("\n\nKey metric context:")
print(f"  • Baseline retention rate          : {_baseline_retention_pct:.2f}%  ({retention_retained} / {retention_total_users} users)")
print(f"  • Activation rate (onboarded)      : {_activation_rate}%")
print(f"  • Agent adoption rate              : {_agent_adoption_rate}%")
print(f"  • Canvas creation funnel drop-off  : {100 - _funnel_canvas_cvr:.1f}%  ({_funnel_canvas_drop} users lost)")
print(f"  • Users below activation threshold : {_low_event_users:,}  (0–10 events, 0.98% retention)")
print(f"  • Highest-impact area              : {top5_recommendations.iloc[0]['area']}  ({top5_recommendations.iloc[0]['recommendation']})")
