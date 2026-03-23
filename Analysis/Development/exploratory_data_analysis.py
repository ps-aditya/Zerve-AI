import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Load & prep ──────────────────────────────────────────────────────────────
_df = pd.read_csv('zerve_hackathon_for_reviewc8fa7c7.csv')
_df['timestamp'] = pd.to_datetime(_df['timestamp'], utc=True, errors='coerce')
_df['date'] = _df['timestamp'].dt.date

# ── Identify meaningful events ───────────────────────────────────────────────
_event_counts = _df['event'].value_counts()

# Key high-value events that signal genuine product use
_success_events = [e for e in _df['event'].unique() if any(kw in str(e).lower() for kw in [
    'agent', 'block_run', 'block_create', 'canvas_create', 'deploy',
    'query', 'script', 'onboarding_tour_finished', 'credit', 'tool'
])]

# ── USER-LEVEL METRICS ───────────────────────────────────────────────────────
_user_col = 'prop_$user_id'
_valid = _df[_df[_user_col].notna() & (_df[_user_col] != '')]

# Total unique users
total_users = _valid[_user_col].nunique()

# Events per user
_events_per_user = _valid.groupby(_user_col).size()
median_events_per_user = int(_events_per_user.median())
p75_events_per_user    = int(_events_per_user.quantile(0.75))
p90_events_per_user    = int(_events_per_user.quantile(0.90))

# Active days per user
_days_per_user = _valid.groupby(_user_col)['date'].nunique()
median_active_days = float(_days_per_user.median())

# ── ACTIVATION: users who completed onboarding ───────────────────────────────
_onboarded = _df[_df['event'] == 'canvas_onboarding_tour_finished']['prop_$user_id'].nunique()
activation_rate = round(_onboarded / max(total_users, 1) * 100, 1)

# ── ENGAGEMENT: agent-usage rate ─────────────────────────────────────────────
_agent_events = _df[_df['event'].str.contains('agent', case=False, na=False)]
_agent_users  = _agent_events['prop_$user_id'].dropna().nunique()
agent_adoption_rate = round(_agent_users / max(total_users, 1) * 100, 1)

# ── RETENTION: users active on 2+ distinct days ──────────────────────────────
_retained_users = (_days_per_user >= 2).sum()
retention_rate = round(_retained_users / max(total_users, 1) * 100, 1)

# ── CREDIT USAGE (depth of use) ──────────────────────────────────────────────
_credits = _df[_df['prop_credits_used'].notna()]['prop_credits_used']
median_credits = round(float(_credits.median()), 2) if len(_credits) else 0.0
total_credit_events = len(_credits)

# ── TOOL USAGE ───────────────────────────────────────────────────────────────
_tools = _df[_df['prop_tool_name'].notna()]['prop_tool_name'].value_counts().head(10)

# ── DAILY ACTIVE USERS trend ─────────────────────────────────────────────────
_dau = _valid.groupby('date')[_user_col].nunique().reset_index()
_dau.columns = ['date', 'dau']
_dau = _dau.sort_values('date').tail(30)  # last 30 days

# ── SUCCESS THRESHOLDS (data-driven) ─────────────────────────────────────────
# A "successful" user = top quartile in engagement
_success_threshold_events = int(_events_per_user.quantile(0.75))
_success_threshold_days   = int(max(_days_per_user.quantile(0.75), 2))
_successful_users = (
    (_events_per_user >= _success_threshold_events) &
    (_days_per_user   >= _success_threshold_days)
).sum()
success_user_rate = round(_successful_users / max(total_users, 1) * 100, 1)

# ── Top events ────────────────────────────────────────────────────────────────
_top_events = _df['event'].value_counts().head(12)

# ════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
_fig = plt.figure(figsize=(20, 22), facecolor='#0f1117')
_gs  = gridspec.GridSpec(4, 3, figure=_fig, hspace=0.55, wspace=0.35,
                          top=0.92, bottom=0.05, left=0.06, right=0.97)

_TITLE_C  = '#e8eaf6'
_BODY_C   = '#b0bec5'
_ACC1     = '#7c4dff'   # purple  – primary
_ACC2     = '#00e5ff'   # cyan    – secondary
_ACC3     = '#69f0ae'   # green   – success
_ACC4     = '#ff6d00'   # orange  – warning
_CARD_BG  = '#1a1d27'

def _kpi_card(ax, value, label, sublabel='', color=_ACC1):
    ax.set_facecolor(_CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, str(value), transform=ax.transAxes,
            ha='center', va='center', fontsize=32, fontweight='bold',
            color=color)
    ax.text(0.5, 0.30, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color=_TITLE_C, fontweight='semibold')
    if sublabel:
        ax.text(0.5, 0.12, sublabel, transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, color=_BODY_C)

# ── Row 0: title ─────────────────────────────────────────────────────────────
_ax_title = _fig.add_subplot(_gs[0, :])
_ax_title.set_facecolor(_CARD_BG)
for sp in _ax_title.spines.values(): sp.set_edgecolor(_ACC1); sp.set_linewidth(2)
_ax_title.set_xticks([]); _ax_title.set_yticks([])
_ax_title.text(0.5, 0.72, '🎯  Definition of Success — Zerve User Analytics',
               transform=_ax_title.transAxes, ha='center', va='center',
               fontsize=19, fontweight='bold', color=_TITLE_C)
_ax_title.text(0.5, 0.30,
    f'Dataset: {len(_df):,} events  ·  {total_users:,} unique users  ·  '
    f'{_df["date"].nunique()} days observed',
    transform=_ax_title.transAxes, ha='center', va='center',
    fontsize=11, color=_BODY_C)

# ── Row 1: KPI cards ─────────────────────────────────────────────────────────
_kpi_card(_fig.add_subplot(_gs[1, 0]),
          f'{activation_rate}%', 'Activation Rate',
          'Users finishing onboarding tour', _ACC3)
_kpi_card(_fig.add_subplot(_gs[1, 1]),
          f'{agent_adoption_rate}%', 'Agent Adoption Rate',
          'Users triggering ≥1 agent event', _ACC2)
_kpi_card(_fig.add_subplot(_gs[1, 2]),
          f'{retention_rate}%', 'Multi-Day Retention',
          'Users active on 2+ distinct days', _ACC4)

# ── Row 2, col 0-1: DAU trend ────────────────────────────────────────────────
_ax_dau = _fig.add_subplot(_gs[2, 0:2])
_ax_dau.set_facecolor(_CARD_BG)
_ax_dau.fill_between(range(len(_dau)), _dau['dau'], alpha=0.25, color=_ACC1)
_ax_dau.plot(range(len(_dau)), _dau['dau'], color=_ACC1, linewidth=2.5, marker='o',
             markersize=4)
_ax_dau.set_xticks(range(0, len(_dau), max(1, len(_dau)//6)))
_ax_dau.set_xticklabels(
    [str(_dau['date'].iloc[i]) for i in range(0, len(_dau), max(1, len(_dau)//6))],
    rotation=30, ha='right', fontsize=8, color=_BODY_C)
_ax_dau.set_title('Daily Active Users (last 30 days)', color=_TITLE_C, fontsize=12,
                   fontweight='bold', pad=8)
_ax_dau.set_ylabel('Unique Users / Day', color=_BODY_C, fontsize=9)
_ax_dau.tick_params(colors=_BODY_C, labelsize=8)
for sp in _ax_dau.spines.values(): sp.set_edgecolor('#333')
_ax_dau.yaxis.label.set_color(_BODY_C)
_ax_dau.grid(axis='y', color='#333', linewidth=0.6)

# ── Row 2, col 2: top events ─────────────────────────────────────────────────
_ax_ev = _fig.add_subplot(_gs[2, 2])
_ax_ev.set_facecolor(_CARD_BG)
_colors_bar = [_ACC1 if 'agent' in e.lower() else
               _ACC3 if 'onboard' in e.lower() else
               _ACC2 for e in _top_events.index]
_ax_ev.barh(range(len(_top_events)), _top_events.values, color=_colors_bar, alpha=0.85)
_ax_ev.set_yticks(range(len(_top_events)))
_ax_ev.set_yticklabels([e.replace('_', ' ')[:28] for e in _top_events.index],
                        fontsize=7.5, color=_BODY_C)
_ax_ev.set_title('Top 12 Events', color=_TITLE_C, fontsize=12,
                  fontweight='bold', pad=8)
_ax_ev.tick_params(colors=_BODY_C, labelsize=8)
for sp in _ax_ev.spines.values(): sp.set_edgecolor('#333')
_ax_ev.xaxis.label.set_color(_BODY_C)
_ax_ev.grid(axis='x', color='#333', linewidth=0.6)

# ── Row 3: success definition scorecard ──────────────────────────────────────
_ax_sc = _fig.add_subplot(_gs[3, :])
_ax_sc.set_facecolor(_CARD_BG)
for sp in _ax_sc.spines.values(): sp.set_edgecolor(_ACC3); sp.set_linewidth(2)
_ax_sc.set_xticks([]); _ax_sc.set_yticks([])

_scorecard = (
    "✅  DEFINITION OF SUCCESS  (data-derived thresholds)\n\n"
    f"  1.  ACTIVATION      → User completes onboarding tour                    Baseline: {activation_rate}% of users today\n"
    f"  2.  ENGAGEMENT      → User triggers ≥1 agent event per session          Baseline: {agent_adoption_rate}% of users today\n"
    f"  3.  RETENTION       → User returns on ≥2 distinct calendar days         Baseline: {retention_rate}% of users today\n"
    f"  4.  DEPTH           → User is in top-quartile: ≥{_success_threshold_events} events & ≥{_success_threshold_days} active days    Baseline: {success_user_rate}% of users today\n"
    f"  5.  CREDIT USAGE    → At least 1 credit-consuming action per session    Baseline: {total_credit_events:,} credit events observed\n\n"
    "  🏆  A user is 'SUCCESSFUL' when they satisfy criteria 1 + 2 + 3 (activation → engagement → retention)."
)

_ax_sc.text(0.02, 0.85, _scorecard, transform=_ax_sc.transAxes,
            ha='left', va='top', fontsize=10.5, color=_TITLE_C,
            fontfamily='monospace', linespacing=1.7)

plt.suptitle('', y=1)
plt.savefig('success_definition.png', dpi=150, bbox_inches='tight',
            facecolor=_fig.get_facecolor())
plt.show()

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 70)
print("DEFINITION OF SUCCESS — KEY METRICS SUMMARY")
print("=" * 70)
print(f"Total unique users observed   : {total_users:,}")
print(f"Activation rate               : {activation_rate}%")
print(f"Agent adoption rate           : {agent_adoption_rate}%")
print(f"Multi-day retention rate      : {retention_rate}%")
print(f"'Power user' threshold        : ≥{_success_threshold_events} events & ≥{_success_threshold_days} active days ({success_user_rate}% of users)")
print(f"Median events per user        : {median_events_per_user}")
print(f"Median active days per user   : {median_active_days}")
print(f"Total credit-consuming events : {total_credit_events:,}")
print("-" * 70)
print("SUCCESS = Activation ✓  +  Agent Engagement ✓  +  Multi-Day Retention ✓")