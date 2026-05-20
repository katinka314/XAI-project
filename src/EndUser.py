"""
End-user counterfactual explanation for FutureFinance loan decisions.

  Plot 1 — Why were you rejected?    (SHAP waterfall)
  Plot 2 — What can you change?      (top-3 per-feature paths to approval)

Run:  python -m src.counterfactual
"""

import os
import re
import matplotlib
from joblib import Parallel, delayed
matplotlib.use('Agg')  # non-interactive backend — required for joblib threading
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import percentileofscore

from src.data_transformation import (
    categorical_cols, numeric_cols,
    load_adult_data, split_X_y, create_train_test_split,
)
from src.model_factory import model_lr, model_dt, model_MLP, fit_and_score

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# ── feature config ────────────────────────────────────────────────────────────

IMMUTABLE   = {'age', 'race', 'sex', 'native-country', 'fnlwgt',
               'relationship', 'marital-status'}
MUTABLE_NUM = [c for c in numeric_cols    if c not in IMMUTABLE]
MUTABLE_CAT = [c for c in categorical_cols if c not in IMMUTABLE]

APPROVED_LABEL = ' >50K'
OUTPUT_DIR     = 'src/EndUser/'

PRETTY = {
    'education-num':  'Years of Education',
    'capital-gain':   'Capital Gain ($)',
    'capital-loss':   'Capital Loss ($)',
    'hours-pr-week':  'Hours per Week',
    'workclass':      'Employment Type',
    'education':      'Education Level',
    'marital-status': 'Marital Status',
    'occupation':     'Occupation',
}


# ── distance metric ───────────────────────────────────────────────────────────

def _pct_cost(orig, new, col, X_train):
    """Percentile distance between orig and new for a numerical feature (0–100)."""
    vals = X_train[col].dropna().values
    return abs(percentileofscore(vals, new, kind='mean') -
               percentileofscore(vals, orig, kind='mean'))


def _cat_cost(orig, new, col, X_train):
    """Cost of switching category: (1 − population frequency) × 100."""
    if new == orig:
        return 0.0
    freq = X_train[col].value_counts(normalize=True)
    return float(1.0 - freq.get(new, 0.0)) * 100.0


# ── counterfactual finder ─────────────────────────────────────────────────────

def find_top_counterfactuals(sample: pd.Series, model, X_train: pd.DataFrame,
                             n: int = 3, threshold: float = 0.5) -> list:
    """
    For every mutable feature, build one option anchored to that feature:
      1. Pick the best value for the anchor feature (highest prob improvement).
      2. If that alone reaches approval, record as a single-feature option.
      3. If not, greedily add other feature changes until approved.

    Each option is guaranteed to change its anchor feature, ensuring diversity.
    Returns the n cheapest options sorted by total percentile cost.

    Returns list of dicts:
        {'changes': [{'feature', 'orig', 'new', 'cost', 'type'}],
         'total_cost': float, 'cf_sample': pd.Series, 'prob': float}
    """
    approved_idx = list(model.named_steps['clf'].classes_).index(APPROVED_LABEL)

    def _prob(s):
        return model.predict_proba(pd.DataFrame([s]))[0][approved_idx]

    if _prob(sample) >= threshold:
        return []

    def _greedy_fill(s_start, locked_col, visited):
        """Greedily add the best remaining changes until approved or no progress."""
        s = s_start.copy()
        for _ in range(20):
            if _prob(s) >= threshold:
                break
            best_ratio, best_col, best_val, best_type = -np.inf, None, None, None

            for col in MUTABLE_NUM:
                if col == locked_col:
                    continue
                vals        = X_train[col].dropna().values
                current_pct = percentileofscore(vals, s[col], kind='mean')
                for delta_pct in [5, 10, 20, 30]:
                    for direction in [+1, -1]:
                        tgt = max(0.0, float(np.percentile(
                            vals, np.clip(current_pct + direction * delta_pct, 1, 99))))
                        if abs(tgt - s[col]) < 1e-6:
                            continue
                        s_try = s.copy(); s_try[col] = tgt
                        gain  = _prob(s_try) - _prob(s)
                        cost  = _pct_cost(sample[col], tgt, col, X_train)
                        if cost > 0 and gain / cost > best_ratio:
                            best_ratio, best_col, best_val, best_type = gain/cost, col, tgt, 'numerical'

            for col in MUTABLE_CAT:
                if col == locked_col:
                    continue
                for cat_val in X_train[col].dropna().unique():
                    if cat_val == s[col]:
                        continue
                    s_try = s.copy(); s_try[col] = cat_val
                    gain  = _prob(s_try) - _prob(s)
                    cost  = _cat_cost(sample[col], cat_val, col, X_train)
                    if cost > 0 and gain / cost > best_ratio:
                        best_ratio, best_col, best_val, best_type = gain/cost, col, cat_val, 'categorical'

            if best_col is None or best_ratio <= 0:
                break
            s[best_col] = best_val
            visited[best_col] = (best_val, best_type)

        return s, visited

    def _trim(s_approved, visited_approved):
        """Binary-search each numerical change back to the minimum value needed."""
        s = s_approved.copy()
        visited = dict(visited_approved)
        for col in list(visited):
            val, typ = visited[col]
            if typ != 'numerical':
                continue
            orig = float(sample[col])
            curr = float(s[col])
            if abs(curr - orig) < 1e-6:
                continue
            if curr > orig:
                lo, hi = orig, curr
                for _ in range(20):
                    mid = (lo + hi) / 2
                    s_try = s.copy(); s_try[col] = mid
                    if _prob(s_try) >= threshold:
                        hi = mid
                    else:
                        lo = mid
                trimmed = hi
            else:
                lo, hi = curr, orig
                for _ in range(20):
                    mid = (lo + hi) / 2
                    s_try = s.copy(); s_try[col] = mid
                    if _prob(s_try) >= threshold:
                        lo = mid
                    else:
                        hi = mid
                trimmed = lo
            s[col] = trimmed
            visited[col] = (trimmed, typ)
        return s, visited

    def _build_option(anchor_col, anchor_val, anchor_type):
        """Build one option anchored to anchor_col = anchor_val."""
        s = sample.copy()
        s[anchor_col] = anchor_val
        visited = {anchor_col: (anchor_val, anchor_type)}

        if _prob(s) < threshold:
            s, visited = _greedy_fill(s, anchor_col, visited)

        if _prob(s) < threshold:
            return None  # couldn't reach approval even with greedy help

        s, visited = _trim(s, visited)

        changes = []
        for col, (val, typ) in visited.items():
            cost = (_pct_cost(sample[col], val, col, X_train) if typ == 'numerical'
                    else _cat_cost(sample[col], val, col, X_train))
            changes.append({'feature': col, 'orig': sample[col],
                            'new': val, 'cost': cost, 'type': typ})
        changes.sort(key=lambda x: x['cost'], reverse=True)

        return {
            'changes':    changes,
            'total_cost': sum(c['cost'] for c in changes),
            'cf_sample':  s,
            'prob':       _prob(s),
            'anchor':     anchor_col,
        }

    # Collect up to 2 best options per anchor feature for diversity fallback
    pool = []
    for col in MUTABLE_NUM:
        vals = np.sort(X_train[col].dropna().unique())
        if len(vals) <= 20:
            pct_candidates = set(float(v) for v in vals)
        else:
            pct_candidates = set(
                max(0.0, float(np.percentile(vals, p)))
                for p in np.arange(10, 101, 10)
            )
        anchor_candidates = sorted(
            pct_candidates - {float(sample[col])},
            key=lambda v: _pct_cost(sample[col], v, col, X_train),
        )
        col_opts = []
        for anchor_val in anchor_candidates:
            if abs(anchor_val - sample[col]) < 1e-6:
                continue
            opt = _build_option(col, anchor_val, 'numerical')
            if opt is not None:
                col_opts.append(opt)
        col_opts.sort(key=lambda x: x['total_cost'])
        pool.extend(col_opts[:2])

    for col in MUTABLE_CAT:
        col_opts = []
        for cat_val in X_train[col].dropna().unique():
            if cat_val == sample[col]:
                continue
            opt = _build_option(col, cat_val, 'categorical')
            if opt is not None:
                col_opts.append(opt)
        col_opts.sort(key=lambda x: x['total_cost'])
        pool.extend(col_opts[:2])

    # Greedily pick top-n diverse options (no two with identical change-sets)
    pool.sort(key=lambda x: x['total_cost'])
    seen, options = set(), []
    for opt in pool:
        key = frozenset(ch['feature'] for ch in opt['changes'])
        if key not in seen:
            seen.add(key)
            options.append(opt)
            if len(options) == n:
                break

    return options


# ── Plot 1: SHAP waterfall ────────────────────────────────────────────────────

def plot_why_rejected(sample: pd.Series, model, X_train: pd.DataFrame,
                      output_path: str = OUTPUT_DIR + 'why_rejected.png'):
    """SHAP waterfall for a single rejected applicant."""
    X_t = model.named_steps['prep'].transform(X_train)
    if hasattr(X_t, 'toarray'):
        X_t = X_t.toarray()

    sample_t = model.named_steps['prep'].transform(pd.DataFrame([sample]))
    if hasattr(sample_t, 'toarray'):
        sample_t = sample_t.toarray()

    feature_names = model.named_steps['prep'].get_feature_names_out()

    clf = model.named_steps['clf']
    try:
        explainer = shap.Explainer(clf, X_t)
    except TypeError:
        # MLP and other models SHAP can't analyse natively — use permutation explainer
        masker    = shap.maskers.Independent(X_t, max_samples=50)
        explainer = shap.Explainer(clf.predict_proba, masker)
    sv = explainer(sample_t)

    # Unstandardize numeric display values (replace z-scores with originals).
    display_data = sv.data[0].copy()
    for i, fname in enumerate(feature_names):
        if fname.startswith('num__'):
            col = fname[5:]
            if col in sample.index:
                display_data[i] = sample[col]

    approved_idx = list(model.named_steps['clf'].classes_).index(APPROVED_LABEL)
    if sv.values.ndim == 3:
        raw_shap = sv.values[0, :, approved_idx]
        base_val = sv.base_values[0, approved_idx]
    else:
        raw_shap = sv.values[0]
        base_val = float(np.atleast_1d(sv.base_values)[0])

    # Collapse OHE columns: one row per category, summed SHAP, active value shown.
    c_shap, c_data, c_names = _collapse_ohe(feature_names, raw_shap, display_data, sample)
    explanation = shap.Explanation(
        values       = c_shap,
        base_values  = base_val,
        data         = c_data,
        feature_names= c_names,
    )

    plt.figure(figsize=(5.7, 7.2))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title('Why was your loan application rejected?', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ── Plot 0: Score bar ────────────────────────────────────────────────────────

def plot_score_bar(prob: float,
                   output_path: str = OUTPUT_DIR + 'score_bar.png'):
    """Slim horizontal bar showing the applicant's approval score vs 50% threshold."""
    fig, ax = plt.subplots(figsize=(11.4, 1.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Red / green zones
    ax.add_patch(Rectangle((0.02, 0.10), 0.48, 0.65, facecolor='#ffd6d6', edgecolor='none'))
    ax.add_patch(Rectangle((0.50, 0.10), 0.48, 0.65, facecolor='#d6f5d6', edgecolor='none'))
    # Border
    ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.65,
                            facecolor='none', edgecolor='#aaaaaa', linewidth=0.8))

    # 50 % threshold line
    ax.plot([0.50, 0.50], [0.05, 0.82], color='#333333', linewidth=1.8, zorder=4)
    ax.text(0.50, 0.88, '50 % — approval threshold',
            ha='center', va='bottom', fontsize=8, color='#333333')

    # Applicant dot
    score_x = 0.02 + prob * 0.96
    dot_color = '#27ae60' if prob >= 0.5 else '#e74c3c'
    ax.plot(score_x, 0.425, 'o', markersize=11, color=dot_color, zorder=5)
    ax.text(score_x, 0.04, f'{prob:.0%}',
            ha='center', va='top', fontsize=8.5, fontweight='bold', color=dot_color)

    # End labels
    ax.text(0.025, 0.425, '0 %', ha='left', va='center', fontsize=7.5, color='#888')
    ax.text(0.975, 0.425, '100 %', ha='right', va='center', fontsize=7.5, color='#888')

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ── Plot 2: Top-3 options ─────────────────────────────────────────────────────

def plot_what_to_change(options: list,
                        output_path: str = OUTPUT_DIR + 'what_to_change.png'):
    """Option cards — one card per counterfactual, side by side."""
    if not options:
        print('No changes required — applicant is already approved.')
        return

    def _difficulty(cost):
        if cost < 60:  return 'Easy',     '#27ae60'
        if cost < 100: return 'Moderate', '#e67e22'
        return             'Hard',     '#e74c3c'

    def _fmt(ch):
        col = ch['feature']
        label = PRETTY.get(col, col.replace('-', ' ').title())
        if ch['type'] == 'numerical':
            orig, new = ch['orig'], ch['new']
            if col in ('capital-gain', 'capital-loss'):
                return label, f"${orig:,.0f}", f"${round(new / 100) * 100:,.0f}"
            if col == 'education-num':
                return label, f"{int(orig)} yrs", f"{round(new)} yrs"
            return label, f"{orig:.0f}", f"{round(new)}"
        return label, str(ch['orig']).strip(), str(ch['new']).strip()

    n_slots = 3
    fig, axes = plt.subplots(n_slots, 1, figsize=(5.7, 7.2), dpi=180)

    for rank, ax in enumerate(axes):
        if rank < len(options):
            opt = options[rank]
            diff_label, diff_color = _difficulty(opt['total_cost'])

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#fafafa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(1.5)

            ax.add_patch(Rectangle((0, 0.78), 1, 0.22,
                                    facecolor=diff_color, edgecolor='none'))
            ax.text(0.5, 0.89, f'Option {rank + 1}',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')

            y = 0.70
            for ch in opt['changes']:
                label, orig_str, new_str = _fmt(ch)
                ax.text(0.10, y, label, ha='left', va='center',
                        fontsize=10, fontweight='bold', color='#2c3e50')
                ax.text(0.10, y - 0.085,
                        f"{orig_str}   →   {new_str}",
                        ha='left', va='center', fontsize=10.5, color='#555555')
                ax.plot([0.08, 0.92], [y - 0.13, y - 0.13],
                        color='#e0e0e0', linewidth=0.7)
                y -= 0.18
        else:
            # Greyed-out placeholder
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#f5f5f5')
            for spine in ax.spines.values():
                spine.set_edgecolor('#dddddd')
                spine.set_linewidth(1.0)
            ax.add_patch(Rectangle((0, 0.78), 1, 0.22,
                                    facecolor='#cccccc', edgecolor='none'))
            ax.text(0.5, 0.88, f'Option {rank + 1}',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')
            ax.text(0.5, 0.50, 'No alternative path found',
                    ha='center', va='center', fontsize=10, color='#aaaaaa')

    fig.suptitle('What can you change to get approved?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _pretty(name: str) -> str:
    name = re.sub(r'^(cat|num)__', '', name)
    name = name.replace('_infrequent_sklearn', ' (rare)').replace('_', ' ').strip()
    return PRETTY.get(name, name.replace('-', ' ').title())


def _collapse_ohe(feature_names, shap_vals, display_data, sample=None):
    """Collapse one-hot cat__ features into one row per category.

    Sums SHAP values across all OHE columns and shows the active category
    value read directly from `sample` (avoids CSV leading-space mismatches).
    Falls back to scanning for the == 1 column when sample is unavailable.
    Numeric features pass through unchanged.
    """
    seen, out = {}, []

    for i, fname in enumerate(feature_names):
        if fname.startswith('cat__'):
            rest     = fname[5:]
            sep      = rest.index('_')
            col_name = rest[:sep]

            if col_name not in seen:
                seen[col_name] = len(out)
                pretty_col = PRETTY.get(col_name, col_name.replace('-', ' ').title())
                # Read category value straight from the original row.
                if sample is not None and col_name in sample.index:
                    raw = str(sample[col_name]).strip()
                    val = '(Rare)' if 'infrequent' in raw.lower() else \
                          raw.replace('-', ' ').replace('_', ' ').title()
                else:
                    val = None  # filled in below by OHE scan
                out.append({'name': pretty_col, 'shap': 0.0, 'val': val})

            idx = seen[col_name]
            out[idx]['shap'] += float(shap_vals[i])

            # OHE fallback: only needed when sample wasn't provided.
            if out[idx]['val'] is None and float(display_data[i]) == 1.0:
                raw_val = rest[sep + 1:]
                out[idx]['val'] = '(Rare)' if 'infrequent_sklearn' in raw_val else \
                                  raw_val.replace('-', ' ').replace('_', ' ').title()
        else:
            col = fname[5:] if fname.startswith('num__') else fname
            pretty_col = PRETTY.get(col, col.replace('-', ' ').title())
            out.append({'name': pretty_col, 'shap': float(shap_vals[i]), 'val': display_data[i]})

    for entry in out:
        if entry['val'] is None:
            entry['val'] = ''

    c_shap  = np.array([e['shap'] for e in out], dtype=float)
    c_data  = np.array([e['val']  for e in out], dtype=object)
    c_names = [e['name'] for e in out]
    return c_shap, c_data, c_names


# ── worker (module-level so joblib can reference it) ─────────────────────────

def _run_job(model_name, model, sample, X_train, user_num, out_dir):
    """Search counterfactuals and save all three plots for one (model, user) pair."""
    print(f'\n[{model_name} / User {user_num}] starting...')
    print(f'User ID: {user_num}  (dataset row {sample.name + 1})')
    print(sample.to_string())
    approved_idx = list(model.named_steps['clf'].classes_).index(APPROVED_LABEL)
    prob = float(model.predict_proba(pd.DataFrame([sample]))[0][approved_idx])
    plot_score_bar(prob, output_path=out_dir + f'score_bar_user{user_num}.png')
    plot_why_rejected(sample, model, X_train,
                      output_path=out_dir + f'why_rejected_user{user_num}.png')
    options = find_top_counterfactuals(sample, model, X_train, n=3)
    plot_what_to_change(options,
                        output_path=out_dir + f'what_to_change_user{user_num}.png')
    print(f'[{model_name} / User {user_num}] done  ({len(options)} options found)')
    return model_name, user_num, options


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data = load_adult_data(data_path='data/adult.data', nrows=None)
    X, y = split_X_y(data)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)


    # Exclude sensitive attributes from training (fairness)
    SENSITIVE = {'sex', 'race', 'relationship'}
    cat_cols_fair = [c for c in categorical_cols if c not in SENSITIVE]
    num_cols_fair = [c for c in numeric_cols    if c not in SENSITIVE]

    MODELS = [
        ('DT',  model_dt(cat_cols_fair, num_cols_fair)),
        ('MLP', model_MLP(cat_cols_fair, num_cols_fair)),
    ]

    PROFILE_FILTERS = [
        # User 1: low-hours private worker, HS grad
        lambda Xt, rej, pr: (
            rej & (pr < 0.35) & (Xt['hours-pr-week'] < 40)
            & (Xt['workclass'].str.strip() == 'Private')
            & (Xt['education'].str.strip() == 'HS-grad')
        )
    ]

    # Fit all models, then build job list
    jobs = []
    for model_name, model in MODELS:
        print(f'Fitting {model_name}...')
        fit_and_score(model, X_train, y_train, X_test, y_test)
        out_dir = OUTPUT_DIR + model_name + '/'
        os.makedirs(out_dir, exist_ok=True)
        approved_idx = list(model.named_steps['clf'].classes_).index(APPROVED_LABEL)
        probs    = model.predict_proba(X_test)[:, approved_idx]
        rejected = model.predict(X_test) != APPROVED_LABEL
        for user_num, pf in enumerate(PROFILE_FILTERS, start=1):
            mask = pf(X_test, rejected, probs)
            pool = X_test[mask]
            if pool.empty:
                print(f'  User {user_num}: no matching sample, skipping.')
                continue
            jobs.append((model_name, model, pool.iloc[0], X_train, user_num, out_dir))

    # Run all (model, user) jobs in parallel
    print(f'\nRunning {len(jobs)} jobs in parallel...')
    Parallel(n_jobs=3, backend='loky')(
        delayed(_run_job)(*job) for job in jobs
    )