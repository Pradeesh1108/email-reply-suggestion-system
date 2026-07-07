/**
 * Email Reply Assistant — Frontend Logic
 *
 * Vanilla JS, no build tooling — plain fetch() calls to the FastAPI backend.
 */

// ── Sample emails for quick testing ──────────────────────────────────────────
const SAMPLE_EMAILS = [
    `Subject: Order #5523 hasn't arrived

Hi,

I ordered a wireless keyboard (order #5523) on June 20th and the estimated delivery was June 27th. It's now July 3rd and I still haven't received it. The tracking page just says "in transit" with no updates since June 25th.

Can you look into this? I need it for work.

Thanks,
Rachel Kim`,

    `Subject: Question about your API pricing

Hello,

We're a mid-size SaaS company evaluating your platform. We expect around 2,000 API calls per day. A few questions:

1. What plan would you recommend for this volume?
2. Do you offer annual billing discounts?
3. Is there a sandbox environment we can test with?

Best regards,
James Chen
CTO, CloudSync Inc.`,

    `Subject: Need to reschedule tomorrow's standup

Hey team,

I have a dentist appointment tomorrow morning and won't be able to make the 9:30am standup. Can we push it to 2pm? If that doesn't work, I can send my updates via Slack.

Thanks!
- Pat`,
];

let sampleIndex = 0;

// ── Main suggestion flow ─────────────────────────────────────────────────────

async function generateSuggestion() {
    const emailInput = document.getElementById('emailInput');
    const email = emailInput.value.trim();

    if (!email || email.length < 10) {
        alert('Please enter an email with at least 10 characters.');
        return;
    }

    // Show loading, hide results
    document.getElementById('loadingOverlay').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('suggestBtn').disabled = true;

    try {
        const response = await fetch('/api/suggest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ incoming_email: email }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Server error');
        }

        const data = await response.json();
        renderResults(data);
        loadHistory();
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('suggestBtn').disabled = false;
    }
}

// ── Render results ───────────────────────────────────────────────────────────

function renderResults(data) {
    document.getElementById('resultsSection').style.display = 'block';

    // Reply text
    document.getElementById('replyText').textContent = data.generated_reply.text;

    // Retrieved examples
    renderExamples(data.retrieved_examples, data.evaluation.retrieval_low_confidence);

    // Evaluation
    renderEvaluation(data.evaluation);
}

function renderExamples(examples, lowConfidence) {
    const badge = document.getElementById('confidenceBadge');
    if (lowConfidence) {
        badge.textContent = '⚠️ Low Confidence';
        badge.className = 'badge badge-warning';
    } else {
        badge.textContent = '✓ Good Match';
        badge.className = 'badge badge-success';
    }

    const list = document.getElementById('examplesList');
    list.innerHTML = '';

    examples.forEach((ex, i) => {
        const scoreColor = getScoreColor(ex.similarity_score * 5); // Normalize to 5-point scale for colors
        const card = document.createElement('div');
        card.className = 'example-card';
        card.innerHTML = `
            <div class="example-header">
                <span class="example-label">${ex.category} — ${ex.record_id}</span>
                <span class="similarity-score" style="background: ${scoreColor}15; color: ${scoreColor}">
                    ${(ex.similarity_score * 100).toFixed(1)}% match
                </span>
            </div>
            <div class="example-content">
                <strong>Email:</strong> ${truncate(ex.incoming_email, 150)}<br>
                <strong>Reply:</strong> ${truncate(ex.sent_reply, 150)}
            </div>
        `;
        list.appendChild(card);
    });
}

function renderEvaluation(evaluation) {
    // Composite score
    const compositeEl = document.getElementById('compositeScore');
    compositeEl.textContent = evaluation.composite_score.toFixed(1) + ' / 5';
    compositeEl.style.background = getScoreColor(evaluation.composite_score) + '18';
    compositeEl.style.color = getScoreColor(evaluation.composite_score);

    // Short-circuit alert
    document.getElementById('shortCircuitAlert').style.display =
        evaluation.short_circuited ? 'block' : 'none';

    // Low confidence alert
    document.getElementById('lowConfidenceAlert').style.display =
        evaluation.retrieval_low_confidence ? 'block' : 'none';

    // Rule checks
    renderRuleChecks(evaluation.rule_checks);

    // Judge scores
    const axisSection = document.getElementById('axisScoresSection');
    if (evaluation.judge_scores) {
        axisSection.style.display = 'block';
        renderAxisScores(evaluation.judge_scores);
    } else {
        axisSection.style.display = 'none';
    }

    // Flags
    renderFlags(evaluation.rule_checks);
}

function renderRuleChecks(ruleChecks) {
    const container = document.getElementById('ruleChecks');
    container.innerHTML = '';

    ruleChecks.checks.forEach(check => {
        const item = document.createElement('div');
        item.className = 'rule-check-item';
        const icon = check.passed ? '✅' : (check.hard_failure ? '🛑' : '⚠️');
        item.innerHTML = `
            <span class="rule-check-icon">${icon}</span>
            <span class="rule-check-name">${formatCheckName(check.name)}</span>
            <span class="rule-check-detail">${check.detail}</span>
        `;
        container.appendChild(item);
    });
}

function renderAxisScores(judgeScores) {
    const container = document.getElementById('axisScores');
    container.innerHTML = '';

    judgeScores.axes.forEach(axis => {
        const color = getScoreColor(axis.score);
        const pct = (axis.score / 5) * 100;

        const item = document.createElement('div');
        item.className = 'axis-score-item';
        item.innerHTML = `
            <div class="axis-score-header">
                <span class="axis-name">${formatAxisName(axis.axis)}</span>
                <span class="axis-value" style="color: ${color}">${axis.score}/5</span>
            </div>
            <div class="axis-bar">
                <div class="axis-bar-fill" style="width: ${pct}%; background: ${color}"></div>
            </div>
            <div class="axis-rationale">${axis.rationale}</div>
        `;
        container.appendChild(item);
    });

    // Overall rationale
    document.getElementById('overallRationale').innerHTML =
        `<strong>Overall:</strong> ${judgeScores.overall_rationale}`;
}

function renderFlags(ruleChecks) {
    const section = document.getElementById('flagsSection');
    const list = document.getElementById('flagsList');
    list.innerHTML = '';

    const allFlags = [];
    ruleChecks.checks.forEach(check => {
        check.flagged_items.forEach(item => allFlags.push(item));
    });

    if (allFlags.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    allFlags.forEach(flag => {
        const el = document.createElement('span');
        el.className = 'flag-item';
        el.textContent = flag;
        list.appendChild(el);
    });
}


// ── History ──────────────────────────────────────────────────────────────────

async function loadHistory() {
    try {
        const res = await fetch('/api/history?limit=10');
        const data = await res.json();
        const list = document.getElementById('historyList');

        if (!data || data.length === 0) {
            list.innerHTML = '<p class="text-muted">No suggestions yet.</p>';
            return;
        }

        list.innerHTML = data.map(item => `
            <div class="history-item">
                <span class="history-email">${escapeHtml(item.incoming_email)}</span>
                <span class="history-score" style="color: ${getScoreColor(item.composite_score)}">
                    ${item.composite_score.toFixed(1)}
                </span>
                <span class="history-time">${formatTime(item.timestamp)}</span>
            </div>
        `).join('');
    } catch (e) {
        // Silently fail — history is non-critical
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function loadSampleEmail() {
    const textarea = document.getElementById('emailInput');
    textarea.value = SAMPLE_EMAILS[sampleIndex % SAMPLE_EMAILS.length];
    sampleIndex++;
}

function copyReply() {
    const text = document.getElementById('replyText').textContent;
    navigator.clipboard.writeText(text).then(() => {
        // Brief visual feedback
        const btn = document.querySelector('[onclick="copyReply()"]');
        const original = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => btn.textContent = original, 1500);
    });
}

function getScoreColor(score) {
    if (score >= 4.0) return '#34d399';  // green
    if (score >= 3.0) return '#60a5fa';  // blue
    if (score >= 2.0) return '#fbbf24';  // yellow
    return '#f87171';                     // red
}

function formatCheckName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatAxisName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatCategoryName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function truncate(text, maxLen) {
    const clean = escapeHtml(text.replace(/\n/g, ' '));
    return clean.length > maxLen ? clean.slice(0, maxLen) + '…' : clean;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    } catch {
        return '';
    }
}

// ── Initialize on page load ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});
