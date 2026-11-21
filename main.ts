// GTA VI trophy rarity forecast — trend model (Node 22, standard library only).
//
// For every trophy of Grand Theft Auto VI on PlayStation 5, forecast the earn
// rate Exophase prints at the case's target instant, as five quantiles at the
// levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
//
// Model: instead of holding each trophy's last rate (the house baseline), fit a
// straight line to logit(rate) against log(1 + hours since launch) over the
// case's reads, by least squares, and read the line at the target, shrunk
// 85% trend / 15% last observed rate so a long extrapolation cannot run away
// from the most recent reading. Earn rates
// move roughly log-linearly in time after launch — early trophies climb fast
// then flatten as the player base saturates — so a line in log-time tracks the
// climb while the baseline stands still. The spread is the random-walk spread
// (0.5 logit units per sqrt day) combined in quadrature with the fit's
// residual scatter, so a trophy whose history wobbles gets a wider band than
// one that rides its line. With fewer than 3 usable reads there is no line to
// fit, so the forecast falls back to the last rate with the random-walk spread.
//
// Rates are clamped to 0.01..99.99 percent before the logit (the scorer clamps
// the same way), quantiles are rounded to 4 decimals, and the program is fully
// deterministic: no randomness anywhere, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Launch instant: the origin of the log-time axis. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Scored quantile levels, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Random-walk spread, in logit units per square-root day. */
const WALK_SIGMA = 0.5;
/** Minimum reads with a usable rate before a line is fit. */
const MIN_READS_FOR_TREND = 3;
/** Weight of the trend line in the shrunk centre (the rest is the last rate). */
const TREND_WEIGHT = 0.85;

interface TrophySpec {
  id: number;
}

interface Read {
  at: string;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophySpec[];
  history: Read[];
}

/** Clamp a percent into the range the logit is defined on. */
function clampPct(pct: number): number {
  if (!Number.isFinite(pct)) return 0.01;
  return Math.min(99.99, Math.max(0.01, pct));
}

/** logit of a percent: ln(p / (1 - p)) with p = pct / 100. */
function logitPct(pct: number): number {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
}

/** Inverse logit back to a percent. */
function invLogitPct(eta: number): number {
  if (eta >= 0) {
    const e = Math.exp(-eta);
    return 100 / (1 + e);
  }
  const e = Math.exp(eta);
  return (100 * e) / (1 + e);
}

function round4(x: number): number {
  return Math.round(x * 10_000) / 10_000;
}

/** Log-time coordinate of an instant: ln(1 + hours since launch). */
function logTime(iso: string): number {
  const hours = (Date.parse(iso) - LAUNCH_MS) / 3_600_000;
  // Reads never start more than an hour before launch, but guard the log.
  return Math.log1p(Math.max(hours, -0.999999));
}

/** Least-squares line through (x, y) points; null when the x's do not vary. */
function fitLine(points: Array<{ x: number; y: number }>): { intercept: number; slope: number } | null {
  const n = points.length;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  const xBar = sx / n;
  const yBar = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    const dx = p.x - xBar;
    sxx += dx * dx;
    sxy += dx * (p.y - yBar);
  }
  if (!(sxx > 0) || !Number.isFinite(sxx)) return null;
  const slope = sxy / sxx;
  return { intercept: yBar - slope * xBar, slope };
}

/**
 * Forecast one trophy: [median logit at target, total spread in logit units].
 * Fits the trend when there are enough usable reads, else holds the last rate.
 */
function forecastTrophy(
  caseFile: CaseFile,
  trophyIndex: number,
  targetX: number,
  walkSpread: number,
): { centre: number; sigma: number } {
  const points: Array<{ x: number; y: number }> = [];
  let lastRate: number | null = null;
  for (const read of caseFile.history) {
    const rate = read.rates[trophyIndex];
    if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
    points.push({ x: logTime(read.at), y: logitPct(rate) });
    lastRate = rate;
  }

  const fallbackCentre = logitPct(lastRate ?? 0.01);
  if (points.length < MIN_READS_FOR_TREND) {
    return { centre: fallbackCentre, sigma: walkSpread };
  }

  const line = fitLine(points);
  if (line === null) {
    return { centre: fallbackCentre, sigma: walkSpread };
  }

  // Residual scatter of the fit (unbiased, n - 2 degrees of freedom).
  let rss = 0;
  for (const p of points) {
    const r = p.y - (line.intercept + line.slope * p.x);
    rss += r * r;
  }
  const residualVar = points.length > 2 ? rss / (points.length - 2) : 0;

  // Shrink the trend centre toward the last observed rate: the trend line is
  // extrapolated 6.5 days past the last read, so blending in 15% of the last
  // rate guards against over-extrapolation when the climb has already flattened.
  const trendCentre = line.intercept + line.slope * targetX;
  const centre = TREND_WEIGHT * trendCentre + (1 - TREND_WEIGHT) * fallbackCentre;

  return {
    centre,
    sigma: Math.sqrt(walkSpread * walkSpread + residualVar),
  };
}

function forecastCase(caseFile: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  if (
    caseFile.levels.length !== LEVELS.length ||
    caseFile.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `case ${caseFile.id} asks for levels ${JSON.stringify(caseFile.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (caseFile.history.length === 0) throw new Error(`case ${caseFile.id} has no history`);

  const days = caseFile.horizon_minutes / 1440;
  const walkSpread = WALK_SIGMA * Math.sqrt(days);
  const targetX = logTime(caseFile.target);

  const trophies = caseFile.trophies.map((trophy, i) => {
    const { centre, sigma } = forecastTrophy(caseFile, i, targetX, walkSpread);
    const q = Z_SCORES.map((z) => {
      const value = round4(invLogitPct(centre + z * sigma));
      return Math.min(100, Math.max(0, value));
    });
    // Rounding can only tie neighbours, never invert them, but enforce order.
    for (let j = 1; j < q.length; j++) {
      if (q[j] < q[j - 1]) q[j] = q[j - 1];
    }
    return { id: trophy.id, q };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });

// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const caseFile = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const forecast = forecastCase(caseFile);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecast)}\n`);
}
