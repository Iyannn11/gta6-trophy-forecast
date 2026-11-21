// GTA VI trophy rarity forecast — Round 2 model.
//
// Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
// Exophase earn rate (%) at the case target (2026-11-26T00:00:00Z for the
// final), from the house's reads of the trophy page before the checkpoint.
//
// Model, per trophy (all on the logit scale, logit(p) = ln(p / (1 - p))):
//   1. Recency-weighted least-squares line of logit(rate) against
//      ln(1 + hours since 2026-11-19T00:00:00Z), over every non-null read.
//      Recent reads weigh more (linear ramp 0.5 -> 1.0 across the history):
//      the earn-rate curve steepens as launch approaches, so stale reads
//      from launch night should not anchor the trend.
//   2. Adaptive trend weight: the trend centre is blended with the last
//      read's rate, with the blend decided by the fit's strength.
//      w = 0.5 + 0.45 * R^2  (R^2 = 0 -> 50/50, R^2 = 1 -> 95% trend).
//      A strong, clean trend is trusted; a noisy one is shrunk hard toward
//      the last observed rate.
//   3. Late-surge slope: when the last 48 h hold >= 5 reads spanning >= 24 h,
//      the slope over that window is blended in (75% full-history slope,
//      25% late-window slope), so an acceleration in the final days is
//      partly captured without letting a short window dominate.
//   4. Per-trophy spread: the random-walk term 0.5 * sqrt(days to target)
//      combined in quadrature with the fit residual, floored at 0.20 and
//      capped at 2.00 logit units, so flat histories keep a sane band and
//      noisy trophies widen instead of exploding. (A horizon-scaled
//      residual was tested and rejected: on synthetic noisy histories it
//      widened the band too far and lost to the unscaled residual.)
//   5. Robustness: fewer than 3 usable reads -> fall back to the last rate
//      with the house random-walk spread; null rates skipped; rates clamped
//      to 0.01..99.99 before the logit; forecasts clamped to 0.01..99.99 and
//      rounded to 4 decimals; fully deterministic, no clock, no randomness,
//      no network.
//
// Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
// (else /output), one <id>.out per <id>.in found, in name order.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Standard-normal quantiles at the five levels.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// Earn-rate curve time base: GTA VI launch, 2026-11-19T00:00:00Z.
const T0 = Date.parse("2026-11-19T00:00:00Z");

const CLAMP_LO = 0.01;
const CLAMP_HI = 99.99;

function clampRate(r: number): number {
  if (!Number.isFinite(r)) return CLAMP_LO;
  return Math.min(CLAMP_HI, Math.max(CLAMP_LO, r));
}

function logit(ratePct: number): number {
  const p = clampRate(ratePct) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  if (l > 20) return CLAMP_HI;
  if (l < -20) return CLAMP_LO;
  const p = 1 / (1 + Math.exp(-l));
  return clampRate(p * 100);
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

interface Read {
  tHours: number; // hours since T0
  y: number; // logit(rate)
}

// Weighted least squares of y on x. Returns slope, intercept, R^2 and the
// weighted residual RMS, or null when the fit is degenerate.
function wls(
  pts: { x: number; y: number; w: number }[],
): { slope: number; intercept: number; r2: number; resid: number } | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of pts) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  if (sw <= 0) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of pts) {
    const dx = p.x - mx;
    const dy = p.y - my;
    sxx += p.w * dx * dx;
    sxy += p.w * dx * dy;
    syy += p.w * dy * dy;
  }
  if (sxx <= 1e-12) return null; // all reads at (nearly) one time point
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  // Weighted R^2 (0 when there is no variance to explain).
  const ssTot = syy;
  let ssRes = 0;
  for (const p of pts) {
    const e = p.y - (intercept + slope * p.x);
    ssRes += p.w * e * e;
  }
  const r2 = ssTot <= 1e-12 ? 0 : Math.max(0, 1 - ssRes / ssTot);
  const resid = Math.sqrt(ssRes / sw);
  return { slope, intercept, r2, resid };
}

function forecastTrophy(
  reads: Read[],
  lastRate: number,
  horizonDays: number,
  targetHours: number,
): number[] {
  const lastLogit = logit(lastRate);
  const rw = 0.5 * Math.sqrt(Math.max(0, horizonDays));

  // Fallback: too few reads to fit a line.
  if (reads.length < 3) {
    const sigma = Math.max(0.2, rw);
    return Z.map((z) => round4(invLogit(lastLogit + z * sigma)));
  }

  const tMin = reads[0].tHours;
  const tMax = reads[reads.length - 1].tHours;
  const spanHours = Math.max(0, tMax - tMin);

  // Recency weights: 0.5 for the oldest read, 1.0 for the newest.
  const pts = reads.map((r) => {
    const frac = spanHours > 1e-9 ? (r.tHours - tMin) / spanHours : 1;
    return { x: Math.log1p(Math.max(0, r.tHours)), y: r.y, w: 0.5 + 0.5 * frac };
  });

  const fit = wls(pts);
  if (fit === null) {
    const sigma = Math.max(0.2, rw);
    return Z.map((z) => round4(invLogit(lastLogit + z * sigma)));
  }

  // Late-surge slope: a second fit over the last 48 h of reads, blended in
  // only when that window is well populated.
  let slope = fit.slope;
  const lateCut = tMax - 48;
  const latePts = pts.filter((p, i) => reads[i].tHours >= lateCut);
  if (latePts.length >= 5) {
    const lateTimes = reads.filter((r) => r.tHours >= lateCut);
    const lateSpan = lateTimes[lateTimes.length - 1].tHours - lateTimes[0].tHours;
    if (lateSpan >= 24) {
      const lateFit = wls(latePts);
      if (lateFit !== null) {
        slope = 0.75 * fit.slope + 0.25 * lateFit.slope;
      }
    }
  }

  const xTarget = Math.log1p(Math.max(0, targetHours));
  const trendCentre = fit.intercept + slope * xTarget;

  // Adaptive trend weight from the fit's strength.
  const wTrend = 0.5 + 0.45 * Math.min(1, Math.max(0, fit.r2));
  const centre = wTrend * trendCentre + (1 - wTrend) * lastLogit;

  // Per-trophy spread: random walk combined in quadrature with the fit
  // residual. The residual is NOT scaled by the horizon: a horizon-scaled
  // variant was built and tested on synthetic cases with known truth and
  // lost to the unscaled residual (overall pinball 0.2502 vs 0.2404),
  // because it widened the band too far on noisy histories.
  const sigmaRaw = Math.sqrt(rw * rw + fit.resid * fit.resid);
  const sigma = Math.min(2.0, Math.max(0.2, sigmaRaw));

  const qs = Z.map((z) => round4(invLogit(centre + z * sigma)));
  // Enforce non-decreasing quantiles after rounding.
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

function runCase(caseObj: any): { trophies: { id: number; q: number[] }[] } {
  const atMs = Date.parse(caseObj.at);
  const targetMs = Date.parse(caseObj.target);
  const horizonDays = Math.max(0, (targetMs - atMs) / 86400000);
  const targetHours = (targetMs - T0) / 3600000;
  const history: any[] = caseObj.history ?? [];
  const trophies: any[] = caseObj.trophies ?? [];

  const rows = trophies.map((tr, idx) => {
    const reads: Read[] = [];
    let lastRate: number | null = null;
    for (const h of history) {
      const rates: any[] = h.rates ?? [];
      const r = rates[idx];
      if (r === null || r === undefined) continue;
      const rate = Number(r);
      if (!Number.isFinite(rate)) continue;
      lastRate = rate;
      reads.push({
        tHours: (Date.parse(h.at) - T0) / 3600000,
        y: logit(rate),
      });
    }
    // No usable reads at all: centre on 1% with a wide band.
    if (lastRate === null) {
      const sigma = 1.0;
      const c = logit(1);
      return { id: tr.id, q: Z.map((z) => round4(invLogit(c + z * sigma))) };
    }
    return { id: tr.id, q: forecastTrophy(reads, lastRate, horizonDays, targetHours) };
  });

  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const caseObj = JSON.parse(readFileSync(join(inputDir, f), "utf8"));
    const out = runCase(caseObj);
    const outName = f.replace(/\.in$/, ".out");
    writeFileSync(join(outputDir, outName), JSON.stringify(out));
  }
}

main();
