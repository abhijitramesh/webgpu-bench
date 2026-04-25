// Hugging Face OAuth + dataset push for the bench page (hosted mode).
//
// Flow:
//   1. On page load, the run controller calls `resumeHFSession()` to handle
//      the OAuth redirect (if the URL has the expected query params) and to
//      reuse any previously-stored access token.
//   2. Clicking [Sign in] calls `beginHFSignIn({ popup: true })` which opens
//      HF in a new window. After consent, HF redirects the popup back here;
//      the popup completes the exchange, persists the token to localStorage,
//      pings the opener via postMessage, and closes itself. The original
//      tab keeps any in-flight benchmark + accumulated results intact.
//   3. Clicking [Submit] calls `submitResultsToDataset()` which commits a
//      JSON file (one per machine-slug / browser / session) as a PR to the
//      leaderboard dataset.

import {
  oauthLoginUrl,
  oauthHandleRedirectIfPresent,
  uploadFile,
  whoAmI,
} from '@huggingface/hub';

import {
  HF_OAUTH_SCOPES,
  HF_DATASET_REPO,
  isHubConfigured,
} from './config.js';

// localStorage (not sessionStorage) so the popup and the original tab share
// the same token namespace, and a `storage` event fires in the opener when
// the popup completes sign-in.
export const HF_TOKEN_STORAGE_KEY = 'webgpu-bench:hfOauth';
export const HF_POPUP_DONE_MESSAGE = 'webgpu-bench:hf-signed-in';
// Marker written by the opener tab right before window.open(). The popup
// reads it on callback to know it's running inside a popup — `window.opener`
// is unreliable because HF's OAuth page sets a COOP header that severs the
// opener relationship in Chrome/Safari. Marker is cleared after callback.
const HF_POPUP_PENDING_KEY = 'webgpu-bench:hfPopupPending';
const HF_POPUP_PENDING_TTL_MS = 10 * 60 * 1000;

export function isHFPopupCallback() {
  if (typeof window === 'undefined') return false;
  if (window.opener && window.opener !== window) return true;
  try {
    const raw = localStorage.getItem(HF_POPUP_PENDING_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (!data?.ts || (Date.now() - data.ts) > HF_POPUP_PENDING_TTL_MS) {
      localStorage.removeItem(HF_POPUP_PENDING_KEY);
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

// ──────────────── session ────────────────

export async function resumeHFSession() {
  if (!isHubConfigured()) return null;

  // Completes the OAuth redirect if the current URL has the expected params.
  try {
    const res = await oauthHandleRedirectIfPresent();
    if (res) {
      const session = {
        accessToken: res.accessToken,
        expiresAt: res.accessTokenExpiresAt,
        userName: res.userInfo?.preferred_username || res.userInfo?.name || null,
        avatarUrl: res.userInfo?.picture || null,
        hubId: res.userInfo?.sub || null,
      };
      localStorage.setItem(HF_TOKEN_STORAGE_KEY, JSON.stringify(session));
      // Clean up the OAuth query params so a reload doesn't retry.
      const url = new URL(location.href);
      for (const k of ['code', 'state']) url.searchParams.delete(k);
      history.replaceState({}, '', url.toString());
      // Popup hand-off: tell the opener tab (if accessible) and close. The
      // opener also gets a `storage` event from the localStorage write —
      // postMessage is the fast path; storage event is the safety net for
      // when COOP severs window.opener.
      if (isHFPopupCallback()) {
        localStorage.removeItem(HF_POPUP_PENDING_KEY);
        if (window.opener && window.opener !== window && !window.opener.closed) {
          try { window.opener.postMessage({ type: HF_POPUP_DONE_MESSAGE }, location.origin); } catch { /* opener gone */ }
        }
        try { window.close(); } catch { /* not script-opened */ }
      }
      return session;
    }
  } catch (err) {
    console.warn('OAuth redirect handling failed:', err.message);
  }

  return readStoredSession();
}

function readStoredSession() {
  try {
    const raw = localStorage.getItem(HF_TOKEN_STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data.accessToken) return null;
    if (data.expiresAt && new Date(data.expiresAt).getTime() < Date.now()) {
      localStorage.removeItem(HF_TOKEN_STORAGE_KEY);
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

export async function beginHFSignIn({ popup = false } = {}) {
  if (!isHubConfigured()) {
    throw new Error('HF hub is not configured. Set HF_DATASET_REPO in run/config.js.');
  }
  // When served from an HF Space with `hf_oauth: true`, HF injects
  // `window.huggingface.variables` with OAUTH_CLIENT_ID + OAUTH_SCOPES.
  // @huggingface/hub needs the clientId passed explicitly.
  const injected = (globalThis.window?.huggingface?.variables) || {};
  const clientId = injected.OAUTH_CLIENT_ID;
  if (!clientId) {
    throw new Error(
      'No OAUTH_CLIENT_ID injected — is this page served from an HF Space with `hf_oauth: true`?',
    );
  }
  // OAuth 2.0 requires space-separated scopes. `@huggingface/hub` (as of
  // esm.sh current build) joins arrays with commas, producing `invalid_scope`.
  // Pass the injected string as-is so the library URL-encodes the spaces.
  const scopes = injected.OAUTH_SCOPES ?? HF_OAUTH_SCOPES.join(' ');
  const url = await oauthLoginUrl({
    clientId,
    scopes,
    redirectUrl: location.origin + location.pathname,
  });
  if (popup) {
    // Set marker BEFORE window.open so the popup callback can detect popup
    // mode even when COOP nullifies window.opener.
    try {
      localStorage.setItem(HF_POPUP_PENDING_KEY, JSON.stringify({ ts: Date.now() }));
    } catch { /* quota / disabled — best effort */ }
    const win = window.open(url, 'hf-oauth', 'popup=yes,width=520,height=720');
    if (!win) {
      try { localStorage.removeItem(HF_POPUP_PENDING_KEY); } catch { /* noop */ }
      const err = new Error('Popup blocked — allow popups for this site, or use full-page sign-in.');
      err.code = 'popup-blocked';
      throw err;
    }
    return { popup: win };
  }
  location.assign(url);
  return null;
}

export function signOutHF() {
  localStorage.removeItem(HF_TOKEN_STORAGE_KEY);
}

export async function fetchWhoAmI(token) {
  return whoAmI({ credentials: { accessToken: token } });
}

// ──────────────── dataset push ────────────────

/**
 * Push a single session's results to the dataset as a PR.
 * @param {object[]} results — array of stripped-shape benchmark records.
 * @param {object} opts
 * @param {string} opts.token — HF access token.
 * @param {string} opts.datasetRepo — "owner/name" of the dataset.
 * @param {string} opts.machineSlug
 * @param {string} opts.browser
 * @param {{name: string, hubId?: string, avatarUrl?: string}} [opts.submittedBy]
 *   — captured from the OAuth session and stamped onto each record so the
 *   dashboard can attribute submissions back to a HF user.
 * @returns {Promise<{ path: string, commit?: string, prUrl?: string }>}
 */
export async function submitResultsToDataset(results, {
  token,
  datasetRepo = HF_DATASET_REPO,
  machineSlug,
  browser,
  submittedBy = null,
}) {
  if (!datasetRepo) throw new Error('HF_DATASET_REPO is not configured.');
  if (!token) throw new Error('Not signed in — call beginHFSignIn() first.');
  if (!Array.isArray(results) || results.length === 0) {
    throw new Error('No results to submit.');
  }

  // Stamp attribution onto each record. Per-record (rather than file-level)
  // so the existing `Array.isArray(records)` parse path in
  // sync-from-dataset.mjs and submit-results.js keeps working unchanged.
  const stamped = submittedBy
    ? results.map(r => ({ ...r, submittedBy }))
    : results;

  const epoch = Date.now();
  const date = new Date().toISOString().slice(0, 10);
  const path = `runs/${date}/${machineSlug || 'unknown'}-${browser || 'browser'}-${epoch}.json`;
  const body = JSON.stringify(stamped, null, 2);
  const blob = new Blob([body], { type: 'application/json' });

  const res = await uploadFile({
    repo: { type: 'dataset', name: datasetRepo },
    credentials: { accessToken: token },
    file: { path, content: blob },
    commitTitle: `bench: ${machineSlug} / ${browser} / ${results.length} variants`,
  });

  return {
    path,
    commit: res?.commit?.oid || null,
    commitUrl: `https://huggingface.co/datasets/${datasetRepo}/blob/main/${path}`,
  };
}
