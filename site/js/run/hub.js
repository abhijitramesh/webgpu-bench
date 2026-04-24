// Hugging Face OAuth + dataset push for the bench page (hosted mode).
//
// Flow:
//   1. On page load, the run controller calls `resumeHFSession()` to handle
//      the OAuth redirect (if the URL has the expected query params) and to
//      reuse any previously-stored access token.
//   2. Clicking [Sign in] calls `beginHFSignIn()` which redirects the
//      browser to HF; after consent, HF redirects back to the page with
//      the OAuth response encoded in the URL. The next page load calls
//      `resumeHFSession()` again, completes the exchange, and stores the
//      token in sessionStorage.
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

const TOKEN_STORAGE_KEY = 'webgpu-bench:hfOauth';

// ──────────────── session ────────────────

export async function resumeHFSession() {
  if (!isHubConfigured()) return null;

  // Completes the OAuth redirect if the current URL has the expected params.
  try {
    const res = await oauthHandleRedirectIfPresent();
    if (res) {
      sessionStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify({
        accessToken: res.accessToken,
        expiresAt: res.accessTokenExpiresAt,
        userName: res.userInfo?.preferred_username || res.userInfo?.name || null,
      }));
      // Clean up the OAuth query params so a reload doesn't retry.
      const url = new URL(location.href);
      for (const k of ['code', 'state']) url.searchParams.delete(k);
      history.replaceState({}, '', url.toString());
      return readStoredSession();
    }
  } catch (err) {
    console.warn('OAuth redirect handling failed:', err.message);
  }

  return readStoredSession();
}

function readStoredSession() {
  try {
    const raw = sessionStorage.getItem(TOKEN_STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data.accessToken) return null;
    if (data.expiresAt && new Date(data.expiresAt).getTime() < Date.now()) {
      sessionStorage.removeItem(TOKEN_STORAGE_KEY);
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

export async function beginHFSignIn() {
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
  location.assign(url);
}

export function signOutHF() {
  sessionStorage.removeItem(TOKEN_STORAGE_KEY);
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
 * @returns {Promise<{ path: string, commit?: string, prUrl?: string }>}
 */
export async function submitResultsToDataset(results, {
  token,
  datasetRepo = HF_DATASET_REPO,
  machineSlug,
  browser,
}) {
  if (!datasetRepo) throw new Error('HF_DATASET_REPO is not configured.');
  if (!token) throw new Error('Not signed in — call beginHFSignIn() first.');
  if (!Array.isArray(results) || results.length === 0) {
    throw new Error('No results to submit.');
  }

  const epoch = Date.now();
  const date = new Date().toISOString().slice(0, 10);
  const path = `runs/${date}/${machineSlug || 'unknown'}-${browser || 'browser'}-${epoch}.json`;
  const body = JSON.stringify(results, null, 2);
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
