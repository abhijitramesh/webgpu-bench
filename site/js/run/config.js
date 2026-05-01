// Public configuration for the one-click bench page.
// Dataset name is NOT a secret. OAuth sign-in uses HF Spaces built-in OAuth
// (declared via `hf_oauth: true` in spaces/README.md) so there's no separate
// OAuth app to register.

// Single source of truth for the consistency-phase prompt. Both the
// interactive Run page (controller.js) and the runner.js-driven harness
// (harness.js) import this so CPU baselines and GPU forced-decode passes
// are always taken against the same input. Plain text on purpose — no
// chat-template wrapping, since this codebase benchmarks 10 different
// model families and each has its own template.
export const CONSISTENCY_PROMPT = 'Explain quantum computing in simple terms.';

export const HF_DATASET_REPO = 'abhijitramesh/webgpu-bench-leaderboard';

// Scopes must match the ones declared in spaces/README.md frontmatter.
export const HF_OAUTH_SCOPES = ['read-repos', 'write-repos'];

export function isHubConfigured() {
  // Space-OAuth works whenever the dataset is set. The OAuth client ID is
  // injected by HF at Space runtime and read by @huggingface/hub at sign-in.
  return HF_DATASET_REPO.length > 0;
}
