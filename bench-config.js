// Public configuration for the one-click bench page.
// Dataset name is NOT a secret. OAuth sign-in uses HF Spaces built-in OAuth
// (declared via `hf_oauth: true` in spaces/README.md) so there's no separate
// OAuth app to register.

export const HF_DATASET_REPO = 'abhijitramesh/webgpu-bench-leaderboard';

// Scopes must match the ones declared in spaces/README.md frontmatter.
export const HF_OAUTH_SCOPES = ['read-repos', 'write-repos'];

export function isHubConfigured() {
  // Space-OAuth works whenever the dataset is set. The OAuth client ID is
  // injected by HF at Space runtime and read by @huggingface/hub at sign-in.
  return HF_DATASET_REPO.length > 0;
}
