/**
 * Single shared ExperimentClient instance for the manager pages.
 *
 * Putting it in its own module (in a .svelte.ts file so $state runes work)
 * means /manager and /manager/[id] talk to the same WebSocket and share
 * the same `experiments` reactive map. Switching between the two routes
 * doesn't tear down the connection or lose live state.
 *
 * The URL is configurable via the SERVER_URL constant (or override at
 * runtime by editing localStorage). For the dev workflow we assume the
 * Python server is on port 9883 of the same host the page is served from,
 * which lets the SvelteKit dev server (vite, port 5173) and the Python
 * server (uvicorn, port 9883) coexist on localhost.
 */

import { ExperimentClient } from './experiment-client.svelte';

function defaultUrl(): string {
  if (typeof window === 'undefined') return 'ws://localhost:9883/ws';
  // Same hostname as the page, hardcoded port. The browser will resolve
  // localhost / 127.0.0.1 / a LAN IP correctly depending on how the user
  // navigated to the dev page.
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';
  // Allow override from localStorage for the "I'm running the server on
  // a different machine" case.
  const override = window.localStorage.getItem('experiment_server_url');
  if (override) return override;
  return `${proto}//${host}:9883/ws`;
}

let _client: ExperimentClient | null = null;

export function getExperimentClient(): ExperimentClient {
  if (_client === null) {
    _client = new ExperimentClient(defaultUrl());
    _client.connect();
  }
  return _client;
}
