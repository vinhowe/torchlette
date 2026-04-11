// Disable SSR for the manager page. The singleton ExperimentClient
// instantiates a WebSocket at module scope, which doesn't exist in Node.
export const ssr = false;
