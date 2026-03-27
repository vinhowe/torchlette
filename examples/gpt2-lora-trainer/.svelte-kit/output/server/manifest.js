export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["datasets/aurelius.txt","datasets/austen.txt","datasets/lovecraft.txt"]),
	mimeTypes: {".txt":"text/plain"},
	_: {
		client: {start:"_app/immutable/entry/start.Gh3DlNQN.js",app:"_app/immutable/entry/app.B47v30Gr.js",imports:["_app/immutable/entry/start.Gh3DlNQN.js","_app/immutable/chunks/CXMw5cI8.js","_app/immutable/chunks/Cft5xwsY.js","_app/immutable/chunks/C679vkDn.js","_app/immutable/entry/app.B47v30Gr.js","_app/immutable/chunks/DtExVYyI.js","_app/immutable/chunks/Cft5xwsY.js","_app/immutable/chunks/BYO0Wanp.js","_app/immutable/chunks/DH20m1UJ.js","_app/immutable/chunks/D7odF2t9.js","_app/immutable/chunks/C679vkDn.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
