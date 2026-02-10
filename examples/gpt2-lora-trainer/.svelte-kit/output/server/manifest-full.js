export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.C20lYnFV.js",app:"_app/immutable/entry/app.Do0iGl4s.js",imports:["_app/immutable/entry/start.C20lYnFV.js","_app/immutable/chunks/t0dpMB91.js","_app/immutable/chunks/Dg8bTafm.js","_app/immutable/chunks/NCSuz3cp.js","_app/immutable/entry/app.Do0iGl4s.js","_app/immutable/chunks/BxLj5Vw_.js","_app/immutable/chunks/Dg8bTafm.js","_app/immutable/chunks/97Z11XPK.js","_app/immutable/chunks/Bhr74y56.js","_app/immutable/chunks/Dnu1u84q.js","_app/immutable/chunks/NCSuz3cp.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
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
