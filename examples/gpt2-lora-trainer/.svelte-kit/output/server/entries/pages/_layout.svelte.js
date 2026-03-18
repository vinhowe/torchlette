function _layout($$renderer, $$props) {
  const { children } = $$props;
  $$renderer.push(`<div class="min-h-screen bg-slate-950">`);
  children($$renderer);
  $$renderer.push(`<!----></div>`);
}
export {
  _layout as default
};
