<script lang="ts">
  import "../app.css";
  import ThemeProvider from "$lib/components/theme/ThemeProvider.svelte";

  const { children } = $props();
</script>

<ThemeProvider>
  {@render children()}
</ThemeProvider>
