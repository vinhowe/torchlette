<script lang="ts">
import "../app.css";

const { children } = $props();
</script>

<div class="min-h-screen bg-slate-950">
  {@render children()}
</div>
