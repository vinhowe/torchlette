<script lang="ts">
  import type { Snippet } from 'svelte';

  type Props = {
    title: string;
    /** Optional caption — supports either a plain string or a snippet for rich content. */
    caption?: string | Snippet;
    children: Snippet;
  };

  let { title, caption, children }: Props = $props();
</script>

<section>
  <h2 class="mb-1 text-[17px] font-semibold tracking-[-0.01em]">{title}</h2>
  {#if caption}
    <p class="mb-[14px] max-w-[680px] text-[13px] leading-[1.6] text-[rgba(0,0,0,0.54)]">
      {#if typeof caption === 'string'}
        {caption}
      {:else}
        {@render caption()}
      {/if}
    </p>
  {/if}
  {@render children()}
</section>
