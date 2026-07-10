<script lang="ts">
  import KatexBlock from "../optional/KatexBlock.svelte";
  import { twMerge } from "tailwind-merge";

  type $$Props = {
    forInputId?: string;
    class?: string;
    value: string;
  };
  let { forInputId, class: additionalClasses = "", value }: $$Props = $props();
</script>

<label
  for={forInputId}
  class={twMerge("block text-muted-foreground type-label", additionalClasses)}
>
  <KatexBlock text={value} />
</label>
