<script lang="ts">
  type Props = {
    /** Optional small uppercase label shown before the value. */
    label?: string;
    /** The displayed value (typically bold). */
    value: string | number;
    /** Render in a slightly smaller font for very long values (phase timings, rpc info). */
    mono?: boolean;
    /** Tooltip on hover. */
    title?: string;
  };

  let { label, value, mono = false, title = '' }: Props = $props();
</script>

<span class={mono ? 'text-[10.5px]' : ''} {title}>
  {label ? `${label} ` : ''}<b class="text-[rgba(0,0,0,0.84)] font-medium">{value}</b>
</span>
