<script lang="ts">
	import { Clock, Gauge, WholeWord } from '@lucide/svelte';

	interface Props {
		predictedTokens: number;
		predictedMs: number;
	}

	let { predictedTokens, predictedMs }: Props = $props();

	let tokensPerSecond = $derived((predictedTokens / predictedMs) * 1000);
	let timeInSeconds = $derived((predictedMs / 1000).toFixed(2));
</script>

<span class="inline-flex items-center gap-1 rounded-sm bg-muted-foreground/15 px-1.5 py-0.75">
	<WholeWord class="h-3 w-3" />
	{predictedTokens} tokens
</span>

<span class="inline-flex items-center gap-1 rounded-sm bg-muted-foreground/15 px-1.5 py-0.75">
	<Clock class="h-3 w-3" />
	{timeInSeconds}s
</span>

<span class="inline-flex items-center gap-1 rounded-sm bg-muted-foreground/15 px-1.5 py-0.75">
	<Gauge class="h-3 w-3" />
	{tokensPerSecond.toFixed(2)} tokens/s
</span>
