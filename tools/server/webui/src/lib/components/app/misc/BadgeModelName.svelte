<script lang="ts">
	import { Package } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { serverStore } from '$lib/stores/server.svelte';
	import { cn } from '$lib/components/ui/utils';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';

	interface Props {
		class?: string;
		onclick?: () => void;
		showTooltip?: boolean;
	}

	let { class: className = '', onclick, showTooltip = false }: Props = $props();

	let model = $derived(serverStore.modelName);
	let isModelMode = $derived(serverStore.isModelMode);
</script>

{#snippet badge()}
	<Badge
		variant="outline"
		class={cn(
			'text-xs',
			onclick ? 'cursor-pointer transition-colors hover:bg-foreground/20' : '',
			className
		)}
		{onclick}
	>
		<div class="icons mr-0.5 flex items-center gap-1.5">
			<Package class="h-3 w-3" />
		</div>

		<span class="block truncate">{model}</span>
	</Badge>
{/snippet}

{#if model && isModelMode}
	{#if showTooltip}
		<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
			<Tooltip.Trigger>
				{@render badge()}
			</Tooltip.Trigger>

			<Tooltip.Content>
				{onclick ? 'Click for model details' : 'Model name'}
			</Tooltip.Content>
		</Tooltip.Root>
	{:else}
		{@render badge()}
	{/if}
{/if}
