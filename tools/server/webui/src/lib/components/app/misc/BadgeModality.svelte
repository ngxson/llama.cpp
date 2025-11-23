<script lang="ts">
	import { Eye, Mic } from '@lucide/svelte';
	import { ModelModality } from '$lib/enums/model';
	import { cn } from '$lib/components/ui/utils';

	interface Props {
		modalities: ModelModality[];
		class?: string;
	}

	let { modalities, class: className = '' }: Props = $props();

	function getModalityIcon(modality: ModelModality) {
		switch (modality) {
			case ModelModality.VISION:
				return Eye;
			case ModelModality.AUDIO:
				return Mic;
			default:
				return null;
		}
	}

	function getModalityLabel(modality: ModelModality): string {
		switch (modality) {
			case ModelModality.VISION:
				return 'Vision';
			case ModelModality.AUDIO:
				return 'Audio';
			default:
				return 'Unknown';
		}
	}
</script>

{#each modalities as modality, index (index)}
	{@const IconComponent = getModalityIcon(modality)}

	<span
		class={cn(
			'inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium',
			className
		)}
	>
		{#if IconComponent}
			<IconComponent class="h-3 w-3" />
		{/if}

		{getModalityLabel(modality)}
	</span>
{/each}
