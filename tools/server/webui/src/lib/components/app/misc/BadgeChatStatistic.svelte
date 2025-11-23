<script lang="ts">
	import { BadgeInfo } from '$lib/components/app';
	import type { Component } from 'svelte';

	interface Props {
		class?: string;
		icon: Component;
		value: string | number;
	}

	let { class: className = '', icon: Icon, value }: Props = $props();
</script>

<BadgeInfo class={className}>
	{#snippet icon()}
		<Icon class="h-3 w-3" />
	{/snippet}

	{value}
</BadgeInfo>
