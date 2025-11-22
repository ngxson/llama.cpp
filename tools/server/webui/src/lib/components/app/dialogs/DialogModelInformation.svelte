<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Table from '$lib/components/ui/table';
	import { BadgeModality } from '$lib/components/app';
	import { serverStore } from '$lib/stores/server.svelte';
	import { ChatService } from '$lib/services/chat';
	import type { ApiModelListResponse } from '$lib/types/api';
	import { Copy } from '@lucide/svelte';
	import { copyToClipboard } from '$lib/utils/copy';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let serverProps = $derived(serverStore.serverProps);
	let modalities = $derived(serverStore.supportedModalities);

	let modelsData = $state<ApiModelListResponse | null>(null);
	let isLoadingModels = $state(false);

	// Fetch models data when dialog opens
	$effect(() => {
		if (open && !modelsData) {
			loadModelsData();
		}
	});

	async function loadModelsData() {
		isLoadingModels = true;
		try {
			modelsData = await ChatService.getModels();
		} catch (error) {
			console.error('Failed to load models data:', error);
			// Set empty data to prevent infinite loading
			modelsData = { object: 'list', data: [] };
		} finally {
			isLoadingModels = false;
		}
	}

	// Format helpers
	function formatSize(sizeBytes: number | unknown): string {
		if (typeof sizeBytes !== 'number') return 'Unknown';

		// Convert to GB for better readability
		const sizeGB = sizeBytes / (1024 * 1024 * 1024);
		if (sizeGB >= 1) {
			return `${sizeGB.toFixed(2)} GB`;
		}

		// Convert to MB for smaller models
		const sizeMB = sizeBytes / (1024 * 1024);
		return `${sizeMB.toFixed(2)} MB`;
	}

	function formatParameters(params: number | unknown): string {
		if (typeof params !== 'number') return 'Unknown';
		if (params >= 1e9) {
			return `${(params / 1e9).toFixed(2)}B`;
		}
		if (params >= 1e6) {
			return `${(params / 1e6).toFixed(2)}M`;
		}
		if (params >= 1e3) {
			return `${(params / 1e3).toFixed(2)}K`;
		}
		return params.toString();
	}

	function formatNumber(num: number | unknown): string {
		if (typeof num !== 'number') return 'Unknown';
		return num.toLocaleString();
	}
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="@container z-9999 !max-w-[60rem] max-w-full">
		<style>
			@container (max-width: 56rem) {
				.resizable-text-container {
					max-width: calc(100vw - var(--threshold));
				}
			}
		</style>

		<Dialog.Header>
			<Dialog.Title>Model Information</Dialog.Title>
			<Dialog.Description>Current model details and capabilities</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-6 py-4">
			{#if isLoadingModels}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">Loading model information...</div>
				</div>
			{:else if modelsData && modelsData.data.length > 0}
				{@const modelMeta = modelsData.data[0].meta}

				{#if serverProps}
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head class="w-[10rem]">Model</Table.Head>

								<Table.Head>
									<div class="inline-flex items-center gap-2">
										<span
											class="resizable-text-container min-w-0 flex-1 truncate"
											style:--threshold="12rem"
										>
											{serverStore.modelName}
										</span>

										<Copy
											class="h-3 w-3 flex-shrink-0 cursor-pointer"
											aria-label="Copy model name to clipboard"
											onclick={() =>
												serverStore.modelName && copyToClipboard(serverStore.modelName)}
										/>
									</div>
								</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							<!-- Model Path -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">File Path</Table.Cell>

								<Table.Cell
									class="inline-flex h-10 items-center gap-2 align-middle font-mono text-xs"
								>
									<span
										class="resizable-text-container min-w-0 flex-1 truncate"
										style:--threshold="14rem"
									>
										{serverProps.model_path}
									</span>

									<Copy
										class="h-3 w-3 flex-shrink-0"
										aria-label="Copy model path to clipboard"
										onclick={() => copyToClipboard(serverProps.model_path)}
									/>
								</Table.Cell>
							</Table.Row>

							<!-- Context Size -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">Context Size</Table.Cell>
								<Table.Cell
									>{formatNumber(serverProps.default_generation_settings.n_ctx)} tokens</Table.Cell
								>
							</Table.Row>

							<!-- Training Context -->
							{#if modelMeta?.n_ctx_train}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Training Context</Table.Cell>
									<Table.Cell>{formatNumber(modelMeta.n_ctx_train)} tokens</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Model Size -->
							{#if modelMeta?.size}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Model Size</Table.Cell>
									<Table.Cell>{formatSize(modelMeta.size)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Parameters -->
							{#if modelMeta?.n_params}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Parameters</Table.Cell>
									<Table.Cell>{formatParameters(modelMeta.n_params)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Embedding Size -->
							{#if modelMeta?.n_embd}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Embedding Size</Table.Cell>
									<Table.Cell>{formatNumber(modelMeta.n_embd)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Vocabulary Size -->
							{#if modelMeta?.n_vocab}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Vocabulary Size</Table.Cell>
									<Table.Cell>{formatNumber(modelMeta.n_vocab)} tokens</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Vocabulary Type -->
							{#if modelMeta?.vocab_type}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Vocabulary Type</Table.Cell>
									<Table.Cell class="align-middle capitalize">{modelMeta.vocab_type}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Total Slots -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">Parallel Slots</Table.Cell>
								<Table.Cell>{serverProps.total_slots}</Table.Cell>
							</Table.Row>

							<!-- Modalities -->
							{#if modalities.length > 0}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Modalities</Table.Cell>
									<Table.Cell>
										<div class="flex flex-wrap gap-1">
											<BadgeModality {modalities} />
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Build Info -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">Build Info</Table.Cell>
								<Table.Cell class="align-middle font-mono text-xs"
									>{serverProps.build_info}</Table.Cell
								>
							</Table.Row>

							<!-- Chat Template -->
							{#if serverProps.chat_template}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Chat Template</Table.Cell>
									<Table.Cell class="py-10">
										<div class="max-h-120 overflow-y-auto rounded-md bg-muted p-4">
											<pre
												class="font-mono text-xs whitespace-pre-wrap">{serverProps.chat_template}</pre>
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}
						</Table.Body>
					</Table.Root>
				{/if}
			{:else if !isLoadingModels}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">No model information available</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
