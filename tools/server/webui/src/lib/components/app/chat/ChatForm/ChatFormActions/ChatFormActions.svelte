<script lang="ts">
	import { Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		ChatFormActionFileAttachments,
		ChatFormActionRecord,
		ChatFormActionSubmit,
		SelectorModel
	} from '$lib/components/app';
	import { FileTypeCategory } from '$lib/enums';
	import { getFileTypeCategory } from '$lib/utils/file-type';
	import { supportsAudio } from '$lib/stores/props.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { modelOptions, selectedModelId, selectModelByName } from '$lib/stores/models.svelte';
	import { getConversationModel } from '$lib/stores/chat.svelte';
	import { activeMessages } from '$lib/stores/conversations.svelte';
	import { isRouterMode } from '$lib/stores/props.svelte';
	import type { ChatUploadedFile } from '$lib/types/chat';

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		hasText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: (fileType?: FileTypeCategory) => void;
		onMicClick?: () => void;
		onStop?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		hasText = false,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop
	}: Props = $props();

	let currentConfig = $derived(config());
	let hasAudioModality = $derived(supportsAudio());
	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !hasText && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	// Get model from conversation messages (last assistant message with model)
	let conversationModel = $derived(getConversationModel(activeMessages() as DatabaseMessage[]));

	// Sync selected model with conversation model when it changes
	// Only sync when conversation HAS a model - don't clear selection for new chats
	// to allow user to select a model before first message
	$effect(() => {
		if (conversationModel) {
			selectModelByName(conversationModel);
		}
	});

	let isRouter = $derived(isRouterMode());

	// Check if any model is selected (either from conversation or user selection)
	// In single MODEL mode, there's always a model available
	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());

	let isSelectedModelInCache = $derived.by(() => {
		// In single MODEL mode, model is always available
		if (!isRouter) return true;

		// Check if conversation model is available
		if (conversationModel) {
			return modelOptions().some((option) => option.model === conversationModel);
		}

		// Check if user-selected model is available
		const currentModelId = selectedModelId();
		if (!currentModelId) return false; // No model selected

		return modelOptions().some((option) => option.id === currentModelId);
	});

	// Determine tooltip message for submit button
	let submitTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}
		if (!isSelectedModelInCache) {
			return 'Selected model is not available, please select another';
		}
		return '';
	});

	// Ref to SelectorModel for programmatic opening
	let selectorModelRef: SelectorModel | undefined = $state(undefined);

	// Export function to open the model selector
	export function openModelSelector() {
		selectorModelRef?.open();
	}
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<ChatFormActionFileAttachments class="mr-auto" {disabled} {onFileUpload} />

	<SelectorModel
		bind:this={selectorModelRef}
		currentModel={conversationModel}
		forceForegroundText={true}
	/>

	{#if isLoading}
		<Button
			type="button"
			onclick={onStop}
			class="h-8 w-8 bg-transparent p-0 hover:bg-destructive/20"
		>
			<span class="sr-only">Stop</span>
			<Square class="h-8 w-8 fill-destructive stroke-destructive" />
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && hasModelSelected && isSelectedModelInCache}
			{disabled}
			{isLoading}
			tooltipLabel={submitTooltip}
			showErrorState={hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>
