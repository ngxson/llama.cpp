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
	import { config } from '$lib/stores/settings.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { activeMessages, usedModalities } from '$lib/stores/conversations.svelte';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
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
	let isRouter = $derived(isRouterMode());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let previousConversationModel: string | null = null;

	$effect(() => {
		if (conversationModel && conversationModel !== previousConversationModel) {
			previousConversationModel = conversationModel;
			modelsStore.selectModelByName(conversationModel);
		}
	});

	// Get active model ID for fetching props
	// Priority: user-selected model > conversation model (allows changing model mid-chat)
	let activeModelId = $derived.by(() => {
		if (!isRouter) return null;

		const options = modelOptions();

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	// State for model props (fetched from /props?model=<id>)
	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	// Fetch model props when active model changes
	$effect(() => {
		if (isRouter && activeModelId) {
			// Check if we already have cached props
			const cached = modelsStore.getModelProps(activeModelId);
			if (!cached) {
				// Fetch props for this model
				modelsStore.fetchModelProps(activeModelId).then(() => {
					// Trigger reactivity update
					modelPropsVersion++;
				});
			}
		}
	});

	// Derive modalities from active model (works for both MODEL and ROUTER mode)
	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion; // Trigger reactivity on props fetch
			return modelsStore.modelSupportsAudio(activeModelId);
		}
		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion; // Trigger reactivity on props fetch
			return modelsStore.modelSupportsVision(activeModelId);
		}
		return false;
	});

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !hasText && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

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

	let selectorModelRef: SelectorModel | undefined = $state(undefined);

	export function openModelSelector() {
		selectorModelRef?.open();
	}

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities: () => usedModalities(),
		onValidationFailure: async (previousModelId) => {
			if (previousModelId) {
				await modelsStore.selectModelById(previousModelId);
			}
		}
	});
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<ChatFormActionFileAttachments
		class="mr-auto"
		{disabled}
		{hasAudioModality}
		{hasVisionModality}
		{onFileUpload}
	/>

	<SelectorModel
		bind:this={selectorModelRef}
		currentModel={conversationModel}
		forceForegroundText={true}
		useGlobalSelection={true}
		onModelChange={handleModelChange}
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
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
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
