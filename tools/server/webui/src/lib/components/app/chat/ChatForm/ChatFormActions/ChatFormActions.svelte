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
	import { supportsAudio } from '$lib/stores/server.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import type { ChatUploadedFile } from '$lib/types/chat';

	interface Props {
		canSend?: boolean;
		className?: string;
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
		className = '',
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
	let shouldShowSubmitButton = $derived(!shouldShowRecordButton || hasAudioAttachments);

	let isSelectedModelInCache = $derived.by(() => {
		const currentModelId = selectedModelId();

		if (!currentModelId) return false;

		return modelOptions().some((option) => option.id === currentModelId);
	});
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<ChatFormActionFileAttachments class="mr-auto" {disabled} {onFileUpload} />

	<SelectorModel forceForegroundText={true} />

	{#if isLoading}
		<Button
			type="button"
			onclick={onStop}
			class="h-8 w-8 bg-transparent p-0 hover:bg-destructive/20"
		>
			<span class="sr-only">Stop</span>
			<Square class="h-8 w-8 fill-destructive stroke-destructive" />
		</Button>
	{:else}
		{#if shouldShowRecordButton}
			<ChatFormActionRecord {disabled} {isLoading} {isRecording} {onMicClick} />
		{/if}

		{#if shouldShowSubmitButton}
			<ChatFormActionSubmit
				{canSend}
				{disabled}
				{isLoading}
				tooltipLabel={isSelectedModelInCache
					? ''
					: 'Selected model is not available, please select another'}
				isModelAvailable={isSelectedModelInCache}
			/>
		{/if}
	{/if}
</div>
