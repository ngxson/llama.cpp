<script lang="ts">
	import { Square, ArrowUp } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		BadgeModelName,
		ChatFormActionFileAttachments,
		ChatFormModelSelector,
		ChatFormActionRecord,
		DialogModelInformation
	} from '$lib/components/app';
	import { FileTypeCategory } from '$lib/enums/files';
	import { getFileTypeCategory } from '$lib/utils/file-type';
	import { isRouterMode, supportsAudio } from '$lib/stores/server.svelte';
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

	let inRouterMode = $derived(isRouterMode());
	let hasAudioModality = $derived(supportsAudio());
	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(hasAudioModality && !hasText && !hasAudioAttachments);
	let shouldShowSubmitButton = $derived(!shouldShowRecordButton || hasAudioAttachments);

	let showModelInfoDialog = $state(false);
</script>

<div class="flex w-full items-center gap-3 {className}">
	<ChatFormActionFileAttachments class="mr-auto" {disabled} {onFileUpload} />

	{#if !inRouterMode}
		<BadgeModelName
			class="clickable max-w-80"
			onclick={() => (showModelInfoDialog = true)}
			showTooltip
		/>
	{:else}
		<ChatFormModelSelector class="shrink-0" />
	{/if}

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
			<Button
				type="submit"
				disabled={!canSend || disabled || isLoading}
				class="h-8 w-8 rounded-full p-0"
			>
				<span class="sr-only">Send</span>
				<ArrowUp class="h-12 w-12" />
			</Button>
		{/if}
	{/if}
</div>

<DialogModelInformation
	bind:open={showModelInfoDialog}
	onOpenChange={(open) => (showModelInfoDialog = open)}
/>
