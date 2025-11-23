<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import type { DatabaseMessageExtra } from '$lib/types/database';
	import { ChatAttachmentPreview } from '$lib/components/app';
	import { formatFileSize } from '$lib/utils/formatters';
	import { getAttachmentTypeLabel } from '$lib/utils/attachment-type';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		// Either an uploaded file or a stored attachment
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		// For uploaded files
		preview?: string;
		name?: string;
		size?: number;
		textContent?: string;
	}

	let {
		open = $bindable(),
		onOpenChange,
		uploadedFile,
		attachment,
		preview,
		name,
		size,
		textContent
	}: Props = $props();

	let chatAttachmentPreviewRef: ChatAttachmentPreview | undefined = $state();

	let displayName = $derived(uploadedFile?.name || attachment?.name || name || 'Unknown File');

	let displaySize = $derived(uploadedFile?.size || size);

	let typeLabel = $derived(getAttachmentTypeLabel(uploadedFile, attachment));

	$effect(() => {
		if (open && chatAttachmentPreviewRef) {
			chatAttachmentPreviewRef.reset();
		}
	});
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden sm:w-auto sm:max-w-6xl">
		<Dialog.Header>
			<Dialog.Title class="pr-8">{displayName}</Dialog.Title>
			<Dialog.Description>
				{typeLabel}
				{#if displaySize}
					• {formatFileSize(displaySize)}
				{/if}
			</Dialog.Description>
		</Dialog.Header>

		<ChatAttachmentPreview
			bind:this={chatAttachmentPreviewRef}
			{uploadedFile}
			{attachment}
			{preview}
			name={displayName}
			{textContent}
		/>
	</Dialog.Content>
</Dialog.Root>
