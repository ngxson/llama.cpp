<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ModelModality } from '$lib/enums/model';
	import { AttachmentType } from '$lib/enums/attachment';
	import type { DatabaseMessageExtra } from '$lib/types/database';
	import { ChatAttachmentPreview } from '$lib/components/app';
	import { formatFileSize } from '$lib/utils/file-preview';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		// Either an uploaded file or a stored attachment
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		// For uploaded files
		preview?: string;
		name?: string;
		type?: string;
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
		type,
		size,
		textContent
	}: Props = $props();

	let chatAttachmentPreviewRef: ChatAttachmentPreview | undefined = $state();

	let displayName = $derived(uploadedFile?.name || attachment?.name || name || 'Unknown File');

	let displayType = $derived(
		uploadedFile
			? uploadedFile.type
			: attachment?.type === AttachmentType.IMAGE
				? 'image'
				: attachment?.type === AttachmentType.TEXT
					? 'text'
					: attachment?.type === AttachmentType.AUDIO
						? attachment.mimeType || ModelModality.AUDIO
						: attachment?.type === AttachmentType.PDF
							? 'application/pdf'
							: type || 'unknown'
	);

	let displaySize = $derived(uploadedFile?.size || size);

	$effect(() => {
		if (open && chatAttachmentPreviewRef) {
			chatAttachmentPreviewRef.reset();
		}
	});
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden sm:w-auto sm:max-w-6xl">
		<Dialog.Header>
			<Dialog.Title>{displayName}</Dialog.Title>
			<Dialog.Description>
				{displayType}
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
			{name}
			{type}
			{textContent}
		/>
	</Dialog.Content>
</Dialog.Root>
